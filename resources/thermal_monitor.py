"""Monitor térmico do Apolo Zenith 1.9.

Fontes de temperatura **reais** tentadas, em ordem:

1. ``/sys/class/thermal/thermal_zone*/temp`` (Linux/Termux) — em Proot do
   Termux o diretório costuma estar sem permissão; nesse caso a leitura
   falha e o monitor **não inventa** valores;
2. ``getprop`` com chaves térmicas do Android;
3. ``dumpsys thermalservice`` (quando disponível);
4. ``termux-thermal`` (Termux API, se instalado).

Se **nenhuma** fonte fornecer leitura confiável, o monitor entra em
``MODO_PREVENTIVO``: aplica pausas programadas e usa a **carga de CPU como
proxy** (claramente rotulado como proxy, nunca como temperatura).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from resources.config import LimitesTermicos


@dataclass
class LeituraTermica:
    """Resultado de uma leitura térmica."""

    temperatura_c: Optional[float]
    fonte: str
    disponivel: bool
    timestamp: float = 0.0
    detalhe: str = ""

    @property
    def proxy_carga(self) -> bool:
        """Indica se o valor é uma temperatura real (False = proxy)."""
        return self.disponivel and self.fonte.startswith("sysfs")


class ThermalMonitor:
    """Monitora a temperatura e decide pausas/retomadas."""

    def __init__(self, limites: Optional[LimitesTermicos] = None) -> None:
        self.limites = limites or LimitesTermicos()
        self._ultima_leitura: Optional[LeituraTermica] = None
        self._ultima_verificacao = 0.0
        self._pausa_desde: Optional[float] = None
        self._motivo_pausa: str = ""
        self._inicio_ciclo: Optional[float] = None
        self.fontes_disponiveis: list[str] = []

    # ------------------------------------------------------------------ #
    # Fontes de leitura
    # ------------------------------------------------------------------ #
    def _ler_sysfs(self) -> Optional[float]:
        """Lê as zonas térmicas do Linux, quando acessíveis."""
        base = Path("/sys/class/thermal")
        if not base.is_dir():
            return None
        try:
            zonas = sorted(base.glob("thermal_zone*"))
        except PermissionError:
            return None
        if not zonas:
            return None
        melhor: Optional[float] = None
        for zona in zonas:
            arquivo = zona / "temp"
            try:
                bruto = arquivo.read_text().strip()
            except (OSError, PermissionError):
                continue
            try:
                milli = float(bruto)
            except ValueError:
                continue
            celsius = milli / 1000.0
            # Valores absurdos indicam sensores não confiáveis.
            if not (0.0 <= celsius <= 120.0):
                continue
            tipo = ""
            try:
                tipo = (zona / "type").read_text().strip()
            except (OSError, PermissionError):
                pass
            preferida = self.limites.zona_preferida
            if preferida and zona.name == preferida:
                return celsius
            if melhor is None or celsius > melhor:
                melhor = celsius
        return melhor

    def _ler_getprop(self) -> Optional[float]:
        """Lê props térmicas do Android (raramente trazem valor)."""
        getprop = shutil.which("getprop")
        if not getprop:
            return None
        try:
            saida = subprocess.run(
                [getprop], capture_output=True, text=True, timeout=10
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        for linha in saida.splitlines():
            if "[persist.sys.cpu_thermal" in linha or "[ro.thermal" in linha:
                # Formato: [chave]: [valor] -> zona, não temperatura.
                return None
        return None

    def _ler_dumpsys(self) -> Optional[float]:
        """Tenta ``dumpsys thermalservice``."""
        dumpsys = shutil.which("dumpsys")
        if not dumpsys:
            return None
        try:
            saida = subprocess.run(
                [dumpsys, "thermalservice"], capture_output=True, text=True, timeout=10
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        for linha in saida.splitlines():
            for chave in ("mTemperature", "Temperature{"):
                if chave in linha:
                    try:
                        valor = float(linha.split("=", 1)[1].split(",")[0].strip())
                        if 0.0 <= valor <= 120.0:
                            return valor
                    except (ValueError, IndexError):
                        continue
        return None

    def _ler_termux_api(self) -> Optional[float]:
        """Tenta a API térmica do Termux (termux-thermal)."""
        binario = shutil.which("termux-thermal")
        if not binario:
            return None
        try:
            saida = subprocess.run(
                [binario], capture_output=True, text=True, timeout=10
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        try:
            import json

            dados = json.loads(saida)
            if isinstance(dados, dict):
                valor = dados.get("temperature")
                if isinstance(valor, (int, float)):
                    return float(valor)
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def ler(self, forcar: bool = False) -> LeituraTermica:
        """Realiza uma leitura térmica, respeitando o intervalo configurado."""
        agora = time.time()
        if (
            not forcar
            and self._ultima_leitura is not None
            and (agora - self._ultima_verificacao) < self.limites.intervalo_leitura_s
        ):
            return self._ultima_leitura

        self._ultima_verificacao = agora
        fontes = [
            ("sysfs", self._ler_sysfs()),
            ("dumpsys", self._ler_dumpsys()),
            ("termux-api", self._ler_termux_api()),
            ("getprop", self._ler_getprop()),
        ]
        self.fontes_disponiveis = [nome for nome, valor in fontes if valor is not None]
        for nome, valor in fontes:
            if valor is not None:
                leitura = LeituraTermica(
                    temperatura_c=valor,
                    fonte=nome,
                    disponivel=True,
                    timestamp=agora,
                    detalhe=f"leitura real via {nome}",
                )
                self._ultima_leitura = leitura
                return leitura

        leitura = LeituraTermica(
            temperatura_c=None,
            fonte="indisponivel",
            disponivel=False,
            timestamp=agora,
            detalhe=(
                "Nenhuma fonte térmica acessível neste ambiente "
                "(sem /sys/class/thermal, sem dumpsys, sem Termux API). "
                "Modo preventivo ativo; a carga de CPU é usada apenas como proxy."
            ),
        )
        self._ultima_leitura = leitura
        return leitura

    def nivel(self, leitura: Optional[LeituraTermica] = None) -> str:
        """Classifica a leitura: ``ok`` | ``aviso`` | ``critico`` | ``indisponivel``."""
        leitura = leitura or self.ler()
        if not leitura.disponivel or leitura.temperatura_c is None:
            return "indisponivel"
        if leitura.temperatura_c >= self.limites.temperatura_maxima_c:
            return "critico"
        if leitura.temperatura_c >= self.limites.temperatura_aviso_c:
            return "aviso"
        return "ok"

    def deve_pausar(self, leitura: Optional[LeituraTermica] = None) -> Tuple[bool, str]:
        """Decide se o treino deve pausar.  Devolve ``(pausar, motivo)``.

        * Com sensor real: pausa quando a temperatura >= ``temperatura_maxima_c``.
        * Sem sensor (modo preventivo): pausa **programada** ao fim de
          ``ciclo_trabalho_min`` minutos de trabalho contínuo.  A carga de CPU
          é apenas registrada como proxy — ela **não** dispara pausa, porque a
          carga elevada é justamente a natureza do treinamento.
        """
        leitura = leitura or self.ler()
        nivel = self.nivel(leitura)
        if nivel == "critico":
            return True, (
                f"temperatura {leitura.temperatura_c:.1f}°C >= "
                f"{self.limites.temperatura_maxima_c}°C (fonte: {leitura.fonte})"
            )
        if nivel == "indisponivel" and self.limites.usar_modo_preventivo:
            if self.ciclo_excedido():
                decorrido = (time.time() - (self._inicio_ciclo or time.time())) / 60.0
                return True, (
                    "modo preventivo (sem sensor térmico): pausa programada após "
                    f"{decorrido:.0f} min de trabalho contínuo "
                    f"(ciclo configurado: {self.limites.ciclo_trabalho_min:.0f} min)"
                )
        return False, ""

    def ciclo_excedido(self) -> bool:
        """``True`` se o ciclo de trabalho contínuo já ultrapassou o limite."""
        if self._inicio_ciclo is None:
            return False
        if self._pausa_desde is not None:
            return False
        decorrido = time.time() - self._inicio_ciclo
        return decorrido >= self.limites.ciclo_trabalho_min * 60.0

    def registrar_inicio_ciclo(self) -> None:
        """Marca o início de um ciclo de trabalho (usado no modo preventivo)."""
        self._inicio_ciclo = time.time()

    def carga_cpu_atual(self) -> Optional[float]:
        """Carga de CPU normalizada (0..1) — **proxy**, não é temperatura."""
        return _carga_cpu()

    def deve_retomar(self, leitura: Optional[LeituraTermica] = None) -> Tuple[bool, str]:
        """Decide se um treino pausado pode voltar."""
        leitura = leitura or self.ler()
        nivel = self.nivel(leitura)
        if nivel == "critico":
            return False, "temperatura ainda acima do limite crítico"
        if nivel == "indisponivel":
            # Sem sensor, retomamos após o intervalo de pausa preventiva.
            if self._pausa_desde is None:
                return True, ""
            decorrido = (time.time() - self._pausa_desde) / 60.0
            if decorrido >= self.limites.pausa_preventiva_min:
                return True, (
                    f"pausa preventiva concluída ({decorrido:.1f} min, sem sensor térmico)"
                )
            return False, (
                f"aguardando resfriamento: {decorrido:.1f}/{self.limites.pausa_preventiva_min} min"
            )
        if leitura.temperatura_c is not None and leitura.temperatura_c > self.limites.temperatura_resume_c:
            return False, (
                f"temperatura {leitura.temperatura_c:.1f}°C acima do limite de "
                f"retomada {self.limites.temperatura_resume_c}°C"
            )
        return True, ""

    def registrar_inicio_pausa(self, motivo: str) -> None:
        self._pausa_desde = time.time()
        self._motivo_pausa = motivo

    def registrar_fim_pausa(self) -> float:
        """Encerra a pausa e devolve a duração em segundos."""
        duracao = 0.0
        if self._pausa_desde is not None:
            duracao = time.time() - self._pausa_desde
        self._pausa_desde = None
        self._motivo_pausa = ""
        # O próximo ciclo de trabalho começa agora.
        self._inicio_ciclo = time.time()
        return duracao

    @property
    def em_pausa(self) -> bool:
        return self._pausa_desde is not None

    @property
    def motivo_pausa(self) -> str:
        return self._motivo_pausa


def _carga_cpu() -> Optional[float]:
    """Carga de CPU do sistema (1 minuto) via ``/proc/loadavg``."""
    try:
        campos = Path("/proc/loadavg").read_text().split()
        return float(campos[0]) / max(os.cpu_count() or 1, 1)
    except (OSError, ValueError, IndexError):
        return None
