"""Gestão de memória RAM do Apolo Zenith 1.9.

Fontes **reais** de informação (nada é inventado):

* ``/proc/meminfo``    – total, disponível e swap do sistema;
* ``/proc/self/status`` – RSS (``VmRSS``) do processo atual;
* ``/proc/self/statm`` – fallback para o RSS;
* cgroup v2 (``memory.max``/``memory.current``) ou v1
  (``memory.limit_in_bytes``/``memory.usage_in_bytes``) – **quando visíveis**
  (no Proot do Termux normalmente não são);
* ``resource.setrlimit(RLIMIT_AS)`` – limite duro por processo, aplicado de
  forma *best effort* (o kernel do Android pode recusar).

Decisões: ``ok`` / ``aviso`` / ``critico``.  Em ``critico`` o treino deve
pausar com checkpoint antes de prosseguir.
"""

from __future__ import annotations

import gc
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from resources.config import LimitesMemoria

_MB = 1024 ** 2


# --------------------------------------------------------------------------- #
# Leitura
# --------------------------------------------------------------------------- #
@dataclass
class LeituraMemoria:
    """Estado da memória em um instante."""

    ram_total_mb: float
    ram_disponivel_mb: float
    ram_usada_percent: float
    rss_mb: float
    swap_total_mb: float = 0.0
    swap_usada_mb: float = 0.0
    limite_cgroup_mb: Optional[float] = None
    uso_cgroup_mb: Optional[float] = None
    fonte: str = "/proc/meminfo"
    timestamp: float = 0.0

    @property
    def ram_usada_mb(self) -> float:
        return max(self.ram_total_mb - self.ram_disponivel_mb, 0.0)

    def para_dict(self) -> dict:
        return {
            "ram_total_mb": round(self.ram_total_mb, 1),
            "ram_disponivel_mb": round(self.ram_disponivel_mb, 1),
            "ram_usada_percent": round(self.ram_usada_percent, 1),
            "rss_mb": round(self.rss_mb, 1),
            "swap_total_mb": round(self.swap_total_mb, 1),
            "swap_usada_mb": round(self.swap_usada_mb, 1),
            "limite_cgroup_mb": self.limite_cgroup_mb,
            "uso_cgroup_mb": self.uso_cgroup_mb,
            "fonte": self.fonte,
            "timestamp": self.timestamp,
        }


class MemoryManager:
    """Monitora e protege a memória do processo de treino."""

    def __init__(self, limites: Optional[LimitesMemoria] = None) -> None:
        self.limites = limites or LimitesMemoria()
        self._ultima_leitura: Optional[LeituraMemoria] = None
        self._ultima_verificacao = 0.0
        self._intervalo_leitura_s = 5.0

    # ------------------------------------------------------------------ #
    # Fontes
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ler_meminfo() -> Optional[dict]:
        try:
            bruto = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        valores: dict[str, float] = {}
        for linha in bruto.splitlines():
            if ":" not in linha:
                continue
            chave, _, resto = linha.partition(":")
            campos = resto.split()
            if not campos:
                continue
            try:
                valores[chave.strip()] = float(campos[0])
            except ValueError:
                continue
        return valores

    @staticmethod
    def _ler_rss() -> float:
        """RSS do processo atual em MB."""
        try:
            for linha in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                if linha.startswith("VmRSS:"):
                    return float(linha.split()[1]) / 1024.0
        except (OSError, ValueError, IndexError):
            pass
        try:
            campos = Path("/proc/self/statm").read_text().split()
            paginas = int(campos[1])
            return paginas * os.sysconf("SC_PAGE_SIZE") / _MB
        except (OSError, ValueError, IndexError):
            return 0.0

    @staticmethod
    def _ler_cgroup() -> Tuple[Optional[float], Optional[float]]:
        """Limite e uso do cgroup de memória, quando visíveis."""
        # cgroup v2
        try:
            limite_txt = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            atual_txt = Path("/sys/fs/cgroup/memory.current").read_text().strip()
            if limite_txt != "max":
                limite = int(limite_txt) / _MB
                uso = int(atual_txt) / _MB
                return limite, uso
        except (OSError, ValueError):
            pass
        # cgroup v1 (o Proot do Termux costuma listar o caminho mas sem permissão)
        for base in (
            Path("/sys/fs/cgroup/memory"),
            Path("/sys/fs/cgroup/memory/apps/uid_10468"),
        ):
            try:
                limite_txt = (base / "memory.limit_in_bytes").read_text().strip()
                uso_txt = (base / "memory.usage_in_bytes").read_text().strip()
                limite = int(limite_txt) / _MB
                uso = int(uso_txt) / _MB
                # Valores absurdos (> RAM física) não são limites reais.
                if limite > 1024 * 1024:
                    continue
                return limite, uso
            except (OSError, ValueError):
                continue
        return None, None

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def ler(self, forcar: bool = False) -> LeituraMemoria:
        """Lê o estado atual da memória."""
        agora = time.time()
        if (
            not forcar
            and self._ultima_leitura is not None
            and (agora - self._ultima_verificacao) < self._intervalo_leitura_s
        ):
            return self._ultima_leitura
        self._ultima_verificacao = agora

        info = self._ler_meminfo() or {}
        total_kb = info.get("MemTotal", 0.0)
        disponivel_kb = info.get("MemAvailable", info.get("MemFree", 0.0))
        swap_total_kb = info.get("SwapTotal", 0.0)
        swap_livre_kb = info.get("SwapFree", 0.0)

        total_mb = total_kb / 1024.0
        disponivel_mb = disponivel_kb / 1024.0
        usada_percent = 0.0
        if total_mb > 0:
            usada_percent = max(0.0, min(100.0, (1.0 - disponivel_mb / total_mb) * 100.0))

        limite_cg, uso_cg = self._ler_cgroup()
        fonte = "/proc/meminfo"
        if limite_cg is not None:
            fonte += "+cgroup"

        leitura = LeituraMemoria(
            ram_total_mb=total_mb,
            ram_disponivel_mb=disponivel_mb,
            ram_usada_percent=usada_percent,
            rss_mb=self._ler_rss(),
            swap_total_mb=swap_total_kb / 1024.0,
            swap_usada_mb=max(swap_total_kb - swap_livre_kb, 0.0) / 1024.0,
            limite_cgroup_mb=limite_cg,
            uso_cgroup_mb=uso_cg,
            fonte=fonte,
            timestamp=agora,
        )
        self._ultima_leitura = leitura
        return leitura

    def nivel(self, leitura: Optional[LeituraMemoria] = None) -> str:
        """Classifica a leitura: ``ok`` | ``aviso`` | ``critico``.

        Observação importante: a **porcentagem usada** do ``/proc/meminfo``
        inclui cache recuperável pelo kernel, então ela é tratada como
        *aviso* — não como motivo de pausa.  O que realmente interrompe o
        treino é a memória **disponível** abaixo do piso, o **RSS** acima do
        limite ou o **cgroup** perto do teto.
        """
        leitura = leitura or self.ler()
        limites = self.limites
        if leitura.rss_mb >= limites.rss_max_mb:
            return "critico"
        if leitura.ram_disponivel_mb < limites.ram_minima_disponivel_mb:
            return "critico"
        if leitura.limite_cgroup_mb and leitura.uso_cgroup_mb:
            if leitura.uso_cgroup_mb >= leitura.limite_cgroup_mb * 0.95:
                return "critico"
        if leitura.ram_usada_percent >= limites.ram_max_utilizacao_percent:
            return "aviso"
        if leitura.ram_disponivel_mb < limites.ram_minima_disponivel_mb + limites.margem_livre_mb:
            return "aviso"
        return "ok"

    def deve_pausar(self, leitura: Optional[LeituraMemoria] = None) -> Tuple[bool, str]:
        """Decide se o treino deve pausar por memória.  Devolve ``(pausar, motivo)``."""
        leitura = leitura or self.ler()
        nivel = self.nivel(leitura)
        if nivel != "critico":
            return False, ""
        motivos = []
        if leitura.rss_mb >= self.limites.rss_max_mb:
            motivos.append(
                f"RSS {leitura.rss_mb:.0f} MB >= limite {self.limites.rss_max_mb} MB"
            )
        if leitura.ram_disponivel_mb < self.limites.ram_minima_disponivel_mb:
            motivos.append(
                f"RAM disponível {leitura.ram_disponivel_mb:.0f} MB < "
                f"{self.limites.ram_minima_disponivel_mb} MB"
            )
        if leitura.limite_cgroup_mb and leitura.uso_cgroup_mb:
            if leitura.uso_cgroup_mb >= leitura.limite_cgroup_mb * 0.95:
                motivos.append(
                    f"cgroup {leitura.uso_cgroup_mb:.0f}/{leitura.limite_cgroup_mb:.0f} MB"
                )
        return True, "; ".join(motivos) or "memória crítica"

    # ------------------------------------------------------------------ #
    # Ações
    # ------------------------------------------------------------------ #
    def liberar(self) -> int:
        """Libera memória não usada (GC + cache da CUDA, se existir)."""
        coletados = gc.collect()
        try:
            import torch  # import tardio: o pacote não exige torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - defensivo
            pass
        return coletados

    def aplicar_rlimit(self, rss_max_mb: Optional[int] = None) -> bool:
        """Tenta impor um limite de espaço de memória virtual (RLIMIT_AS).

        Retorna ``True`` se o limite foi aplicado.  Em alguns kernels
        Android/Proot a chamada é recusada — nesse caso o retorno é ``False``
        e a proteção passa a ser apenas reativa (medição de RSS).
        """
        limite_mb = rss_max_mb or self.limites.rss_max_mb
        bytes_limite = int(limite_mb) * _MB
        try:
            suave, duro = resource.getrlimit(resource.RLIMIT_AS)
            if duro != resource.RLIM_INFINITY and duro < bytes_limite:
                bytes_limite = duro
            resource.setrlimit(resource.RLIMIT_AS, (bytes_limite, duro))
            return True
        except (ValueError, OSError):
            return False

    def snapshot(self) -> dict:
        """Snapshot serializável para logs e relatórios."""
        return self.ler(forcar=True).para_dict()
