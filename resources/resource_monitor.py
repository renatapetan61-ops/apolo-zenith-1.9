"""Orquestrador de estados de execução do Apolo Zenith 1.9.

Combina os três guardas (térmico, memória e armazenamento) em uma única
máquina de estados que o loop de treino consulta a cada passo:

``IDLE`` → ``RUNNING`` → (``WARNING``) → ``PAUSED_THERMAL`` /
``PAUSED_MEMORY`` / ``PAUSED_STORAGE`` → ``RECOVERING`` → ``RUNNING`` →
``COMPLETED`` / ``FAILED``

Todas as transições são registradas em ``logs/resource_events.jsonl`` para
auditoria posterior.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from resources.config import ConfigRecursos
from resources.memory_manager import LeituraMemoria, MemoryManager
from resources.storage_manager import LeituraArmazenamento, StorageManager
from resources.thermal_monitor import LeituraTermica, ThermalMonitor


class EstadoMonitor(str, Enum):
    """Estados possíveis do monitor de recursos."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    WARNING = "WARNING"
    PAUSED_THERMAL = "PAUSED_THERMAL"
    PAUSED_MEMORY = "PAUSED_MEMORY"
    PAUSED_STORAGE = "PAUSED_STORAGE"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    @property
    def pausado(self) -> bool:
        return self in (
            EstadoMonitor.PAUSED_THERMAL,
            EstadoMonitor.PAUSED_MEMORY,
            EstadoMonitor.PAUSED_STORAGE,
        )


@dataclass
class DecisaoMonitor:
    """Resultado de uma verificação de recursos."""

    estado: EstadoMonitor
    deve_pausar: bool
    motivo: str = ""
    leitura_termica: Optional[LeituraTermica] = None
    leitura_memoria: Optional[LeituraMemoria] = None
    leitura_armazenamento: Optional[LeituraArmazenamento] = None
    timestamp: float = 0.0
    detalhes: Dict[str, Any] = field(default_factory=dict)

    @property
    def pode_prosseguir(self) -> bool:
        return not self.deve_pausar

    def para_dict(self) -> Dict[str, Any]:
        return {
            "estado": self.estado.value,
            "deve_pausar": self.deve_pausar,
            "motivo": self.motivo,
            "timestamp": self.timestamp,
            "termico": (
                {
                    "temperatura_c": self.leitura_termica.temperatura_c,
                    "fonte": self.leitura_termica.fonte,
                    "disponivel": self.leitura_termica.disponivel,
                }
                if self.leitura_termica
                else None
            ),
            "memoria": self.leitura_memoria.para_dict() if self.leitura_memoria else None,
            "armazenamento": (
                self.leitura_armazenamento.para_dict() if self.leitura_armazenamento else None
            ),
            "detalhes": self.detalhes,
        }


class ResourceMonitor:
    """Coordena os guardas de recurso e mantém a máquina de estados."""

    def __init__(
        self,
        config: Optional[ConfigRecursos] = None,
        raiz: str | Path = ".",
        log_eventos: Optional[str | Path] = None,
    ) -> None:
        self.config = config or ConfigRecursos()
        self.raiz = Path(raiz)
        self.termico = ThermalMonitor(self.config.termicos)
        self.memoria = MemoryManager(self.config.memoria)
        self.armazenamento = StorageManager(self.config.armazenamento, raiz=self.raiz)
        self.estado = EstadoMonitor.IDLE
        self._inicio_pausa: Optional[float] = None
        self._motivo_pausa = ""
        self._total_pausas = 0.0
        self._historico: list[Dict[str, Any]] = []
        caminho_log = log_eventos or self.config.log_eventos
        self.log_eventos = Path(caminho_log)
        if not self.log_eventos.is_absolute():
            self.log_eventos = self.raiz / self.log_eventos

    # ------------------------------------------------------------------ #
    # Registro de eventos
    # ------------------------------------------------------------------ #
    def registrar_evento(self, evento: str, **campos: Any) -> None:
        """Acrescenta uma linha JSON ao log de eventos de recurso."""
        registro = {
            "timestamp": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "evento": evento,
            "estado": self.estado.value,
            **campos,
        }
        self._historico.append(registro)
        try:
            self.log_eventos.parent.mkdir(parents=True, exist_ok=True)
            with self.log_eventos.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(registro, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _transicionar(self, novo: EstadoMonitor, motivo: str = "") -> None:
        if novo == self.estado:
            return
        anterior = self.estado
        self.estado = novo
        self.registrar_evento("transicao", de=anterior.value, para=novo.value, motivo=motivo)

    # ------------------------------------------------------------------ #
    # Verificação
    # ------------------------------------------------------------------ #
    def verificar(self, forcar: bool = False) -> DecisaoMonitor:
        """Verifica os três recursos e devolve a decisão para o loop de treino."""
        agora = time.time()
        leitura_t = self.termico.ler(forcar=forcar)
        leitura_m = self.memoria.ler(forcar=forcar)
        leitura_s = self.armazenamento.ler(forcar=forcar)

        pausar_t, motivo_t = self.termico.deve_pausar(leitura_t)
        pausar_m, motivo_m = self.memoria.deve_pausar(leitura_m)
        pausar_s, motivo_s = self.armazenamento.deve_pausar(leitura_s)

        estado = EstadoMonitor.RUNNING
        deve_pausar = False
        motivo = ""

        if pausar_t:
            estado = EstadoMonitor.PAUSED_THERMAL
            deve_pausar = True
            motivo = motivo_t
        elif pausar_m:
            estado = EstadoMonitor.PAUSED_MEMORY
            deve_pausar = True
            motivo = motivo_m
        elif pausar_s:
            estado = EstadoMonitor.PAUSED_STORAGE
            deve_pausar = True
            motivo = motivo_s
        elif (
            self.termico.nivel(leitura_t) == "aviso"
            or self.memoria.nivel(leitura_m) == "aviso"
            or self.armazenamento.nivel(leitura_s) == "aviso"
        ):
            estado = EstadoMonitor.WARNING

        # Retomada: está pausado e as condições voltaram ao normal.
        if not deve_pausar and self.estado.pausado:
            if self.estado is EstadoMonitor.PAUSED_THERMAL:
                pode, motivo_ret = self.termico.deve_retomar(leitura_t)
            else:
                # Pausa por memória/storage: retoma assim que as métricas normalizam.
                pode, motivo_ret = True, f"condições de {self.estado.value} normalizadas"
            if pode:
                duracao = self.termico.registrar_fim_pausa()
                self._total_pausas += duracao
                self._transicionar(EstadoMonitor.RECOVERING, motivo_ret)
                self.registrar_evento(
                    "retomada",
                    pausado_em=self.estado.value,
                    duracao_pausa_s=round(duracao, 1),
                    motivo=motivo_ret,
                )
                self._transicionar(EstadoMonitor.RUNNING, "condições normalizadas")
            else:
                estado = self.estado
                deve_pausar = True
                motivo = motivo_ret or self._motivo_pausa

        if deve_pausar and not self.estado.pausado:
            self._inicio_pausa = agora
            self._motivo_pausa = motivo
            self.termico.registrar_inicio_pausa(motivo)
            self.registrar_evento("pausa", motivo=motivo, estado_destino=estado.value)
        elif not deve_pausar and self.estado.pausado:
            # ainda não liberado pelo monitor térmico
            duracao = (agora - (self._inicio_pausa or agora))
            self.registrar_evento(
                "pausa_prolongada",
                estado=self.estado.value,
                duracao_s=round(duracao, 1),
                motivo=self._motivo_pausa,
            )

        self._transicionar(estado, motivo)

        decisao = DecisaoMonitor(
            estado=estado,
            deve_pausar=deve_pausar,
            motivo=motivo,
            leitura_termica=leitura_t,
            leitura_memoria=leitura_m,
            leitura_armazenamento=leitura_s,
            timestamp=agora,
            detalhes={
                "nivel_termico": self.termico.nivel(leitura_t),
                "nivel_memoria": self.memoria.nivel(leitura_m),
                "nivel_armazenamento": self.armazenamento.nivel(leitura_s),
                "fonte_termica": leitura_t.fonte,
                "sensor_termico_disponivel": leitura_t.disponivel,
            },
        )
        return decisao

    # ------------------------------------------------------------------ #
    # Ciclo de vida
    # ------------------------------------------------------------------ #
    def iniciar(self) -> None:
        self._transicionar(EstadoMonitor.RUNNING, "início do treinamento")
        self.termico.registrar_inicio_ciclo()
        self.registrar_evento("inicio")

    def concluir(self, motivo: str = "treino concluído") -> None:
        self._transicionar(EstadoMonitor.COMPLETED, motivo)
        self.registrar_evento("conclusao", motivo=motivo, tempo_pausado_s=round(self._total_pausas, 1))

    def falhar(self, motivo: str) -> None:
        self._transicionar(EstadoMonitor.FAILED, motivo)
        self.registrar_evento("falha", motivo=motivo)

    @property
    def tempo_total_pausado(self) -> float:
        return self._total_pausas

    def resumo(self) -> Dict[str, Any]:
        """Resumo serializável do monitor (para relatórios)."""
        decisao = self.verificar(forcar=True)
        return {
            "estado": decisao.estado.value,
            "detalhes": decisao.detalhes,
            "tempo_pausado_s": round(self._total_pausas, 1),
            "memoria": decisao.leitura_memoria.para_dict() if decisao.leitura_memoria else None,
            "armazenamento": (
                decisao.leitura_armazenamento.para_dict() if decisao.leitura_armazenamento else None
            ),
        }
