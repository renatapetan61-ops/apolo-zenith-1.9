"""Batch adaptativo para o Apolo Zenith 1.9.

Regras:

* o batch vive sempre entre ``batch_minimo`` (1) e ``batch_maximo`` (4);
* um **OOM** reduz o batch pela metade (mínimo 1) e entra em *cooldown* —
  não há repetição infinita de tentativas;
* após ``passos_estaveis`` passos bem-sucedidos, o batch pode subir um
  degrau (até o máximo), desde que a RAM disponível permita;
* todo evento é registrado em um histórico auditável.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from resources.config import LimitesBatch


@dataclass
class EventoBatch:
    """Registro de uma mudança de batch."""

    passo: int
    batch_anterior: int
    batch_novo: int
    motivo: str
    timestamp: float = 0.0

    def para_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdaptiveBatchController:
    """Controla o tamanho do lote durante o treinamento."""

    def __init__(
        self,
        limites: Optional[LimitesBatch] = None,
        batch_inicial: Optional[int] = None,
    ) -> None:
        self.limites = limites or LimitesBatch()
        inicial = int(batch_inicial or self.limites.batch_inicial)
        self.batch_inicial = max(self.limites.batch_minimo, min(inicial, self.limites.batch_maximo))
        self._batch = self.batch_inicial
        self._passos_desde_mudanca = 0
        self._passos_desde_oom = 0
        self._tentativas_oom = 0
        self._oom_no_minimo = 0
        self._em_cooldown = False
        self.historico: List[EventoBatch] = []
        #: nº de passos bem-sucedidos necessários para aumentar o batch
        self.passos_estaveis = 20
        #: nº máximo de OOMs consecutivos já no batch mínimo antes de abortar
        self.max_tentativas_oom = 3

    # ------------------------------------------------------------------ #
    # Consultas
    # ------------------------------------------------------------------ #
    @property
    def batch_atual(self) -> int:
        return self._batch

    @property
    def em_cooldown(self) -> bool:
        return self._em_cooldown

    @property
    def tentativas_oom(self) -> int:
        return self._tentativas_oom

    def pode_aumentar(self, ram_disponivel_mb: Optional[float] = None) -> bool:
        """Verifica se há margem para aumentar o batch."""
        if not self.limites.permitir_aumento_automatico:
            return False
        if self._batch >= self.limites.batch_maximo:
            return False
        if self._em_cooldown:
            return False
        if ram_disponivel_mb is not None:
            # Margem conservadora: cada passo extra de batch exige ~128 MB.
            if ram_disponivel_mb < 512:
                return False
        return True

    # ------------------------------------------------------------------ #
    # Eventos
    # ------------------------------------------------------------------ #
    def registrar_oom(self, passo: int, erro: Optional[BaseException] = None) -> EventoBatch:
        """Registra um OOM: reduz o batch e ativa o cooldown.

        Se o batch já está no mínimo e o limite de tentativas foi atingido,
        levanta ``MemoryError`` — o treino deve parar, não repetir para sempre.
        """
        self._tentativas_oom += 1
        self._passos_desde_oom = 0
        anterior = self._batch

        if self._batch > self.limites.batch_minimo:
            self._batch = max(self.limites.batch_minimo, self._batch // 2)
            self._oom_no_minimo = 0
            motivo = f"OOM detectado (tentativa {self._tentativas_oom}); batch reduzido"
        else:
            self._oom_no_minimo += 1
            motivo = (
                f"OOM detectado (tentativa {self._tentativas_oom}) com batch já no mínimo "
                f"({self._batch})"
            )

        evento = EventoBatch(
            passo=passo,
            batch_anterior=anterior,
            batch_novo=self._batch,
            motivo=motivo,
            timestamp=time.time(),
        )
        self.historico.append(evento)

        if self._batch <= self.limites.batch_minimo:
            self._em_cooldown = True
            if self._oom_no_minimo > self.max_tentativas_oom:
                raise MemoryError(
                    f"OOM repetido {self._oom_no_minimo}x mesmo com batch={self._batch}. "
                    "Treinamento interrompido para proteger o dispositivo."
                )
        return evento

    def registrar_sucesso(
        self,
        passo: int,
        ram_disponivel_mb: Optional[float] = None,
    ) -> Optional[EventoBatch]:
        """Registra um passo bem-sucedido; pode elevar o batch."""
        self._passos_desde_oom += 1
        self._passos_desde_mudanca += 1
        self._oom_no_minimo = 0

        if self._em_cooldown:
            if self._passos_desde_oom >= self.limites.cooldown_apos_oom_steps:
                self._em_cooldown = False
            else:
                return None

        if self._passos_desde_mudanca < self.passos_estaveis:
            return None
        if not self.pode_aumentar(ram_disponivel_mb):
            self._passos_desde_mudanca = 0
            return None

        anterior = self._batch
        self._batch = min(self.limites.batch_maximo, self._batch + 1)
        self._passos_desde_mudanca = 0
        evento = EventoBatch(
            passo=passo,
            batch_anterior=anterior,
            batch_novo=self._batch,
            motivo="batch elevado após passos estáveis",
            timestamp=time.time(),
        )
        self.historico.append(evento)
        return evento

    def reset(self) -> None:
        """Volta ao batch inicial (usado ao retomar de um checkpoint)."""
        self._batch = self.batch_inicial
        self._passos_desde_mudanca = 0
        self._passos_desde_oom = 0
        self._tentativas_oom = 0
        self._oom_no_minimo = 0
        self._em_cooldown = False

    def estado(self) -> Dict[str, Any]:
        return {
            "batch_atual": self._batch,
            "batch_inicial": self.batch_inicial,
            "batch_minimo": self.limites.batch_minimo,
            "batch_maximo": self.limites.batch_maximo,
            "em_cooldown": self._em_cooldown,
            "tentativas_oom": self._tentativas_oom,
            "eventos": [e.para_dict() for e in self.historico[-10:]],
        }
