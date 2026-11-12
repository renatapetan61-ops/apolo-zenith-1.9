"""Escalonadores de taxa de aprendizado do Apolo.

Implementa o esquema clássico de LLMs:

1. **Warmup linear** — sobe de ``min_lr`` até ``lr`` em ``warmup_steps``;
2. **Cosine decay** — decai suavemente até ``min_lr`` em ``max_steps``.

A mesma forma é usada no fine-tuning dos módulos multimodais.
"""

from __future__ import annotations

import math
from typing import List


class WarmupCosineScheduler:
    """Agendador com warmup linear e decaimento cosseno."""

    def __init__(
        self,
        optimizer,
        max_steps: int,
        warmup_steps: int = 100,
        min_lr: float = 1e-5,
        last_epoch: int = -1,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps deve ser positivo")
        if warmup_steps < 0:
            raise ValueError("warmup_steps não pode ser negativo")
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = min(warmup_steps, max_steps)
        self.min_lr = min_lr
        self.base_lrs: List[float] = [g["lr"] for g in optimizer.param_groups]
        self.last_epoch = last_epoch
        self.step_update(0)

    def lr_at(self, step: int) -> float:
        """Taxa de aprendizado para um passo ``step`` (baseado em 1-index)."""
        if step < self.warmup_steps:
            alpha = step / max(1, self.warmup_steps)
            return self.min_lr + (self.base_lrs[0] - self.min_lr) * alpha
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosseno = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lrs[0] - self.min_lr) * cosseno

    def step_update(self, step: int) -> float:
        """Aplica a taxa de aprendizado correspondente a ``step``."""
        lr = self.lr_at(step)
        for grupo in self.optimizer.param_groups:
            grupo["lr"] = lr
        self.last_epoch = step
        return lr

    def state_dict(self) -> dict:
        return {"last_epoch": self.last_epoch, "base_lrs": self.base_lrs}

    def load_state_dict(self, estado: dict) -> None:
        self.last_epoch = estado.get("last_epoch", -1)
        self.base_lrs = estado.get("base_lrs", self.base_lrs)
        self.step_update(max(0, self.last_epoch))
