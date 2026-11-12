"""Normalizações usedas pelo Apolo Language Core.

O Core utiliza RMSNorm (Root Mean Square Layer Normalization) com
normalização *pré*-transformer, que é numericamente mais estável e mais barata
que LayerNorm em CPUs de baixo consumo (caso dos dispositivos móveis).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm: normaliza pela raiz da média quadrática e reescala com gamma.

    Fórmula (para cada token):

    ``rms(x) = sqrt(mean(x^2) + eps)`` e ``y = gamma * x / rms(x)``

    A normalização é aplicada na dimensão de embedd (``dim=-1``).
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # Inicializado em 1: a identidade no início do treinamento.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()  # acumula em fp32 para estabilidade mesmo em bf16/fp16
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        x = x.to(dtype)
        return self.weight * x

    def extra_repr(self) -> str:  # pragma: no cover - apenas depuração
        return f"dim={self.weight.numel()}, eps={self.eps}"
