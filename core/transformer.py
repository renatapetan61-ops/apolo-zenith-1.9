"""Blocos do Transformer decoder-only do Apolo Language Core.

Arquitetura: pré-normalização (RMSNorm), atenção causal multi-cabeça com
RoPE, MLP SwiGLU (SiLU gate) e conexões residuais.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attention import CausalSelfAttention
from core.config import ModelConfig
from core.kv_cache import KVCache
from core.normalization import RMSNorm


class SwiGLU(nn.Module):
    """MLP com gating: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, n_embd: int, hidden: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, hidden, bias=bias)
        self.up_proj = nn.Linear(n_embd, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Um bloco decoder: atenção + MLP com normalização pré-bloco."""

    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx=layer_idx)
        self.mlp = SwiGLU(config.n_embd, config.mlp_hidden, config.dropout, config.bias)
        self.norm1 = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.norm2 = RMSNorm(config.n_embd, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cache=cache, position_offset=position_offset)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Pilha de blocos Transformer com normalização final."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.n_layer)]
        )
        self.norm_final = RMSNorm(config.n_embd, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cache=cache, position_offset=position_offset)
        return self.norm_final(x)
