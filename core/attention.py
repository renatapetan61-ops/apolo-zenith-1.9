"""Auto-atenção causal multi-cabeça do Apolo Language Core.

Implementa:
  * Multi-Head Self-Attention com máscara causal;
  * RoPE (Rotary Positional Embeddings) aplicado em q e k;
  * Grouped-Query Attention (GQA) opcional para reduzir memória do KV Cache;
  * Integração com o :class:`~core.kv_cache.KVCache` para inferência rápida.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import ModelConfig
from core.embeddings import RotaryEmbedding
from core.kv_cache import KVCache


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repete cabeças de chave/valor para suportar GQA.

    ``x``: ``(B, n_kv_head, T, head_dim)`` -> ``(B, n_head, T, head_dim)``
    """
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)


class CausalSelfAttention(nn.Module):
    """Bloco de auto-atenção causal com cabeças rotacionais."""

    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.n_rep = self.n_head // self.n_kv_head
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=config.bias)

        self.rotary = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position=max(1024, config.block_size),
            theta=config.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Posições absolutas: com cache, continuamos a partir do comprimento atual.
        position_ids = torch.arange(
            position_offset, position_offset + t, device=x.device, dtype=torch.long
        )
        q, k = self.rotary.aplicar(q, k, position_ids)

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # during decoding com cache T=1 => attends to all cached positions.
        is_causal = t > 1
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, self.n_head * self.head_dim)
        return self.o_proj(y)
