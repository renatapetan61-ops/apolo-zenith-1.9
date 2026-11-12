"""Embeddings do Apolo Language Core: tokens e posições rotacionais (RoPE)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Tabela de embeddings treináveis para tokens discretos."""

    def __init__(self, vocab_size: int, n_embd: int, padding_idx: int | None = None) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].fill_(0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotaciona metade das dimensões: ``[-x2, x1]``.

    Convenção igual à usada no paper original de RoPE.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Codificação posicional rotacional (RoPE).

    A posição é codificada como uma rotação no plano complexo: em cada par de
    dimensões a rotação é angle ``pos * inv_freq``.  Isso permite que a
    atenção interprete distâncias relativas de forma extrapolável.
    """

    def __init__(self, head_dim: int, max_position: int = 8192, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim precisa ser par para aplicar RoPE")
        self.head_dim = head_dim
        self.max_position = max_position
        self.theta = theta

        freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        pos = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(pos, freq)  # (max_position, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_position, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Devolve ``(cos, sin)`` com shape ``(1, 1, T, head_dim)``.

        O formato permite o broadcasting sobre ``q``/``k`` de shape
        ``(B, n_head, T, head_dim)``.
        """
        max_pos = int(position_ids.max().item()) + 1
        if max_pos > self.max_position:
            # Cresce o cache de forma segura (buffers não entram no checkpoint).
            self._grow(max_pos)
        cos = self.cos_cached[:max_pos][position_ids].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:max_pos][position_ids].unsqueeze(0).unsqueeze(0)
        return cos, sin

    @torch.no_grad()
    def _grow(self, novo_max: int) -> None:
        novo_max = max(novo_max, self.max_position * 2)
        freq = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        pos = torch.arange(novo_max, dtype=torch.float32)
        freqs = torch.outer(pos, freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()
        self.max_position = novo_max

    def aplicar(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        """Aplica a rotação às consultas e chaves.

        O nome evita colisão com ``nn.Module.apply`` (que percorre submódulos).

        Shapes: ``q``/``k`` em ``(B, n_head, T, head_dim)``.
        """
        cos, sin = self(position_ids)
        q_out = q * cos + rotate_half(q) * sin
        k_out = k * cos + rotate_half(k) * sin
        return q_out, k_out
