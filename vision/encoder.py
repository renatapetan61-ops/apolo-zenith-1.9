"""Apolo Vision — encoder visual compacto (ViT) + projetor.

Implementação **real** (treinada e avaliável), dimensionada para CPU de
celular:

* ``PatchEmbedding``  – divide a imagem em patches 8x8 (ou 16x16);
* ``VisionEncoder``   – Transformer **bidirecional** (atenção sem máscara
  causal) sobre os patches + token de classe;
* ``VisionProjector`` – MLP que leva o embedding visual até a dimensão do
  Language Core, permitindo o alinhamento multimodal.

O encoder **não** substitui o Language Core: ele produz embeddings que são
alinhados aos embeddings textuais por um projetor treinado (ver
``training/image_trainer.py``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VisionConfig:
    """Hiperparâmetros do encoder visual."""

    dim: int = 192
    n_layer: int = 4
    n_head: int = 4
    patch_size: int = 8
    image_size: int = 64
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    dim_projetor: int = 128          # dimensão do Language Core (n_embd)
    rms_norm_eps: float = 1e-5

    @property
    def num_patches(self) -> int:
        lado = self.image_size // self.patch_size
        return lado * lado

    def para_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RMSNorm(nn.Module):
    """Normalização por RMS (usada também no Language Core)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.peso = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).type_as(x) * self.peso


class AtencaoBidirecional(nn.Module):
    """Auto-atenção sem máscara causal (usada na visão)."""

    def __init__(self, dim: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % n_head != 0:
            raise ValueError("dim precisa ser divisível por n_head")
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.o_proj(y)


class BlocoVision(nn.Module):
    """Bloco Transformer pré-normalizado."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        hidden = int(config.dim * config.mlp_ratio)
        self.norm1 = RMSNorm(config.dim, config.rms_norm_eps)
        self.attn = AtencaoBidirecional(config.dim, config.n_head, config.dropout)
        self.norm2 = RMSNorm(config.dim, config.rms_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):
    """Converte a imagem em uma sequência de embeddings de patch."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Conv2d(
            3, config.dim, kernel_size=config.patch_size, stride=config.patch_size
        )

    def forward(self, imagem: torch.Tensor) -> torch.Tensor:
        x = self.proj(imagem)                    # (B, dim, L, L)
        x = x.flatten(2).transpose(1, 2)          # (B, N, dim)
        return x


class VisionProjector(nn.Module):
    """MLP que projeta o embedding visual na dimensão do Language Core."""

    def __init__(self, dim_entrada: int, dim_saida: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_entrada, dim_entrada),
            nn.GELU(),
            nn.Linear(dim_entrada, dim_saida),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisionEncoder(nn.Module):
    """Vision Transformer compacto que devolve o embedding da imagem."""

    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        super().__init__()
        self.config = config or VisionConfig()
        self.patch_embed = PatchEmbedding(self.config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.config.num_patches + 1, self.config.dim)
        )
        self.blocks = nn.ModuleList([BlocoVision(self.config) for _ in range(self.config.n_layer)])
        self.norm_final = RMSNorm(self.config.dim, self.config.rms_norm_eps)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, imagem: torch.Tensor) -> torch.Tensor:
        """``imagem``: (B, 3, H, W) -> embedding (B, dim)."""
        x = self.patch_embed(imagem)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for bloco in self.blocks:
            x = bloco(x)
        x = self.norm_final(x)
        return x[:, 0]          # token de classe


class ApoloVisionModel(nn.Module):
    """Encoder visual + projetor de alinhamento com o Language Core."""

    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        super().__init__()
        self.config = config or VisionConfig()
        self.encoder = VisionEncoder(self.config)
        self.projetor = VisionProjector(
            self.config.dim, self.config.dim_projetor, self.config.dropout
        )

    def forward(self, imagem: torch.Tensor) -> torch.Tensor:
        return self.projetor(self.encoder(imagem))

    def embeddings_visuais(self, imagem: torch.Tensor) -> torch.Tensor:
        """Embedding visual **antes** do projetor (para análise)."""
        return self.encoder(imagem)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
