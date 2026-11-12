"""Apolo Video — encoder temporal + projetor.

Implementação **real** (treinada e avaliável):

* ``VideoEncoder``  – codifica cada frame com o encoder visual e aplica um
  Transformer **temporal** (bidirecional) sobre a sequência de frames;
* ``VideoProjector``– MLP de alinhamento com o Language Core.

Os vídeos originais permanecem em ``datasets/video/raw``; os frames
extraídos ficam em ``datasets/video/frames`` (nunca sobrescritos).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from vision.encoder import VisionConfig, VisionEncoder


@dataclass
class VideoConfig:
    """Hiperparâmetros do encoder de vídeo."""

    max_frames: int = 8
    frame_size: int = 64
    amostra_fps: float = 1.0          # frames extraídos por segundo
    vision: VisionConfig = None       # type: ignore[assignment]
    n_camadas_temporais: int = 2
    n_head: int = 4
    dropout: float = 0.0
    dim_projetor: int = 128

    def __post_init__(self) -> None:
        if self.vision is None:
            self.vision = VisionConfig(image_size=self.frame_size)
        else:
            self.vision.image_size = self.frame_size

    def para_dict(self) -> Dict[str, Any]:
        dados = asdict(self)
        return dados


class TemporalAttention(nn.Module):
    """Atenção temporal bidirecional (nn.MultiheadAttention)."""

    def __init__(self, dim: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalizado = self.norm(x)
        saida, _ = self.attn(normalizado, normalizado, normalizado, need_weights=False)
        return x + saida


class BlocoTemporal(nn.Module):
    """Bloco temporal: atenção + MLP."""

    def __init__(self, dim: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TemporalAttention(dim, n_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VideoEncoder(nn.Module):
    """Encoder temporal sobre frames (reaproveita o encoder visual)."""

    def __init__(self, config: Optional[VideoConfig] = None) -> None:
        super().__init__()
        self.config = config or VideoConfig()
        self.frame_encoder = VisionEncoder(self.config.vision)
        dim = self.config.vision.dim
        self.pos_embed = nn.Parameter(torch.zeros(1, self.config.max_frames, dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocos = nn.ModuleList(
            [
                BlocoTemporal(dim, self.config.n_head, self.config.dropout)
                for _ in range(self.config.n_camadas_temporais)
            ]
        )
        self.norm_final = nn.LayerNorm(dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """``frames``: (B, T, 3, H, W) -> embedding (B, dim)."""
        b, t = frames.shape[:2]
        achatado = frames.reshape(b * t, *frames.shape[2:])
        vetores = self.frame_encoder(achatado).reshape(b, t, -1)
        vetores = vetores + self.pos_embed[:, :t]
        for bloco in self.blocos:
            vetores = bloco(vetores)
        return self.norm_final(vetores.mean(dim=1))


class VideoProjector(nn.Module):
    """MLP de alinhamento do embedding temporal."""

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


class ApoloVideoModel(nn.Module):
    """Encoder de vídeo + projetor de alinhamento."""

    def __init__(self, config: Optional[VideoConfig] = None) -> None:
        super().__init__()
        self.config = config or VideoConfig()
        self.encoder = VideoEncoder(self.config)
        self.projetor = VideoProjector(
            self.config.vision.dim, self.config.dim_projetor, self.config.dropout
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.projetor(self.encoder(frames))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
