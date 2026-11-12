"""Apolo Audio — encoder de mel-spectrogram + projetor.

Implementação **real** (treinada e avaliável),dimensionada para CPU:

* ``MelFrontend``   – calcula o mel-spectrogram da forma canônica
  (16 kHz, 25 ms de janela, 10 ms de salto, 80 bandas mel);
* ``MelEncoder``    – pilha convolucional 2D sobre o mel + projeção para
  uma sequência de embeddings;
* ``AudioProjector``– MLP de alinhamento com o Language Core.

Não há decoder de geração de voz: este módulo produz **representações** e
sustenta reconhecimento/associação áudio-texto.  Qualquer afirmação sobre
síntese de voz seria falsa.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AudioConfig:
    """Hiperparâmetros do encoder de áudio."""

    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 512
    hop_length: int = 160          # 10 ms
    win_length: int = 400          # 25 ms
    max_seconds: float = 3.0
    dim: int = 192
    n_layer: int = 3
    n_head: int = 4
    dropout: float = 0.0
    dim_projetor: int = 128
    rms_norm_eps: float = 1e-5

    @property
    def max_frames(self) -> int:
        return int(self.max_seconds * self.sample_rate / self.hop_length)

    def para_dict(self) -> Dict[str, Any]:
        return asdict(self)


def mel_frontend(
    onda: np.ndarray,
    config: Optional[AudioConfig] = None,
) -> np.ndarray:
    """Calcula o mel-spectrogram (log-comprimido) de uma forma de onda.

    ``onda``: float32 mono na taxa de ``config.sample_rate``.
    Devolve ``(n_mels, frames)`` em escala log.
    """
    config = config or AudioConfig()
    import librosa

    if onda.ndim > 1:
        onda = onda.mean(axis=1)
    espec = librosa.feature.melspectrogram(
        y=onda.astype(np.float32, copy=False),
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.n_mels,
        fmax=config.sample_rate // 2,
    )
    log = np.log(np.maximum(espec, 1e-10))
    return log.astype(np.float32)


class MelEncoder(nn.Module):
    """Encoder convolucional sobre o mel-spectrogram.

    A saída da pilha convolucional tem forma ``(B, C, M, T)``; a dimensão
    temporal vira a sequência de tokens e ``C * M`` vira a característica.
    """

    def __init__(self, config: Optional[AudioConfig] = None) -> None:
        super().__init__()
        self.config = config or AudioConfig()
        c = self.config
        self.conv = nn.Sequential(
            nn.Conv2d(1, c.dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, c.dim // 2),
            nn.GELU(),
            nn.Conv2d(c.dim // 2, c.dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, c.dim),
            nn.GELU(),
            nn.Conv2d(c.dim, c.dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, c.dim),
            nn.GELU(),
        )
        # Altura do mel após as 3 convoluções com stride 2.
        self.mel_saida = c.n_mels // 8
        self.proj_entrada = nn.Linear(c.dim * max(self.mel_saida, 1), c.dim)
        self.proj_pos = nn.Linear(c.dim, c.dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=c.dim,
                nhead=c.n_head,
                dim_feedforward=c.dim * 2,
                dropout=c.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=c.n_layer,
        )
        self.norm_final = nn.LayerNorm(c.dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """``mel``: (B, n_mels, frames) -> sequência (B, T, dim)."""
        x = mel.unsqueeze(1)                    # (B, 1, n_mels, frames)
        x = self.conv(x)                        # (B, dim, mel', frames')
        b, d, m, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, d * m)   # (B, T, dim*mel')
        x = self.proj_entrada(x)                # (B, T, dim)
        x = x + self.proj_pos(x)
        x = self.transformer(x)
        return self.norm_final(x)


class AudioProjector(nn.Module):
    """Projeta a sequência de embeddings de áudio para o espaço do LM."""

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


class ApoloAudioModel(nn.Module):
    """Encoder de áudio + projetor de alinhamento."""

    def __init__(self, config: Optional[AudioConfig] = None) -> None:
        super().__init__()
        self.config = config or AudioConfig()
        self.encoder = MelEncoder(self.config)
        self.projetor = AudioProjector(
            self.config.dim, self.config.dim_projetor, self.config.dropout
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Devolve o embedding de áudio agregado (média attentiva simples)."""
        sequencia = self.encoder(mel)
        return self.projetor(sequencia.mean(dim=1))

    def sequencia(self, mel: torch.Tensor) -> torch.Tensor:
        """Devolve a sequência projetada (B, T, dim_projetor)."""
        return self.projetor(self.encoder(mel))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def normalizar_mel(mel: np.ndarray) -> np.ndarray:
    """Normaliza por arquivo (média/desvio) — evita domínios numéricos distintos."""
    media = float(mel.mean())
    desvio = float(mel.std()) or 1.0
    return ((mel - media) / desvio).astype(np.float32)
