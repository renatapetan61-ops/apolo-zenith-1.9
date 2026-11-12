"""Apolo Video — compreensão temporal de vídeo (Fase 6).

STATUS REAL: **implementado e treinado**.

O que existe:

* :class:`~video.encoder.VideoEncoder` — encoder temporal (Transformer
  bidirecional) sobre uma sequência de frames;
* :class:`~video.encoder.VideoProjector` — MLP de alinhamento com o LM;
* ``datasets/video_loader.py`` — extração de frames com **PyAV** (embute
  ffmpeg), por amostragem de intervalo (vídeos longos não são processados
  frame a frame);
* ``training/video_trainer.py` — treino com **ordem temporal** (a cabeça
  decide se a sequência foi embaralhada) + InfoNCE contra o título.

Limite explícito: sem decoder de linguagem, o modelo **não resume** vídeos em
linguagem natural.
"""

from __future__ import annotations

from core.status import StatusModulo
from video.encoder import (
    ApoloVideoModel,
    BlocoTemporal,
    VideoConfig,
    VideoEncoder,
    VideoProjector,
)

STATUS = StatusModulo(
    nome="Apolo Video",
    fase=6,
    implementado=True,
    descricao=(
        "Encoder temporal (Transformer sobre frames) com projetor para o "
        "Language Core e cabeça de ordem temporal, treinado com dados do "
        "Wikimedia Commons e avaliado por acurácia de ordenação. Sem decoder de "
        "linguagem: não gera resumos de vídeo."
    ),
)


def encoder_video(*args, **kwargs):
    """Atalho para :class:`~video.encoder.ApoloVideoModel`."""
    return ApoloVideoModel(*args, **kwargs)


__all__ = [
    "STATUS",
    "StatusModulo",
    "ApoloVideoModel",
    "VideoConfig",
    "VideoEncoder",
    "VideoProjector",
    "BlocoTemporal",
    "encoder_video",
]
