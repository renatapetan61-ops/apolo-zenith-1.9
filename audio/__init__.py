"""Apolo Audio — processamento de áudio (Fase 5).

STATUS REAL: **implementado e treinado**.

O que existe:

* :func:`~audio.encoder.mel_frontend` — mel-spectrogram canônico
  (16 kHz, 25 ms/10 ms, 80 bandas mel) calculado com librosa;
* :class:`~audio.encoder.MelEncoder` — pilha convolucional 2D + Transformer
  sobre o espectrograma;
* :class:`~audio.encoder.AudioProjector` — MLP de alinhamento com o LM;
* ``training/audio_trainer.py`` — treino contrastivo (InfoNCE) contra as
  transcrições, medido por recuperação áudio↔texto.

Limite explícito: **não há decoder de geração de voz**. Este módulo produz
representações e sustenta o reconhecimento/associação de fala; afirmar que o
Apolo "gera voz" seria falso.
"""

from __future__ import annotations

from core.status import StatusModulo
from audio.encoder import (
    ApoloAudioModel,
    AudioConfig,
    AudioProjector,
    MelEncoder,
    mel_frontend,
    normalizar_mel,
)

STATUS = StatusModulo(
    nome="Apolo Audio",
    fase=5,
    implementado=True,
    descricao=(
        "Encoder de mel-spectrogram (CNN + Transformer) com projetor para o "
        "Language Core, treinado com perda contrastiva sobre LibriSpeech e "
        "avaliado por recuperação áudio↔texto. Sem decoder de voz: não gera fala."
    ),
)


def encoder_audio(*args, **kwargs):
    """Atalho para :class:`~audio.encoder.ApoloAudioModel`."""
    return ApoloAudioModel(*args, **kwargs)


__all__ = [
    "STATUS",
    "StatusModulo",
    "ApoloAudioModel",
    "AudioConfig",
    "AudioProjector",
    "MelEncoder",
    "mel_frontend",
    "normalizar_mel",
    "encoder_audio",
]
