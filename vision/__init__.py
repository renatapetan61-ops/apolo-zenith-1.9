"""Apolo Vision — visão computacional (Fase 4).

STATUS REAL: **implementado e treinado**.

O que existe:

* :class:`~vision.encoder.VisionEncoder` — Vision Transformer compacto
  (patch embedding + Transformer bidirecional + token de classe);
* :class:`~vision.encoder.VisionProjector` — MLP que leva o embedding visual
  à dimensão do Language Core;
* :class:`~vision.encoder.ApoloVisionModel` — encoder + projetor;
* ``training/image_trainer.py`` — treino contrastivo (InfoNCE) real contra os
  embeddings de texto do Language Core;
* ``scripts/evaluate_multimodal.py`` — métrica de recuperação medida.

Limite explícito: o módulo produz **representações** alinhadas ao espaço
textual. **Não** há decoder de linguagem, portanto o Apolo **não descreve**
imagens em linguagem natural. Essa capacidade é declarada apenas quando
existir (e for testada) um decoder treinado.
"""

from __future__ import annotations

from core.status import StatusModulo
from vision.encoder import (
    ApoloVisionModel,
    PatchEmbedding,
    VisionConfig,
    VisionEncoder,
    VisionProjector,
)

STATUS = StatusModulo(
    nome="Apolo Vision",
    fase=4,
    implementado=True,
    descricao=(
        "Vision Transformer compacto com patch embeddings, projetor para o "
        "Language Core e treino contrastivo (InfoNCE) medido por recuperação "
        "imagem↔texto. Sem decoder de linguagem: não gera descrições de imagens."
    ),
)


def encoder_visual(*args, **kwargs):
    """Atalho para :class:`~vision.encoder.ApoloVisionModel`."""
    return ApoloVisionModel(*args, **kwargs)


__all__ = [
    "STATUS",
    "StatusModulo",
    "ApoloVisionModel",
    "VisionConfig",
    "VisionEncoder",
    "VisionProjector",
    "PatchEmbedding",
    "encoder_visual",
]
