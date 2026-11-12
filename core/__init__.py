"""Apolo Language Core — núcleo neural compartilhado do ecossistema.

Exporta a configuração, o modelo de linguagem e os blocos de arquitetura.
"""

from core.config import ModelConfig
from core.language_model import ApoloLanguageModel

__all__ = ["ModelConfig", "ApoloLanguageModel"]
