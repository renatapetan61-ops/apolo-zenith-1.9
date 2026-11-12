"""Vocabulário e tokens especiais do Apolo.

O vocabulário do tokenizer é dividido em duas famílias:

1. **Bytes** (256 IDs) — garante que qualquer texto, em qualquer idioma ou
   codificação, possa ser representado (fallback byte-level).  Isso é
   essencial para línguas com alfabetos ricos, como o português.
2. **Tokens especiais** — controles de formato e de modalidade multimodal
   (visão, áudio, vídeo, ferramentas).  Eles já existem no vocabulário desde o
   início do treinamento, o que evita que um modelo multimodal precise ser
   retreinado do zero quando novos módulos forem plugged no ecossistema.
"""

from __future__ import annotations

from typing import Dict, List

# Tokens especiais ordenados: a ordem define o ID (0..N-1) e portanto é
# parte do contrato de compatibilidade dos checkpoints.
SPECIAL_TOKEN_NAMES: List[str] = [
    "<|pad|>",          # preenchimento (padding)
    "<|unk|>",          # token desconhecido
    "<|bos|>",          # início de sequência
    "<|eos|>",          # fim de sequência
    "<|sep|>",          # separador de segmentos
    "<|im_start|>",     # início de turno de conversa
    "<|im_end|>",       # fim de turno de conversa
    "<|vision|>",       # início de uma região visual
    "<|vision_pad|>",   # preenchimento de patches visuais
    "<|audio|>",        # início de uma região de áudio
    "<|audio_pad|>",    # preenchimento de frames de áudio
    "<|video|>",        # início de uma região de vídeo
    "<|video_pad|>",    # preenchimento de frames de vídeo
    "<|tool_call|>",    # chamada de ferramenta
    "<|tool_result|>",  # resultado de ferramenta
    "<|tool|>",         # marcador genérico de ferramenta
]

SPECIAL_TOKENS: Dict[str, int] = {nome: idx for idx, nome in enumerate(SPECIAL_TOKEN_NAMES)}

DESCRICOES_TOKENS_ESPECIAIS: Dict[str, str] = {
    "<|pad|>": "Preenchimento de sequência; ignorado na perda.",
    "<|unk|>": "Token de fallback (em prática raro: o nível byte cobre tudo).",
    "<|bos|>": "Início de documento.",
    "<|eos|>": "Fim de documento / fim de geração.",
    "<|sep|>": "Separa segmentos de contexto.",
    "<|im_start|>": "Início de mensagem do usuário/sistema.",
    "<|im_end|>": "Fim de mensagem.",
    "<|vision|>": "Delimitador de entrada do Apolo Vision.",
    "<|vision_pad|>": "Padding de patches visuais.",
    "<|audio|>": "Delimitador de entrada do Apolo Audio.",
    "<|audio_pad|>": "Padding de frames de áudio.",
    "<|video|>": "Delimitador de entrada do Apolo Video.",
    "<|video_pad|>": "Padding de frames de vídeo.",
    "<|tool_call|>": "Marcador de chamada de ferramenta.",
    "<|tool_result|>": "Marcador de retorno de ferramenta.",
    "<|tool|>": "Marcador genérico de ferramenta.",
}


def ids_especiales() -> Dict[str, int]:
    """Retorna uma cópia do mapeamento nome -> id dos tokens especiais."""
    return dict(SPECIAL_TOKENS)


def nomes_especiales() -> List[str]:
    """Lista (cópia) dos nomes dos tokens especiais, em ordem de ID."""
    return list(SPECIAL_TOKEN_NAMES)
