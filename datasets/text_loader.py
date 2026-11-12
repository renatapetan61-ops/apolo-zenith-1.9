"""Pipeline de dados textuais do Apolo.

Responsável por:

* normalização de texto (Unicode NFC, remoção de caracteres de controle);
* remoção de dados corrompidos e de baixa qualidade;
* deduplicação exata e aproximada (simhash);
* remoção de padrões de informações pessoais (PII);
* divisão em treino/validação/teste;
* empacotamento em sequências fixas para o Transformer;
* dataset PyTorch pronto para o Trainer.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Limpeza
# --------------------------------------------------------------------------- #
_RE_GUTENBERG = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE | re.DOTALL)
_RE_GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE | re.DOTALL)
_RE_CARACTERES_CONTROLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_ESPACOS = re.compile(r"[ \t]+")
_RE_LINHAS_BRANCO = re.compile(r"\n{3,}")
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_RE_TELEFONE = re.compile(r"(?<!\d)(?:\+?55\s?)?(?:\(?\d{2}\)?[\s.-]?)?\d{4,5}[\s.-]?\d{4}(?!\d)")
_RE_CPF = re.compile(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_RE_WIKI_LINK = re.compile(r"\[\[([^\]|]*)(?:\|([^\]]*))?\]\]")
_RE_TAG_HTML = re.compile(r"<[^>]+>")
_RE_CABECALHO = re.compile(r"^\s*(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)


def normalizar_unicode(texto: str) -> str:
    """Normaliza para NFC e remove caracteres de controle."""
    texto = unicodedata.normalize("NFC", texto)
    return _RE_CARACTERES_CONTROLE.sub("", texto)


def remover_cabecalho_gutenberg(texto: str) -> str:
    """Remove o cabeçalho/rodapé padrão do Project Gutenberg."""
    m = _RE_GUTENBERG.search(texto)
    if m:
        texto = texto[m.end():]
    m2 = _RE_GUTENBERG_END.search(texto)
    if m2:
        texto = texto[: m2.start()]
    return texto


def _substituir_link_wiki(match: "re.Match[str]") -> str:
    """``[[alvo|rótulo]]`` -> ``rótulo``; ``[[alvo]]`` -> ``alvo``."""
    rotulo = match.group(2)
    if rotulo is not None:
        return rotulo
    return match.group(1)


def limpar_wikitexto(texto: str) -> str:
    """Conversão simplificada de wikitexto para texto plano.

    Preserva o texto dos cabeçalhos (``== Título ==`` -> ``Título``), remove
    templates e resolve links internos para o rótulo legível.
    """
    texto = _RE_CABECALHO.sub(r"\2", texto)
    texto = _RE_TEMPLATE.sub(" ", texto)
    texto = _RE_WIKI_LINK.sub(_substituir_link_wiki, texto)
    texto = _RE_TAG_HTML.sub(" ", texto)
    texto = re.sub(r"'{2,}", "", texto)
    return texto


def normalizar_texto(texto: str, fonte: str = "gutenberg") -> str:
    """Normaliza e normaliza espaços/linhas de um documento bruto."""
    if fonte == "gutenberg":
        texto = remover_cabecalho_gutenberg(texto)
    elif fonte == "wikisource":
        texto = limpar_wikitexto(texto)
    texto = normalizar_unicode(texto)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = _RE_ESPACOS.sub(" ", texto)
    texto = "\n".join(linha.strip() for linha in texto.split("\n"))
    texto = _RE_LINHAS_BRANCO.sub("\n\n", texto)
    return texto.strip()


# --------------------------------------------------------------------------- #
# PII
# --------------------------------------------------------------------------- #
def remover_pii(texto: str, mascarar_urls: bool = True) -> str:
    """Remove/mascara padrões típicos de informações pessoais."""
    texto = _RE_EMAIL.sub("[EMAIL]", texto)
    texto = _RE_CPF.sub("[CPF]", texto)
    texto = _RE_TELEFONE.sub("[TELEFONE]", texto)
    if mascarar_urls:
        texto = _RE_URL.sub("[URL]", texto)
    return texto


# --------------------------------------------------------------------------- #
# Qualidade
# --------------------------------------------------------------------------- #
def qualidade_texto(texto: str, min_chars: int = 200) -> Tuple[bool, str]:
    """Avalia a qualidade de um documento.

    Retorna ``(aprovado, motivo)``.
    """
    if len(texto) < min_chars:
        return False, "muito_curto"
    letras = sum(1 for c in texto if c.isalpha())
    if letras / max(1, len(texto)) < 0.45:
        return False, "baixa_proporcao_alfabetica"
    palavras = texto.split()
    if len(palavras) < 30:
        return False, "poucas_palavras"
    unicas = len(set(p.lower() for p in palavras))
    if unicas / max(1, len(palavras)) < 0.12:
        return False, "repeticao_excessiva"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Deduplicação (simhash)
# --------------------------------------------------------------------------- #
def simhash(texto: str, bits: int = 64, ngram: int = 5) -> int:
    """Calcula um simhash de 64 bits do texto (para deduplicação aproximada)."""
    tokens = texto.lower().split()
    if len(tokens) < ngram:
        shingles = [" ".join(tokens)] if tokens else [""]
    else:
        shingles = [" ".join(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)]
    vetor = [0] * bits
    for shingle in shingles:
        h = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(bits):
            vetor[i] += 1 if (h >> i) & 1 else -1
    resultado = 0
    for i in range(bits):
        if vetor[i] > 0:
            resultado |= 1 << i
    return resultado


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def deduplicar(
    documentos: Sequence[str],
    limiar: int = 3,
    max_comparacoes_por_candidato: int = 50,
) -> Tuple[List[str], int]:
    """Remove duplicatas exatas e aproximadas.

    Usa assinaturas de simhash agrupadas em blocos para evitar comparação
    quadrática completa.
    """
    vistos_exatos: set[str] = set()
    assinaturas: List[int] = []
    mantidos: List[str] = []
    removidos = 0
    for doc in documentos:
        hash_exato = hashlib.blake2b(doc.encode("utf-8"), digest_size=16).hexdigest()
        if hash_exato in vistos_exatos:
            removidos += 1
            continue
        vistos_exatos.add(hash_exato)
        assinatura = simhash(doc)
        duplicado = False
        comparacoes = 0
        for outra in assinaturas:
            comparacoes += 1
            if comparacoes > max_comparacoes_por_candidato:
                break
            if _hamming(assinatura, outra) <= limiar:
                duplicado = True
                break
        if duplicado:
            removidos += 1
            continue
        assinaturas.append(assinatura)
        mantidos.append(doc)
    return mantidos, removidos


# --------------------------------------------------------------------------- #
# Divisão
# --------------------------------------------------------------------------- #
def dividir_documentos(
    documentos: Sequence[str],
    proporcoes: Tuple[float, float, float] = (0.98, 0.01, 0.01),
    semente: int = 1337,
) -> Dict[str, List[str]]:
    """Divide os documentos em treino/validação/teste de forma determinística."""
    import random

    total = sum(proporcoes)
    if total <= 0:
        raise ValueError("proporções inválidas")
    proporcoes = tuple(p / total for p in proporcoes)
    indices = list(range(len(documentos)))
    random.Random(semente).shuffle(indices)
    n = len(indices)
    n_treino = int(n * proporcoes[0])
    n_validacao = int(n * proporcoes[1])
    treino = [documentos[i] for i in indices[:n_treino]]
    validacao = [documentos[i] for i in indices[n_treino : n_treino + n_validacao]]
    teste = [documentos[i] for i in indices[n_treino + n_validacao :]]
    return {"train": treino, "val": validacao, "test": teste}


# --------------------------------------------------------------------------- #
# Empacotamento em sequências
# --------------------------------------------------------------------------- #
def salvar_tokens(tokens: Sequence[int], caminho: str | Path) -> Path:
    """Salva a fluxo de tokens em binário uint16 (memmap-efficient)."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(tokens, dtype=np.uint16)
    arr.tofile(caminho)
    return caminho


def carregar_tokens(caminho: str | Path) -> np.ndarray:
    """Carrega o fluxo de tokens como ``numpy.uint16`` (memmap)."""
    return np.memmap(caminho, dtype=np.uint16, mode="r")


class PackedTokenDataset(Dataset):
    """Dataset de sequências consecutivas de tokens.

    Cada exemplo é ``(x, y)`` com ``x = tokens[i:i+B]`` e
    ``y = tokens[i+1:i+B+1]`` (deslocamento de um token para a previsão
    autoregressiva).
    """

    def __init__(self, caminho: str | Path, block_size: int) -> None:
        self.caminho = Path(caminho)
        self.dados = np.memmap(self.caminho, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.n = max(0, (len(self.dados) - 1) // block_size)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inicio = idx * self.block_size
        x = np.asarray(self.dados[inicio : inicio + self.block_size], dtype=np.int64)
        y = np.asarray(self.dados[inicio + 1 : inicio + self.block_size + 1], dtype=np.int64)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())

    @property
    def num_tokens(self) -> int:
        return int(len(self.dados))
