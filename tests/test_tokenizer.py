"""Testes do tokenizer BPE do Apolo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenizer.tokenizer import BPETokenizer
from tokenizer.vocabulary import SPECIAL_TOKENS


@pytest.fixture(scope="module")
def tokenizer_treinado() -> BPETokenizer:
    textos = [
        "Era uma vez um rei que morava num castelo muito grande e velho.",
        "O rato roeu a roupa do rei de Roma, e o rei de Roma foi aoJudge.",
        "Era uma vez um rei que morava num castelo muito grande e velho.",
        "A快地 Child Said: 'Hello, world! This is a test of the tokenizer.'",
        "Programação em Python: def funcao(x): return x * 2 + 1",
        "A mathematician writes: f(x) = x^2 + 2x + 1 = (x+1)^2",
    ] * 20
    return BPETokenizer.train(textos, vocab_size=400, min_frequencia=1)


def test_roundtrip_texto_simples(tokenizer_treinado: BPETokenizer) -> None:
    texto = "Era uma vez um rei que morava num castelo."
    ids = tokenizer_treinado.encode(texto)
    assert len(ids) > 0
    assert tokenizer_treinado.decode(ids) == texto


def test_roundtrip_preserva_acentos_e_unicode(tokenizer_treinado: BPETokenizer) -> None:
    texto = "Ação, coração, ilusão, çedilha, ão, 日本, 🧠, naïve café."
    ids = tokenizer_treinado.encode(texto)
    assert tokenizer_treinado.decode(ids) == texto


def test_byte_fallback_cobre_todo_byte(tokenizer_treinado: BPETokenizer) -> None:
    """Todo byte isolado deve ser codificável (fallback byte-level)."""
    for b in range(256):
        ids = tokenizer_treinado.encode(chr(b))
        assert len(ids) >= 1


def test_tokens_especiais_reservados(tokenizer_treinado: BPETokenizer) -> None:
    for nome, idx in SPECIAL_TOKENS.items():
        assert tokenizer_treinado.token_id(nome) == idx
    assert tokenizer_treinado.vocab_size >= 256 + len(SPECIAL_TOKENS)


def test_tokens_especiais_apenas_com_permissao(tokenizer_treinado: BPETokenizer) -> None:
    texto = "antes<|vision|>depois"
    ids_sem = tokenizer_treinado.encode(texto)
    ids_com = tokenizer_treinado.encode(texto, allow_special=True)
    assert SPECIAL_TOKENS["<|vision|>"] not in ids_sem
    assert SPECIAL_TOKENS["<|vision|>"] in ids_com


def test_bos_e_eos(tokenizer_treinado: BPETokenizer) -> None:
    ids = tokenizer_treinado.encode("teste", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer_treinado.bos_id
    assert ids[-1] == tokenizer_treinado.eos_id


def test_save_load_preserva_vocabulario(tokenizer_treinado: BPETokenizer, tmp_path: Path) -> None:
    tokenizer_treinado.save(tmp_path)
    recarregado = BPETokenizer.load(tmp_path)
    assert recarregado.vocab_size == tokenizer_treinado.vocab_size
    texto = "O Language Core do Apolo está funcionando."
    assert recarregado.encode(texto) == tokenizer_treinado.encode(texto)
    assert recarregado.decode(recarregado.encode(texto)) == texto


def test_compressao_real(tokenizer_treinado: BPETokenizer) -> None:
    """O BPE deve reduzir o número de tokens em relação aos bytes crus."""
    texto = "Era uma vez um rei que morava num castelo muito grande e velho." * 10
    n_tokens = len(tokenizer_treinado.encode(texto))
    n_bytes = len(texto.encode("utf-8"))
    assert n_tokens < n_bytes / 2


def test_encode_dataset_concatena(tokenizer_treinado: BPETokenizer) -> None:
    docs = ["primeiro documento", "segundo documento"]
    fluxo = tokenizer_treinado.encode_dataset(docs)
    esperado = tokenizer_treinado.encode("primeiro documento", add_bos=True, add_eos=True)
    esperado += tokenizer_treinado.encode("segundo documento", add_bos=True, add_eos=True)
    assert fluxo == esperado
