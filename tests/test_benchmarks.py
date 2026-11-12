"""Testes do harness de benchmarks do Apolo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from core.config import ModelConfig
from core.language_model import ApoloLanguageModel
from evaluation.benchmarks import (
    avaliar_codigo,
    avaliar_matematica,
    avaliar_mult_escolha,
    avaliar_perda,
    carregar_jsonl,
    normalizar_exemplos,
)
from evaluation.metrics import acuracia_numerica, extrair_primeiro_numero
from tokenizer.tokenizer import BPETokenizer


@pytest.fixture(scope="module")
def artefatos():
    tokenizer = BPETokenizer.train(
        [
            "A resposta é 42. Porque 2 vezes 21 dá 42.",
            "A capital do Brasil é Brasília.",
            "def soma(a, b): return a + b",
            "Paris é a capital da França.",
        ] * 20,
        vocab_size=320,
        min_frequencia=1,
    )
    torch.manual_seed(3)
    modelo = ApoloLanguageModel(
        ModelConfig(vocab_size=tokenizer.vocab_size, n_layer=2, n_head=2, n_embd=32,
                    block_size=32, mlp_hidden=64)
    )
    modelo.eval()
    return modelo, tokenizer


def test_carregar_jsonl(tmp_path: Path) -> None:
    arquivo = tmp_path / "d.jsonl"
    arquivo.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert len(carregar_jsonl(arquivo)) == 2


def test_normalizacao_de_exemplos() -> None:
    exemplos = normalizar_exemplos(
        [{"task_id": "x1", "pergunta": "P?", "opcoes": ["a", "b"], "resposta": 1, "testes": ["assert True"]}]
    )
    assert exemplos[0].task_id == "x1"
    assert exemplos[0].prompt == "P?"
    assert exemplos[0].opcoes == ["a", "b"]
    assert exemplos[0].referencia == "1"
    assert exemplos[0].testes == ["assert True"]


def test_avaliacao_de_perda(artefatos) -> None:
    modelo, tokenizer = artefatos
    exemplos = normalizar_exemplos([{"pergunta": "A capital do Brasil é Brasília."}])
    resultado = avaliar_perda(modelo, tokenizer, exemplos)
    assert resultado["perda"] > 0
    assert resultado["perplexidade"] > 1
    assert resultado["n"] == 1


def test_perda_da_avaliacao_bate_com_a_perda_do_modelo(artefatos) -> None:
    """Regressão: a avaliação não pode estar deslocada em um token.

    A perda calculada pelo harness sobre um texto deve ser igual à perda
    autoregressiva padrão do modelo sobre os mesmos tokens.
    """
    modelo, tokenizer = artefatos
    texto = "A resposta é 42. Porque 2 vezes 21 dá 42."
    ids = tokenizer.encode(texto)
    entrada = torch.tensor([ids[:-1]], dtype=torch.long)
    alvos = torch.tensor([ids[1:]], dtype=torch.long)
    with torch.no_grad():
        logits, _ = modelo(entrada)
        esperado = torch.nn.functional.cross_entropy(logits[0].float(), alvos[0]).item()

    exemplos = normalizar_exemplos([{"pergunta": texto}])
    obtido = avaliar_perda(modelo, tokenizer, exemplos)["perda"]
    assert abs(obtido - esperado) < 1e-4


def test_avaliacao_multipla_escolha(artefatos) -> None:
    modelo, tokenizer = artefatos
    exemplos = normalizar_exemplos(
        [
            {"pergunta": "Qual é a capital do Brasil?", "opcoes": ["Rio", "Brasília"], "resposta": "B"},
            {"pergunta": "2+2?", "opcoes": ["4", "5"], "resposta": "A"},
        ]
    )
    resultado = avaliar_mult_escolha(modelo, tokenizer, exemplos)
    assert resultado["tarefa"] == "mult_escolha"
    assert 0.0 <= resultado["media"] <= 1.0


def test_avaliacao_matematica_executa_inferencia(artefatos) -> None:
    modelo, tokenizer = artefatos
    exemplos = normalizar_exemplos(
        [{"pergunta": "Quanto é 2 mais 2?", "resposta": "4"}, {"pergunta": "Quanto é 5 mais 5?", "resposta": "10"}]
    )
    resultado = avaliar_matematica(modelo, tokenizer, exemplos, max_new_tokens=8)
    assert resultado["tarefa"] == "matematica"
    assert len(resultado["amostras"]) == 2


def test_avaliacao_codigo_executa_testes(artefatos) -> None:
    """O harness de código deve rodar a execução real dos testes."""
    modelo, tokenizer = artefatos
    exemplos = normalizar_exemplos(
        [{"task_id": "c1", "prompt": "Escreva def f(): return 1", "testes": ["assert f() == 1"]}]
    )
    resultado = avaliar_codigo(modelo, tokenizer, exemplos, max_new_tokens=32)
    assert resultado["tarefa"] == "codigo"
    assert "pass@1" in resultado


def test_execucao_de_testes_detalha_erro():
    from evaluation.benchmarks import executar_testes

    assert executar_testes("x = 2", ["assert x == 2"]) is True
    assert executar_testes("x = 2", ["assert x == 3"]) is False
    assert executar_testes("x = 2", []) is False


def test_extracao_de_numero() -> None:
    assert extrair_primeiro_numero("Resposta: 42 reais") == "42"
    assert extrair_primeiro_numero("3,5 kg") == "3.5"
    assert extrair_primeiro_numero("sem números") is None
    assert acuracia_numerica("Resposta: 42", "42") == 1.0
