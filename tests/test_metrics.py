"""Testes das métricas de avaliação do Apolo."""

from __future__ import annotations

import math

import pytest

from evaluation.metrics import (
    acuracia_numerica,
    acuracia_token,
    agregar_metricas,
    exact_match,
    f1_token,
    pass_at_k,
    perplexity,
)


def test_perplexidade() -> None:
    assert perplexity(math.log(10)) == pytest.approx(10.0, rel=1e-6)
    assert perplexity(0.0) == 1.0


def test_exact_match_normalizado() -> None:
    assert exact_match("A resposta é 42.", "42") == 1.0
    assert exact_match("Paris", "paris") == 1.0
    assert exact_match("Paris", "Londres") == 0.0


def test_f1_token() -> None:
    assert f1_token("o gato preto", "o gato preto") == pytest.approx(1.0)
    # "o" e "preto" em comum -> precisão 2/3, recall 2/3 -> F1 = 2/3
    assert f1_token("o gato preto", "o cachorro preto") == pytest.approx(2 / 3)
    assert f1_token("", "abc") == 0.0


def test_acuracia_numerica() -> None:
    assert acuracia_numerica("A resposta é 42", "42") == 1.0
    assert acuracia_numerica("resultado: 3,5", "3.5") == 1.0
    assert acuracia_numerica("42", "43") == 0.0


def test_acuracia_token() -> None:
    assert acuracia_token([1, 2, 3], [1, 2, 4]) == pytest.approx(2 / 3)
    assert acuracia_token([], []) == 0.0


def test_pass_at_k() -> None:
    assert pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(10, 0, 1) == 0.0
    # Caso conhecido da literatura: n=10, c=1, k=1 -> 0.1
    assert pass_at_k(10, 1, 1) == pytest.approx(0.1)
    assert pass_at_k(5, 2, 2) == pytest.approx(0.7, rel=1e-6)


def test_pass_at_k_parametros_invalidos() -> None:
    with pytest.raises(ValueError):
        pass_at_k(0, 0, 1)
    with pytest.raises(ValueError):
        pass_at_k(5, 6, 1)


def test_agregar_metricas() -> None:
    resumo = agregar_metricas([0.0, 0.5, 1.0])
    assert resumo["media"] == pytest.approx(0.5)
    assert resumo["n"] == 3
    assert resumo["minimo"] == 0.0
    assert resumo["maximo"] == 1.0
    assert agregar_metricas([])["n"] == 0
