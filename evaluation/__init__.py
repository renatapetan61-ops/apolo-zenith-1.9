"""Infraestrutura de avaliação do Apolo Zenith 1.9."""

from evaluation.benchmarks import (
    avaliar_benchmark,
    avaliar_codigo,
    avaliar_matematica,
    avaliar_mult_escolha,
    avaliar_perda,
    carregar_jsonl,
    normalizar_exemplos,
)
from evaluation.metrics import (
    acuracia_numerica,
    acuracia_token,
    agregar_metricas,
    exact_match,
    f1_token,
    pass_at_k,
    perplexity,
)

__all__ = [
    "avaliar_benchmark",
    "avaliar_codigo",
    "avaliar_matematica",
    "avaliar_mult_escolha",
    "avaliar_perda",
    "carregar_jsonl",
    "normalizar_exemplos",
    "acuracia_numerica",
    "acuracia_token",
    "agregar_metricas",
    "exact_match",
    "f1_token",
    "pass_at_k",
    "perplexity",
]
