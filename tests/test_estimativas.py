"""Testes das estimativas de custo de treinamento."""

from __future__ import annotations

import pytest

from core.config import ModelConfig
from training.estimativas import estimar_treino
from training.trainer import TrainingConfig


def _config() -> tuple[ModelConfig, TrainingConfig]:
    return (
        ModelConfig(vocab_size=8192, n_layer=4, n_head=4, n_embd=128, block_size=256),
        TrainingConfig(batch_size=16, max_steps=700, grad_accum_steps=1),
    )


def test_estimativa_sem_medicao_nao_inventa_tempo() -> None:
    modelo, treino = _config()
    estimativa = estimar_treino(modelo, treino, tokens_por_segundo=None)
    assert estimativa.tempo_estimado_s is None
    assert any("Sem medição" in o for o in estimativa.observacoes)


def test_estimativa_com_medicao() -> None:
    modelo, treino = _config()
    estimativa = estimar_treino(modelo, treino, tokens_por_segundo=1200.0)
    tokens_total = 16 * 256 * 700
    assert estimativa.tokens_total == tokens_total
    assert estimativa.tempo_estimado_s == pytest.approx(tokens_total / 1200.0)
    assert estimativa.para_dict()["tempo_estimado_h"] > 0


def test_memoria_estimada() -> None:
    modelo, treino = _config()
    estimativa = estimar_treino(modelo, treino, tokens_por_segundo=1000.0)
    memoria = estimativa.memoria
    # Pesos fp32 + gradientes fp32 + 2 estados do AdamW fp32 = 16 bytes/param
    esperado = modelo.n_parameters * 16 / (1024**3)
    assert abs(memoria["total_gb"] - esperado) / esperado < 0.01
    esperado_ckpt = modelo.n_parameters * 4 / (1024**3)
    assert abs(memoria["checkpoint_gb"] - esperado_ckpt) / esperado_ckpt < 1e-4


def test_grad_accum_aumenta_tokens_por_passo() -> None:
    modelo, treino = _config()
    treino.grad_accum_steps = 4
    estimativa = estimar_treino(modelo, treino, tokens_por_segundo=1000.0)
    assert estimativa.tokens_por_passo == 16 * 4 * 256
