"""Testes do otimizador AdamW e do agendador de taxa de aprendizado."""

from __future__ import annotations

import math

import pytest
import torch

from training.adamw import AdamW, build_optimizer
from training.scheduler import WarmupCosineScheduler


def _modelo_simples() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )


def _dados():
    torch.manual_seed(1)
    x = torch.randn(32, 8)
    y = torch.randn(32, 4)
    return x, y


def test_adamw_equivalente_ao_torch() -> None:
    """Nosso AdamW deve numerically bater com torch.optim.AdamW."""
    x, y = _dados()
    modelo_a = _modelo_simples()
    modelo_b = _modelo_simples()
    assert all(
        torch.equal(pa, pb) for pa, pb in zip(modelo_a.parameters(), modelo_b.parameters())
    )

    otim_a = torch.optim.AdamW(modelo_a.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    otim_b = AdamW(modelo_b.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    for _ in range(10):
        for modelo, otim in ((modelo_a, otim_a), (modelo_b, otim_b)):
            otim.zero_grad()
            perda = torch.nn.functional.mse_loss(modelo(x), y)
            perda.backward()
            otim.step()

    for pa, pb in zip(modelo_a.parameters(), modelo_b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-5)


def test_weight_decay_desacoplado() -> None:
    """Com gradiente zero, o AdamW apenas aplica o decaimento de pesos."""
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    p.grad = torch.zeros(2)
    otim = AdamW([p], lr=0.1, weight_decay=0.5)
    otim.step()
    esperado = torch.tensor([1.0 * (1 - 0.1 * 0.5), 2.0 * (1 - 0.1 * 0.5)])
    assert torch.allclose(p.data, esperado, atol=1e-6)


def test_um_parametro_reduz_perda() -> None:
    """Otimizador deve reduzir a perda em um problema quadrático simples."""
    p = torch.nn.Parameter(torch.tensor([5.0]))
    otim = AdamW([p], lr=0.2, weight_decay=0.0)
    perda_inicial = float((p ** 2).item())
    for _ in range(200):
        otim.zero_grad()
        perda = (p ** 2).sum()
        perda.backward()
        otim.step()
    assert float((p ** 2).item()) < perda_inicial * 0.01


def test_build_optimizer_separa_pesos() -> None:
    modelo = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    otim = build_optimizer(modelo, lr=1e-3, weight_decay=0.2)
    assert isinstance(otim, AdamW)
    decay = otim.param_groups[0]
    sem_decay = otim.param_groups[1]
    assert decay["weight_decay"] == 0.2
    assert sem_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])


def test_scheduler_warmup_e_cosseno() -> None:
    modelo = _modelo_simples()
    otim = AdamW(modelo.parameters(), lr=1e-3)
    agend = WarmupCosineScheduler(otim, max_steps=100, warmup_steps=10, min_lr=1e-5)
    assert agend.lr_at(0) == pytest.approx(1e-5)
    assert agend.lr_at(10) == pytest.approx(1e-3, rel=1e-6)
    assert agend.lr_at(55) == pytest.approx(1e-3 * 0.5 + 1e-5 * 0.5, rel=1e-4)
    assert agend.lr_at(100) == pytest.approx(1e-5, abs=1e-7)
    assert agend.lr_at(10) > agend.lr_at(50) > agend.lr_at(100)


def test_scheduler_aplica_lr_ao_otimizador() -> None:
    modelo = _modelo_simples()
    otim = AdamW(modelo.parameters(), lr=1e-3)
    agend = WarmupCosineScheduler(otim, max_steps=50, warmup_steps=5, min_lr=0.0)
    lr = agend.step_update(5)
    assert otim.param_groups[0]["lr"] == pytest.approx(lr)


def test_scheduler_state_dict() -> None:
    modelo = _modelo_simples()
    otim = AdamW(modelo.parameters(), lr=1e-3)
    agend = WarmupCosineScheduler(otim, max_steps=50, warmup_steps=5)
    agend.step_update(20)
    estado = agend.state_dict()
    assert estado["last_epoch"] == 20
    novo = WarmupCosineScheduler(otim, max_steps=50, warmup_steps=5)
    novo.load_state_dict(estado)
    assert novo.last_epoch == 20
