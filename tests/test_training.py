"""Testes do loop de treinamento do Apolo."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from core.config import ModelConfig
from core.language_model import ApoloLanguageModel
from datasets.text_loader import PackedTokenDataset
from training.checkpoint import CheckpointManager, load_checkpoint
from training.trainer import Trainer, TrainingConfig


class _DatasetSintetico(Dataset):
    """Sequência repetitiva: o modelo deve conseguir sobreajustá-la."""

    def __init__(self, n: int = 32, block_size: int = 32, semente: int = 0) -> None:
        rng = np.random.default_rng(semente)
        self.sequencia = rng.integers(0, 64, size=block_size).tolist()
        self.x = torch.tensor([self.sequencia] * n, dtype=torch.long)
        self.y = torch.tensor([self.sequencia[1:] + [self.sequencia[0]]] * n, dtype=torch.long)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _modelo_pequeno() -> ApoloLanguageModel:
    torch.manual_seed(123)
    return ApoloLanguageModel(
        ModelConfig(
            vocab_size=64,
            n_layer=2,
            n_head=2,
            n_embd=32,
            block_size=32,
            mlp_hidden=64,
        )
    )


def test_treino_reduz_perda(tmp_path: Path) -> None:
    """O Trainer deve reduzir a perda de forma mensurável."""
    modelo = _modelo_pequeno()
    dataset = _DatasetSintetico(n=16)
    cfg = TrainingConfig(
        saida=str(tmp_path / "treino"),
        batch_size=4,
        block_size=32,
        max_steps=40,
        lr=3e-3,
        min_lr=3e-4,
        warmup_steps=5,
        eval_interval=20,
        log_interval=10,
        save_interval=20,
        device="cpu",
        seed=7,
    )
    trainer = Trainer(modelo, cfg, dataset, dataset)
    antes = trainer.avaliar()["val_loss"]
    resumo = trainer.treinar()
    depois = trainer.avaliar()["val_loss"]
    assert resumo["passos"] == 40
    assert depois < antes * 0.8, f"perda não caiu o suficiente: {antes} -> {depois}"


def test_grad_accum_equivalencia(tmp_path: Path) -> None:
    """Gradient accumulation deve Approximately reproduzir um batch maior."""
    torch.manual_seed(5)
    dataset = _DatasetSintetico(n=8, semente=2)

    def _rodar(accum: int) -> float:
        torch.manual_seed(99)
        modelo = _modelo_pequeno()
        cfg = TrainingConfig(
            saida=str(tmp_path / f"acum{accum}"),
            batch_size=2,
            block_size=32,
            max_steps=5,
            lr=1e-3,
            min_lr=1e-4,
            warmup_steps=2,
            grad_accum_steps=accum,
            eval_interval=1000,
            log_interval=1000,
            save_interval=0,
            device="cpu",
            seed=3,
        )
        trainer = Trainer(modelo, cfg, dataset, dataset)
        trainer.treinar()
        return trainer.avaliar()["val_loss"]

    perda_1 = _rodar(1)
    perda_4 = _rodar(4)
    assert perda_4 < perda_1 * 1.5  # mesma ordem de grandeza, sem explodir


def test_checkpoint_salva_e_carrega(tmp_path: Path) -> None:
    modelo = _modelo_pequeno()
    dataset = _DatasetSintetico(n=8)
    cfg = TrainingConfig(
        saida=str(tmp_path / "ckpt"),
        batch_size=4,
        max_steps=3,
        block_size=32,
        eval_interval=1000,
        log_interval=1000,
        save_interval=0,
        device="cpu",
    )
    trainer = Trainer(modelo, cfg, dataset, dataset)
    trainer.treinar()
    caminho = tmp_path / "ckpt" / "checkpoints" / "final"
    assert caminho.exists()
    assert (caminho / "model.pt").exists()
    assert (caminho / "meta.json").exists()

    recarregado = _modelo_pequeno()
    meta = load_checkpoint(caminho, model=recarregado)
    assert meta["step"] == 3
    x, y = dataset[0]
    with torch.no_grad():
        a, _ = modelo(x.unsqueeze(0), y.unsqueeze(0))
        b, _ = recarregado(x.unsqueeze(0), y.unsqueeze(0))
    assert torch.allclose(a, b, atol=1e-6)


def test_checkpoint_manager_rotaciona(tmp_path: Path) -> None:
    modelo = _modelo_pequeno()
    gerenciador = CheckpointManager(tmp_path / "ckpts", manter=2)
    for i in range(4):
        gerenciador.salvar(f"step_{i:06d}", modelo, step=i, metrica=1.0 / (i + 1))
    restantes = sorted(p.name for p in (tmp_path / "ckpts").glob("step_*"))
    assert len(restantes) == 2
    assert (tmp_path / "ckpts" / "best").exists()


def test_avaliacao_do_script_bate_com_a_do_trainer(tmp_path: Path) -> None:
    """Regressão: o script de avaliação não pode usar um alinhamento diferente
    do loop de treino (off-by-one no deslocamento dos alvos)."""
    from datasets.text_loader import salvar_tokens
    from scripts.evaluate import avaliar_perda_split

    modelo = _modelo_pequeno()
    rng = np.random.default_rng(5)
    tokens = rng.integers(0, 64, size=2048).tolist()
    caminho = salvar_tokens(tokens, tmp_path / "dados.bin")

    dataset = PackedTokenDataset(caminho, 32)
    cfg = TrainingConfig(
        saida=str(tmp_path / "eval"),
        batch_size=1,  # mesmo batch do script: comparação determinística
        block_size=32,
        max_steps=1,
        eval_interval=1000,
        log_interval=1000,
        save_interval=0,
        device="cpu",
    )
    trainer = Trainer(modelo, cfg, dataset, dataset)
    perda_trainer = trainer.avaliar()["val_loss"]

    perda_script = avaliar_perda_split(modelo, None, caminho, 32, max_lotes=100)["perda"]
    assert abs(perda_script - perda_trainer) < 1e-5


def test_retomada_continua_a_partir_do_passo(tmp_path: Path) -> None:
    modelo = _modelo_pequeno()
    dataset = _DatasetSintetico(n=8)
    cfg = TrainingConfig(
        saida=str(tmp_path / "resume"),
        batch_size=4,
        max_steps=4,
        block_size=32,
        eval_interval=1000,
        log_interval=1000,
        save_interval=0,
        device="cpu",
    )
    trainer = Trainer(modelo, cfg, dataset, dataset)
    trainer.treinar()
    checkpoint = tmp_path / "resume" / "checkpoints" / "final"

    cfg2 = TrainingConfig(
        saida=str(tmp_path / "resume"),
        batch_size=4,
        max_steps=8,
        block_size=32,
        eval_interval=1000,
        log_interval=1000,
        save_interval=0,
        device="cpu",
    )
    trainer2 = Trainer(_modelo_pequeno(), cfg2, dataset, dataset)
    trainer2.retomar(checkpoint)
    assert trainer2.passo == 4
    resumo = trainer2.treinar()
    assert resumo["passos"] == 8
