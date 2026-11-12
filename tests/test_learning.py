"""Testes que comprovam que o modelo **realmente aprendeu**.

Estes testes operam sobre o checkpoint produzido pelo treinamento real
(``checkpoints/apolo-lm-tiny``).  Se o checkpoint ainda não existir, os testes
são ignorados com uma mensagem explícita — nunca são "aprovados" sem evidência.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest
import torch

from core.language_model import ApoloLanguageModel
from datasets.text_loader import PackedTokenDataset
from tokenizer.tokenizer import BPETokenizer

RAIZ = Path(__file__).resolve().parents[1]
CHECKPOINT = RAIZ / "checkpoints" / "apolo-lm-tiny" / "checkpoints" / "final"
DADOS = RAIZ / "datasets_storage" / "processed"


def _localizar_checkpoint() -> Optional[Path]:
    if CHECKPOINT.exists():
        return CHECKPOINT
    raiz = RAIZ / "checkpoints" / "apolo-lm-tiny" / "checkpoints"
    if raiz.exists():
        for nome in ("best", "final"):
            if (raiz / nome).exists():
                return raiz / nome
    return None


pytestmark = pytest.mark.skipif(
    _localizar_checkpoint() is None,
    reason="Nenhum checkpoint treinado encontrado (execute: python main.py train).",
)


@pytest.fixture(scope="module")
def artefatos():
    caminho = _localizar_checkpoint()
    modelo = ApoloLanguageModel.load(caminho, map_location="cpu")
    modelo.eval()
    tokenizer_dir = caminho / "tokenizer"
    if not tokenizer_dir.exists():
        tokenizer_dir = DADOS / "tokenizer"
    tokenizer = BPETokenizer.load(tokenizer_dir)
    return modelo, tokenizer


def _perda_em_tokens(modelo, ids: torch.Tensor) -> float:
    """Perda autoregressiva sobre uma sequência de tokens.

    ``logits[j]`` prevê ``ids[j+1]`` — por isso comparamos ``logits[:-1]``
    com ``ids[1:]``.
    """
    with torch.no_grad():
        logits, _ = modelo(ids.unsqueeze(0))
        perda = torch.nn.functional.cross_entropy(
            logits[0, :-1].float(), ids[1:], reduction="mean"
        )
    return float(perda.item())


def test_perda_validacao_menor_que_modelo_nao_treinado(artefatos) -> None:
    """O checkpoint treinado deve ter perda MUITO menor que a inicialização aleatória."""
    modelo, tokenizer = artefatos
    val_bin = DADOS / "val.bin"
    if not val_bin.exists():
        pytest.skip("split de validação ausente (execute prepare-data)")

    dataset = PackedTokenDataset(val_bin, modelo.config.block_size)
    assert len(dataset) > 0

    x, _ = dataset[0]
    ids = x.long()
    perda_treinado = _perda_em_tokens(modelo, ids)

    torch.manual_seed(0)
    aleatorio = ApoloLanguageModel.load(
        _localizar_checkpoint(), map_location="cpu"
    )
    # Re-inicializa os pesos para simular um modelo não treinado.
    for modulo in aleatorio.modules():
        if isinstance(modulo, torch.nn.Linear):
            torch.nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
            if modulo.bias is not None:
                torch.nn.init.zeros_(modulo.bias)
        elif isinstance(modulo, torch.nn.Embedding):
            torch.nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
    aleatorio.eval()
    perda_aleatorio = _perda_em_tokens(aleatorio, ids)

    assert perda_treinado < perda_aleatorio, (
        f"o modelo treinado ({perda_treinado:.4f}) não é melhor que o aleatório "
        f"({perda_aleatorio:.4f})"
    )
    assert perda_treinado < perda_aleatorio - 0.2, (
        "a diferença entre o modelo treinado e o aleatório é pequena demais "
        "para evidenciar aprendizado"
    )


def test_modelo_prefere_sequencia_real_versus_embaralhada(artefatos) -> None:
    """O modelo treinado deve dar menor perda ao texto real que ao embaralhado."""
    modelo, tokenizer = artefatos
    val_bin = DADOS / "val.bin"
    if not val_bin.exists():
        pytest.skip("split de validação ausente (execute prepare-data)")

    dataset = PackedTokenDataset(val_bin, modelo.config.block_size)
    x, _ = dataset[0]
    ids = x.long()

    gerador = torch.Generator().manual_seed(1234)
    permutacao = torch.randperm(len(ids), generator=gerador)
    embaralhado = ids[permutacao]

    perda_real = _perda_em_tokens(modelo, ids)
    perda_embaralhada = _perda_em_tokens(modelo, embaralhado)
    assert perda_real < perda_embaralhada, (
        f"perda real={perda_real:.4f} não é menor que embaralhada={perda_embaralhada:.4f}"
    )


def test_perda_por_token_fica_abaixo_do_aleatorio_em_multiplos_lotes(artefatos) -> None:
    """A evidência de aprendizado deve se manter em vários trechos do corpus."""
    modelo, tokenizer = artefatos
    val_bin = DADOS / "val.bin"
    if not val_bin.exists():
        pytest.skip("split de validação ausente (execute prepare-data)")

    dataset = PackedTokenDataset(val_bin, modelo.config.block_size)
    torch.manual_seed(0)
    aleatorio = ApoloLanguageModel.load(_localizar_checkpoint(), map_location="cpu")
    for modulo in aleatorio.modules():
        if isinstance(modulo, torch.nn.Linear):
            torch.nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
            if modulo.bias is not None:
                torch.nn.init.zeros_(modulo.bias)
        elif isinstance(modulo, torch.nn.Embedding):
            torch.nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
    aleatorio.eval()

    lotes = min(5, len(dataset))
    for i in range(lotes):
        x, _ = dataset[i]
        ids = x.long()
        assert _perda_em_tokens(modelo, ids) < _perda_em_tokens(aleatorio, ids)


def test_metricas_de_treino_foram_registradas() -> None:
    """O log JSONL deve conter a curva de perda de treino e de validação."""
    log = RAIZ / "checkpoints" / "apolo-lm-tiny" / "metricas.jsonl"
    if not log.exists():
        pytest.skip("log de métricas ausente")
    import json

    registros = [json.loads(linha) for linha in log.read_text(encoding="utf-8").splitlines() if linha.strip()]
    treino = [r for r in registros if r.get("fase") == "treino"]
    validacao = [r for r in registros if r.get("fase") == "validacao"]
    assert len(treino) >= 2
    assert validacao, "nenhuma validação foi registrada"
    assert treino[-1]["perda"] < treino[0]["perda"], "a perda de treino não diminuiu"
