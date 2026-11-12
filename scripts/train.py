"""Treina o Apolo Language Core.

Uso::

    python -m scripts.train --config configs/training.yaml \
        --dados datasets_storage/processed --saida checkpoints/apolo-lm-tiny

Retomada::

    python -m scripts.train --config configs/training.yaml --retomar checkpoints/apolo-lm-tiny/checkpoints/final
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from core.config import ModelConfig  # noqa: E402
from core.language_model import ApoloLanguageModel  # noqa: E402
from datasets.text_loader import PackedTokenDataset  # noqa: E402
from tokenizer.tokenizer import BPETokenizer  # noqa: E402
from training.trainer import Trainer, TrainingConfig  # noqa: E402


def carregar_config(caminho: str | Path) -> Dict[str, Any]:
    """Lê o YAML de configuração do experimento."""
    with Path(caminho).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def aplicar(dc: Any, dados: Dict[str, Any]) -> Any:
    """Aplica um dicionário de overrides em uma dataclass, validando campos."""
    validos = {f.name for f in fields(dc)}
    desconhecidos = set(dados) - validos
    if desconhecidos:
        raise ValueError(f"Chaves desconhecidas para {type(dc).__name__}: {sorted(desconhecidos)}")
    for chave, valor in dados.items():
        setattr(dc, chave, valor)
    return dc


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento do Apolo Language Core")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--dados", default="datasets_storage/processed")
    parser.add_argument("--saida", default=None)
    parser.add_argument("--retomar", default=None, help="caminho de um checkpoint para retomar")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--sobrescrever",
        action="store_true",
        help="permite sobrescrever um experimento existente no diretório de saída",
    )
    args = parser.parse_args()

    bruto = carregar_config(args.config)
    cfg_modelo = bruto.get("model", {})
    cfg_treino = bruto.get("training", {})

    modelo_config = ModelConfig.from_dict(cfg_modelo)
    treino_config = aplicar(TrainingConfig(), dict(cfg_treino))

    # Overrides pela linha de comando
    if args.max_steps is not None:
        treino_config.max_steps = args.max_steps
    if args.batch_size is not None:
        treino_config.batch_size = args.batch_size
    if args.lr is not None:
        treino_config.lr = args.lr
    if args.block_size is not None:
        treino_config.block_size = args.block_size
        modelo_config.block_size = args.block_size
    if args.device is not None:
        treino_config.device = args.device
    if args.seed is not None:
        treino_config.seed = args.seed
    if args.saida is not None:
        treino_config.saida = args.saida

    # Trava de segurança: um novo treino não pode sobrescrever silenciosamente
    # os checkpoints de um experimento anterior no mesmo diretório.
    saida_experimento = Path(treino_config.saida)
    experimento_existente = saida_experimento / "checkpoints" / "final"
    if experimento_existente.exists() and not args.sobrescrever:
        print(
            f"[erro] já existe um experimento em '{saida_experimento}'.\n"
            "       Use outro --saida, ou --sobrescrever para substituir "
            "intencionalmente (o checkpoint anterior será perdido)."
        )
        raise SystemExit(2)

    dados = Path(args.dados)
    tokenizer = BPETokenizer.load(dados / "tokenizer")
    if tokenizer.vocab_size != modelo_config.vocab_size:
        print(
            f"[aviso] vocab do tokenizer ({tokenizer.vocab_size}) difere do modelo "
            f"({modelo_config.vocab_size}); ajustando o modelo ao tokenizer."
        )
        modelo_config.vocab_size = tokenizer.vocab_size

    treino = PackedTokenDataset(dados / "train.bin", treino_config.block_size)
    validacao = PackedTokenDataset(dados / "val.bin", treino_config.block_size)
    print(
        f"[dados] treino={len(treino)} sequências ({treino.num_tokens} tokens) | "
        f"validação={len(validacao)} sequências ({validacao.num_tokens} tokens)"
    )

    modelo = ApoloLanguageModel(modelo_config)
    print(
        f"[modelo] {modelo_config.n_layer} camadas, {modelo_config.n_embd} dims, "
        f"{modelo_config.n_head} cabeças, {modelo_config.block_size} contexto, "
        f"vocab={modelo_config.vocab_size} | parâmetros={modelo.num_parameters():,}"
    )

    trainer = Trainer(
        modelo,
        treino_config,
        treino,
        validacao,
        tokenizer_dir=dados / "tokenizer",
    )

    if args.retomar:
        trainer.retomar(args.retomar)

    resumo = trainer.treinar()
    Path(treino_config.saida).mkdir(parents=True, exist_ok=True)
    (Path(treino_config.saida) / "resumo_treino.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
