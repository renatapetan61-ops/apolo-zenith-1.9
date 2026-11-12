"""Treinador da modalidade **TEXTO**.

Continua o treino do Language Core sobre o corpus textual (Gutenberg +
Wikisource) usando o mesmo loop com guardas de recursos.

O que é treinado: o Transformer decoder-only do Language Core, com AdamW,
warmup+cosine, gradient clipping, acúmulo de gradiente e checkpoints com
estado completo (RNG, dataset, batch, fila).

Métricas: perda e perplexidade de validação (texto).

Uso::

    python -m training.text_trainer --dados datasets_storage/processed \
        --saida checkpoints/apolo-lm-tiny --steps 700 --retomar <checkpoint>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from core.config import ModelConfig  # noqa: E402
from core.language_model import ApoloLanguageModel  # noqa: E402
from datasets.text_loader import PackedTokenDataset  # noqa: E402
from training.trainer import Trainer, TrainingConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento da modalidade texto")
    parser.add_argument("--dados", default="datasets_storage/processed")
    parser.add_argument("--saida", default="checkpoints/apolo-lm-tiny")
    parser.add_argument("--tokenizer", default="datasets_storage/processed/tokenizer")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--retomar", default=None)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sobrescrever", action="store_true")
    args = parser.parse_args()

    from scripts.train import aplicar, carregar_config

    bruto = carregar_config(args.config)
    modelo_config = ModelConfig.from_dict(bruto.get("model", {}))
    modelo_config.block_size = args.block_size

    treino_config = aplicar(TrainingConfig(), dict(bruto.get("training", {})))
    treino_config.saida = args.saida
    treino_config.batch_size = args.batch_size
    treino_config.block_size = args.block_size
    treino_config.max_steps = args.steps
    treino_config.lr = args.lr
    treino_config.eval_interval = args.eval_interval
    treino_config.save_interval = args.save_interval
    treino_config.seed = args.seed
    treino_config.aplicar_limites_batch = True
    # No celular, avaliar 11.700 sequências custaria mais que o treino.
    treino_config.max_lotes_validacao = 40

    # Trava de segurança: um novo treino não pode sobrescrever
    # silenciosamente um experimento existente (o checkpoint preservado do
    # Language Core é insubstituível).
    experimento_existente = Path(treino_config.saida) / "checkpoints" / "final"
    if experimento_existente.exists() and not args.sobrescrever:
        raise SystemExit(
            f"[erro] já existe um experimento em '{treino_config.saida}'.\n"
            "       Use outro --saida, ou --sobrescrever para substituir "
            "intencionalmente."
        )

    dados = Path(args.dados)
    treino = PackedTokenDataset(dados / "train.bin", treino_config.block_size)
    validacao = PackedTokenDataset(dados / "val.bin", treino_config.block_size)
    print(
        f"[dados] treino={len(treino)} sequências ({treino.num_tokens:,} tokens) | "
        f"validação={len(validacao)} sequências ({validacao.num_tokens:,} tokens)"
    )
    if len(treino) == 0:
        raise SystemExit("corpus textual vazio — rode a preparação antes")

    modelo = ApoloLanguageModel(modelo_config)
    print(f"[modelo] parâmetros={modelo.num_parameters():,}")

    # Ao retomar, ``max_steps`` é **relativo** ao passo já atingido
    # (senão o loop nem começaria: passo 1800 > max_steps 300).
    if args.retomar:
        caminho = Path(args.retomar)
        meta_path = caminho / "meta.json" if caminho.is_dir() else caminho
        if meta_path.exists():
            try:
                passo_anterior = int(json.loads(meta_path.read_text(encoding="utf-8")).get("step", 0))
                treino_config.max_steps = passo_anterior + args.steps
                print(f"[retomada] continuando do passo {passo_anterior} até {treino_config.max_steps}")
            except (OSError, json.JSONDecodeError, ValueError):
                pass

    trainer = Trainer(
        modelo,
        treino_config,
        treino,
        validacao,
        tokenizer_dir=args.tokenizer,
    )
    if args.retomar:
        trainer.retomar(args.retomar)

    resumo = trainer.treinar()
    resumo["modalidade"] = "texto"
    Path(args.saida).mkdir(parents=True, exist_ok=True)
    (Path(args.saida) / "resumo_treino.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
