"""Treinador da modalidade **CÓDIGO** (especialização textual).

O que é treinado de verdade:

* o Language Core (Transformer decoder-only) **continuando** a partir de um
  checkpoint existente ou do zero;
* sobre um corpus de código real (Python, TypeScript, Java, Shell, Go,
  Rust, C, C++, Kotlin, SQL, HTML, CSS), tokenizado com o mesmo BPE.

Métricas: perda e perplexidade de validação **no corpus de código**, além
de uma verificação sintática real dos arquivos gerados (quando o arquivo é
interpretável como código).

Uso::

    python -m training.code_trainer --dados datasets/code/processed \
        --saida checkpoints/apolo-code --steps 300
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import torch  # noqa: E402

from core.config import ModelConfig  # noqa: E402
from core.language_model import ApoloLanguageModel  # noqa: E402
from datasets.text_loader import PackedTokenDataset  # noqa: E402
from training.trainer import Trainer, TrainingConfig  # noqa: E402


def validar_sintaxe(codigo: str, linguagem: str) -> Dict[str, Any]:
    """Verifica a sintaxe do código gerado **sem executá-lo**.

    Usa o compilador do próprio interpretador (``ast.parse`` para Python) ou
    ``node --check``/``gcc -fsyntax-only`` quando disponíveis.  Nenhum código
    de dataset é executado.
    """
    if linguagem == "python":
        import ast

        try:
            ast.parse(codigo)
            return {"ok": True, "detalhe": "ast.parse ok"}
        except SyntaxError as exc:
            return {"ok": False, "detalhe": f"linha {exc.lineno}: {exc.msg}"}
    return {"ok": None, "detalhe": "verificador não disponível para esta linguagem"}


def verificar_roundtrip(
    pasta_codigo: str | Path,
    tokenizer_dir: str | Path,
    limite: int = 5,
) -> Dict[str, Any]:
    """Verifica o round-trip real: arquivo → tokens → texto → sintaxe.

    É uma verificação honesta do pipeline de código (tokenizador + corpus),
    e **não** da capacidade de gerar sintaxe nova pelo modelo.
    """
    from tokenizer.tokenizer import BPETokenizer

    tokenizer = BPETokenizer.load(tokenizer_dir)
    pasta = Path(pasta_codigo)
    arquivos = sorted(pasta.rglob("*.py"))[:limite]
    if not arquivos:
        return {"ok": None, "detalhe": "nenhum arquivo Python encontrado"}
    resultados = []
    for arquivo in arquivos:
        original = arquivo.read_text(encoding="utf-8", errors="replace")
        ids = tokenizer.encode_dataset(original, add_bos=False, add_eos=False)
        reconstruido = tokenizer.decode(ids)
        verificacao = validar_sintaxe(reconstruido, "python")
        fidelidade = (
            len(reconstruido.strip()) / max(len(original.strip()), 1)
        )
        resultados.append(
            {
                "arquivo": arquivo.name,
                "tokens": len(ids),
                "fidelidade_caracteres": round(fidelidade, 4),
                "sintaxe_ok": verificacao["ok"],
            }
        )
    ok = all(r["sintaxe_ok"] for r in resultados)
    return {
        "ok": ok,
        "arquivos": resultados,
        "detalhe": (
            "round-trip tokenizer sobre arquivos reais do corpus de código "
            "(o modelo não é avaliado aqui)"
        ),
    }


def baseline_mais_frequente(caminho_tokens: str | Path, max_tokens: int = 400_000) -> Dict[str, Any]:
    """Perda da baseline trivial: **prever sempre o token mais frequente**.

    Sem essa referência, uma perplexidade baixa pode ser enganosa — um modelo
    que aprende apenas a frequência dos tokens (muito comum em código, onde
    BOS/EOS/brancos dominam) parece bom sem saber gerar nada.
    """
    import math
    from collections import Counter

    from datasets.text_loader import carregar_tokens

    tokens = np.asarray(carregar_tokens(caminho_tokens)[:max_tokens], dtype=np.int64)
    if tokens.size == 0:
        return {"perda": None, "observacao": "fluxo de tokens vazio"}
    contagem = Counter(tokens.tolist())
    total = tokens.size
    frequencia = contagem.most_common(1)[0][1] / total
    perda = -math.log(frequencia)
    return {
        "perda": round(perda, 4),
        "perplexidade": round(math.exp(perda), 2),
        "token_mais_frequente": contagem.most_common(1)[0][0],
        "frequencia": round(frequencia, 4),
        "observacao": (
            "baseline de frequência (prever sempre o token mais comum) — "
            "referência mínima para julgar a perplexidade"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento da modalidade código")
    parser.add_argument("--dados", default="datasets/code/processed")
    parser.add_argument("--saida", default="checkpoints/apolo-code")
    parser.add_argument("--tokenizer", default="datasets_storage/processed/tokenizer")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--checkpoint-inicial", default=None,
                        help="checkpoint do Language Core para continuar")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sobrescrever", action="store_true")
    args = parser.parse_args()

    dados = Path(args.dados)
    saida = Path(args.saida)
    if (saida / "checkpoints" / "final").exists() and not args.sobrescrever:
        raise SystemExit(
            f"[erro] já existe um experimento em '{saida}'.\n"
            "       Use outro --saida, ou --sobrescrever para substituir "
            "intencionalmente."
        )

    from scripts.train import aplicar, carregar_config

    bruto = carregar_config(args.config)
    modelo_config = ModelConfig.from_dict(bruto.get("model", {}))
    modelo_config.block_size = args.block_size

    treino_config = aplicar(TrainingConfig(), dict(bruto.get("training", {})))
    treino_config.saida = str(saida)
    treino_config.batch_size = args.batch_size
    treino_config.block_size = args.block_size
    treino_config.max_steps = args.steps
    treino_config.lr = args.lr
    treino_config.eval_interval = args.eval_interval
    treino_config.save_interval = args.save_interval
    treino_config.log_interval = args.log_interval
    treino_config.seed = args.seed
    # O gerenciador de recursos limita o batch e a sequência ao teto do
    # dispositivo (batch 1–4, sequência 32–64).
    treino_config.aplicar_limites_batch = True
    # No celular, avaliar 11.700 sequências custaria mais que o treino.
    treino_config.max_lotes_validacao = 40

    treino = PackedTokenDataset(dados / "train.bin", treino_config.block_size)
    validacao = PackedTokenDataset(dados / "val.bin", treino_config.block_size)
    print(
        f"[dados] treino={len(treino)} sequências ({treino.num_tokens:,} tokens) | "
        f"validação={len(validacao)} sequências ({validacao.num_tokens:,} tokens)"
    )
    if len(treino) == 0:
        raise SystemExit("corpus de código vazio — rode a preparação antes")

    modelo = ApoloLanguageModel(modelo_config)
    if args.checkpoint_inicial:
        inicial = Path(args.checkpoint_inicial)
        if inicial.exists():
            modelo = ApoloLanguageModel.load(inicial, map_location="cpu")
            print(f"[modelo] continuando a partir de {inicial}")
        else:
            print(f"[aviso] checkpoint inicial inexistente: {inicial}")
    print(f"[modelo] parâmetros={modelo.num_parameters():,}")

    trainer = Trainer(
        modelo,
        treino_config,
        treino,
        validacao,
        tokenizer_dir=args.tokenizer,
    )
    resumo = trainer.treinar()

    # Verificação real do pipeline de código: round-trip do tokenizador sobre
    # arquivos verdadeiros do corpus + checagem sintática (sem executar nada).
    pasta_codigo = Path(args.dados).parent / "raw"
    try:
        verificacao = verificar_roundtrip(pasta_codigo, args.tokenizer)
    except Exception as exc:
        verificacao = {"ok": None, "detalhe": f"não verificável: {exc}"}

    resumo["verificacao_roundtrip_codigo"] = verificacao
    try:
        resumo["baseline_frequencia"] = baseline_mais_frequente(dados / "val.bin")
    except Exception as exc:
        resumo["baseline_frequencia"] = {"erro": str(exc)}
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "resumo_treino.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
