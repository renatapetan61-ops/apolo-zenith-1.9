"""Avaliação real do Apolo Zenith 1.9.

Executa:
1. perplexidade no split de teste (dados próprios);
2. benchmarks locais opcionais (JSONL) nas tarefas ``perda``,
   ``mult_escolha``, ``matematica`` e ``codigo``.

O relatório é gravado em JSON e identifica exatamente quais benchmarks foram
executados — nenhum resultado é estimado ou preenchido manualmente.

Uso::

    python -m scripts.evaluate --checkpoint checkpoints/apolo-lm-tiny/checkpoints/best \
        --dados datasets_storage/processed --saida logs/avaliacao.json \
        --benchmark "math.jsonl:matematica"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from core.language_model import ApoloLanguageModel  # noqa: E402
from datasets.text_loader import PackedTokenDataset  # noqa: E402
from evaluation.benchmarks import avaliar_benchmark  # noqa: E402
from evaluation.metrics import perplexity  # noqa: E402
from tokenizer.tokenizer import BPETokenizer  # noqa: E402


def avaliar_perda_split(modelo, tokenizer, caminho: Path, block_size: int, max_lotes: int = 40) -> Dict:
    """Perplexidade sobre o split de teste empacotado."""
    if not caminho.exists():
        return {"erro": f"split não encontrado: {caminho}"}
    dataset = PackedTokenDataset(caminho, block_size)
    if len(dataset) == 0:
        return {"erro": "split vazio"}
    modelo.eval()
    perda_total = 0.0
    tokens = 0
    with torch.no_grad():
        for i in range(min(len(dataset), max_lotes)):
            x, y = dataset[i]
            entrada = x.unsqueeze(0)
            alvos = y.unsqueeze(0)
            # y já é x deslocado em 1 token: logits[j] prevê y[j].
            # Usamos a própria perda do modelo (todas as posições), que é
            # exatamente a métrica usada durante o treino.
            _, perda_modelo = modelo(entrada, alvos)
            if perda_modelo is None:
                continue
            n = int(alvos.numel())
            perda_total += float(perda_modelo.item()) * n
            tokens += n
    perda = perda_total / max(tokens, 1)
    return {
        "perda": round(perda, 6),
        "perplexidade": round(perplexity(perda), 4),
        "tokens": tokens,
        "lotes": min(len(dataset), max_lotes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação do Apolo Zenith 1.9")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dados", default="datasets_storage/processed")
    parser.add_argument("--saida", default="logs/avaliacao.json")
    parser.add_argument("--benchmark", action="append", default=[], help="arquivo.jsonl:tarefa")
    parser.add_argument("--max-exemplos", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    modelo = ApoloLanguageModel.load(checkpoint, map_location="cpu")
    tokenizer_dir = checkpoint / "tokenizer"
    if not tokenizer_dir.exists():
        tokenizer_dir = Path(args.dados) / "tokenizer"
    tokenizer = BPETokenizer.load(tokenizer_dir)

    inicio = time.time()
    relatorio: Dict = {
        "checkpoint": str(checkpoint),
        "parametros": modelo.num_parameters(),
        "vocab_size": tokenizer.vocab_size,
        "benchmarks": {},
    }

    print("[1/2] avaliando perplexidade no split de teste…")
    relatorio["teste_proprio"] = avaliar_perda_split(
        modelo, tokenizer, Path(args.dados) / "test.bin", modelo.config.block_size
    )
    print(json.dumps(relatorio["teste_proprio"], indent=2, ensure_ascii=False))

    print("[2/2] executando benchmarks externos…")
    for especificacao in args.benchmark:
        if ":" not in especificacao:
            print(f"  [aviso] formato inválido: {especificacao} (use arquivo.jsonl:tarefa)")
            continue
        arquivo, tarefa = especificacao.rsplit(":", 1)
        caminho = Path(arquivo)
        if not caminho.exists():
            relatorio["benchmarks"][especificacao] = {"erro": f"arquivo não encontrado: {arquivo}"}
            continue
        print(f"  executando {tarefa} em {arquivo}…")
        resultado = avaliar_benchmark(
            modelo,
            tokenizer,
            caminho,
            tarefa=tarefa,
            max_exemplos=args.max_exemplos,
            max_new_tokens=args.max_tokens,
        )
        relatorio["benchmarks"][especificacao] = resultado
        print(json.dumps(resultado, indent=2, ensure_ascii=False))

    relatorio["tempo_s"] = round(time.time() - inicio, 2)
    relatorio["nota"] = (
        "Resultados gerados por inferência real do modelo neste checkpoint. "
        "Comparações com modelos de referência exigem condições equivalentes "
        "(mesmos prompts, decoding e conjunto de testes)."
    )
    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[relatório] {saida}")


if __name__ == "__main__":
    main()
