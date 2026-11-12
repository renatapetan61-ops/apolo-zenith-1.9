"""Inferência de texto com o Apolo Language Core.

Uso não interativo::

    python -m scripts.infer --checkpoint checkpoints/apolo-lm-tiny/checkpoints/best \
        --prompt "Era uma vez" --max-tokens 100 --temperatura 0.8

Modo interativo (sessão de conversa simples)::

    python -m scripts.infer --checkpoint checkpoints/apolo-lm-tiny/checkpoints/best --interativo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from core.language_model import ApoloLanguageModel  # noqa: E402
from tokenizer.tokenizer import BPETokenizer  # noqa: E402


def carregar(checkpoint: str | Path):
    checkpoint = Path(checkpoint)
    modelo = ApoloLanguageModel.load(checkpoint, map_location="cpu")
    tokenizer_dir = checkpoint / "tokenizer"
    if not tokenizer_dir.exists():
        tokenizer_dir = checkpoint.parent.parent / "processed" / "tokenizer"
    tokenizer = BPETokenizer.load(tokenizer_dir)
    return modelo, tokenizer


def gerar(modelo, tokenizer, prompt: str, max_tokens: int, temperatura: float, seed: int, top_k: int, top_p: float):
    ids = tokenizer.encode(prompt)
    if not ids:
        return "", 0, 0.0
    entrada = torch.tensor([ids[-modelo.config.block_size :]], dtype=torch.long)
    gerador = torch.Generator().manual_seed(seed)
    inicio = time.time()
    with torch.no_grad():
        saida = modelo.generate(
            entrada,
            max_new_tokens=max_tokens,
            temperature=temperatura,
            top_k=top_k if temperatura > 0 else None,
            top_p=top_p if temperatura > 0 else None,
            repetition_penalty=1.05,
            stop_token_ids=[tokenizer.eos_id] if tokenizer.eos_id is not None else None,
            generator=gerador,
        )
    decorrido = max(time.time() - inicio, 1e-6)
    return tokenizer.decode(saida[0].tolist()), int(saida.shape[1]), saida.shape[1] / decorrido


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferência do Apolo Language Core")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--interativo", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperatura", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    modelo, tokenizer = carregar(args.checkpoint)
    print(f"[apolo] checkpoint carregado: {args.checkpoint}")
    print(f"[apolo] parâmetros: {modelo.num_parameters():,} | vocab: {tokenizer.vocab_size}")

    if args.interativo:
        print("Digite 'sair' para encerrar.\n")
        historico = ""
        while True:
            try:
                entrada = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if entrada.lower() in {"sair", "exit", "quit"}:
                break
            if not entrada:
                continue
            prompt = (historico + entrada) if historico else entrada
            texto, n, tps = gerar(
                modelo, tokenizer, prompt, args.max_tokens, args.temperatura, args.seed, args.top_k, args.top_p
            )
            print(texto)
            print(f"\n[{n} tokens, {tps:.1f} tok/s]\n")
            historico = prompt + texto
        return

    prompt = args.prompt
    if not prompt:
        parser.error("informe --prompt ou --interativo")
    texto, n, tps = gerar(
        modelo, tokenizer, prompt, args.max_tokens, args.temperatura, args.seed, args.top_k, args.top_p
    )
    print(f"--- prompt ---\n{prompt}\n--- continuação ---\n{texto}\n")
    print(f"[{n} tokens gerados, {tps:.1f} tok/s]")


if __name__ == "__main__":
    main()
