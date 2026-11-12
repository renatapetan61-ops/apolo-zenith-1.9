"""Exporta um checkpoint para o cliente mobile do Apolo Zenith 1.9.

Aplica quantização dinâmica INT8 (quando o backend suportar) e grava um
diretório autocontido com pesos, tokenizer e configuração mobile.

Uso::

    python -m scripts.quantize --checkpoint checkpoints/apolo-lm-tiny/checkpoints/best \
        --saida checkpoints/apolo-lm-tiny/mobile
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from core.language_model import ApoloLanguageModel  # noqa: E402
from mobile.inference import quantizar_int4, quantizar_int8  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta modelo compacto para o mobile")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--saida", required=True)
    parser.add_argument("--quantizacao", default="int8", choices=["off", "int8", "int4"])
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    modelo = ApoloLanguageModel.load(checkpoint, map_location="cpu")
    modelo.eval()

    quantizado = False
    if args.quantizacao == "int8":
        modelo, quantizado = quantizar_int8(modelo)
    elif args.quantizacao == "int4":
        modelo, quantizado = quantizar_int4(modelo)

    if quantizado:
        caminho_pesos = saida / "model_int8.pt" if args.quantizacao == "int8" else saida / "model_int4.pt"
        torch.save(modelo.state_dict(), caminho_pesos)
    else:
        caminho_pesos = saida / "model.pt"
        torch.save(modelo.state_dict(), caminho_pesos)
        print("[aviso] exportado em fp32: a quantização solicitada não está disponível neste backend.")

    (saida / "config.json").write_text(
        json.dumps(modelo.config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tokenizer_origem = checkpoint / "tokenizer"
    if tokenizer_origem.exists():
        shutil.copytree(tokenizer_origem, saida / "tokenizer", dirs_exist_ok=True)
    else:
        print("[aviso] o checkpoint não contém o tokenizer; copie-o manualmente.")

    tamanho = sum(p.stat().st_size for p in saida.rglob("*") if p.is_file())
    print(
        f"[mobile] exportado em {saida} | quantização={args.quantizacao} "
        f"(efetiva={quantizado}) | tamanho={tamanho/1024/1024:.2f} MB | threads={args.threads}"
    )


if __name__ == "__main__":
    main()
