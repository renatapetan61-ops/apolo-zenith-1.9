#!/usr/bin/env python3
"""Executa a sequência de treinos reais do Apolo Zenith 1.9 (todas as modalidades).

Uso::

    python3 scripts/treinar_tudo.py            # sequência completa
    python3 scripts/treinar_tudo.py --so visao # apenas uma modalidade

Cada etapa é independente e grava o seu próprio log em ``logs/``.  O script
respeita o lock de treino (apenas um processo por vez) e mantém o estado em
``logs/treinos_realizados.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from resources.config import carregar_config_recursos  # noqa: E402
from resources.training_lock import TreinoConcorrenteError, TrainingLock  # noqa: E402

ETAPAS: Dict[str, List[str]] = {
    "texto": [
        sys.executable, "-u", "-m", "training.text_trainer",
        "--steps", "300", "--batch-size", "4", "--block-size", "64",
        "--saida", "checkpoints/apolo-lm-continuacao", "--retomar",
        "checkpoints/apolo-lm-tiny/checkpoints/best",
    ],
    "codigo": [
        sys.executable, "-u", "-m", "training.code_trainer",
        "--steps", "3000", "--batch-size", "4", "--block-size", "64",
        "--dados", "datasets/code/processed", "--saida", "checkpoints/apolo-code",
        "--checkpoint-inicial", "checkpoints/apolo-lm-tiny/checkpoints/best",
        "--lr", "3e-4", "--sobrescrever",
    ],
    "visao": [
        sys.executable, "-u", "-m", "training.image_trainer",
        "--passos", "1000", "--batch-size", "16",
        "--parquets-extra", "datasets/images/raw/flickr8k/train_0.parquet",
        "--dim-encoder", "128", "--camadas", "3", "--saida", "models/vision",
    ],
    "audio": [
        sys.executable, "-u", "-m", "training.audio_trainer",
        "--passos", "250", "--batch-size", "4", "--saida", "models/audio",
    ],
    "video": [
        sys.executable, "-u", "-m", "training.video_trainer",
        "--passos", "150", "--batch-size", "2", "--saida", "models/video",
    ],
    "multimodal": [
        sys.executable, "-u", "-m", "training.multimodal_trainer",
        "--passos", "200", "--batch-size", "2", "--max-itens", "300",
        "--saida", "models/multimodal",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinos reais do Apolo Zenith 1.9")
    parser.add_argument("--so", nargs="*", default=[], help="modalidades a executar")
    parser.add_argument("--estado", default="logs/treinos_realizados.json")
    args = parser.parse_args()

    cfg = carregar_config_recursos(RAIZ / "configs" / "resource_manager.yaml")
    lock = TrainingLock(RAIZ / cfg.diretorio_lock)
    try:
        lock.adquirir("multimodal")
    except TreinoConcorrenteError as exc:
        print(f"[erro] {exc}")
        raise SystemExit(2)

    arquivo_estado = RAIZ / args.estado
    arquivo_estado.parent.mkdir(parents=True, exist_ok=True)
    resultados = {}
    if arquivo_estado.exists():
        try:
            resultados = json.loads(arquivo_estado.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            resultados = {}

    selecionadas = args.so or list(ETAPAS)
    try:
        for nome in selecionadas:
            if nome not in ETAPAS:
                print(f"[aviso] modalidade desconhecida: {nome}")
                continue
            print(f"\n{'=' * 70}\n=== TREINO: {nome.upper()} ===\n{'=' * 70}", flush=True)
            inicio = time.time()
            log = RAIZ / "logs" / f"treino_{nome}.log"
            with log.open("w", encoding="utf-8") as fh:
                codigo = subprocess.call(
                    ETAPAS[nome], cwd=str(RAIZ), stdout=fh, stderr=subprocess.STDOUT
                )
            duracao = time.time() - inicio
            resultados[nome] = {
                "codigo_saida": codigo,
                "duracao_s": round(duracao, 1),
                "log": str(log),
                "quando": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            arquivo_estado.write_text(
                json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"[{nome}] terminou em {duracao / 60:.1f} min (código {codigo})", flush=True)
    finally:
        lock.liberar()
    print("\n[resumo]")
    print(json.dumps(resultados, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
