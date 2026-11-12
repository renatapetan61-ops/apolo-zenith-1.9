"""Prepara os datasets processados de cada modalidade do Apolo Zenith 1.9.

Tarefas (todas com contagem real de amostras):

* **código**  – tokeniza e empacota o corpus (``train/val/test.bin``);
* **imagens** – exporta o índice de legendas (JSONL);
* **áudio**   – exporta o índice de transcrições (JSONL);
* **vídeo**   – extrai os frames (com amostragem por intervalo) e o índice.

Os arquivos originais em ``raw/`` **nunca** são alterados.

Uso::

    python -m scripts.prepare_multimodal --modalidade code
    python -m scripts.prepare_multimodal --todas
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from datasets.audio_loader import exportar_indice_audio  # noqa: E402
from datasets.code_loader import contar_linguagens, preparar_corpus_codigo  # noqa: E402
from datasets.image_loader import exportar_legendas_jsonl  # noqa: E402
from datasets.manifest import DatasetManifest  # noqa: E402
from datasets.video_loader import exportar_indice_video, extrair_frames_para_pasta  # noqa: E402


def preparar_code(args) -> Dict[str, Any]:
    """Tokeniza e empacota o corpus de código."""
    print("[code] tokenizando o corpus de código ...")
    inicio = time.time()
    estatisticas = preparar_corpus_codigo(
        Path(args.code_raw),
        Path(args.code_saida),
        tokenizer_dir=args.tokenizer,
        max_arquivos=args.max_arquivos,
    )
    linguagens = contar_linguagens(args.code_raw)
    estatisticas["linguagens"] = linguagens
    estatisticas["tempo_s"] = round(time.time() - inicio, 1)
    print(f"[code] {json.dumps(estatisticas['train'], ensure_ascii=False)}")
    print(f"[code] linguagens: {linguagens}")
    return estatisticas


def preparar_images(args) -> Dict[str, Any]:
    """Exporta o índice de legendas das imagens."""
    destino = Path(args.images_saida) / "legendas.jsonl"
    total = exportar_legendas_jsonl(args.images_parquet, destino)
    print(f"[images] {total} imagens com legendas -> {destino}")
    return {"imagens": total, "caminho": str(destino)}


def preparar_audio(args) -> Dict[str, Any]:
    """Exporta o índice de transcrições do áudio."""
    destino = Path(args.audio_saida) / "transcricoes.jsonl"
    total = exportar_indice_audio(args.audio_pasta, destino)
    print(f"[audio] {total} falas indexadas -> {destino}")
    return {"falas": total, "caminho": str(destino)}


def preparar_video(args) -> Dict[str, Any]:
    """Extrai os frames dos vídeos (amostragem por intervalo)."""
    print("[video] extraindo frames ...")
    inicio = time.time()
    resumo = extrair_frames_para_pasta(
        args.video_raw,
        args.video_frames,
        max_frames=args.max_frames,
        amostra_fps=args.fps,
        tamanho=args.frame_size,
    )
    indice = Path(args.video_saida) / "indice.jsonl"
    exportar_indice_video(args.video_raw, indice)
    resumo["frames_por_video"] = args.max_frames
    resumo["tempo_s"] = round(time.time() - inicio, 1)
    print(
        f"[video] {resumo['videos']} vídeos -> {resumo['frames_total']} frames "
        f"({resumo['tempo_s']}s)"
    )
    return resumo


def main() -> None:
    parser = argparse.ArgumentParser(description="Preparação dos datasets multimodais")
    parser.add_argument("--modalidade", default=None, choices=["code", "images", "audio", "video"])
    parser.add_argument("--todas", action="store_true")
    parser.add_argument("--code-raw", default="datasets/code/raw")
    parser.add_argument("--code-saida", default="datasets/code/processed")
    parser.add_argument("--images-parquet", default="datasets/images/raw/flickr8k/test.parquet")
    parser.add_argument("--images-saida", default="datasets/images/processed")
    parser.add_argument("--audio-pasta", default="datasets/audio/raw/librispeech/LibriSpeech")
    parser.add_argument("--audio-saida", default="datasets/audio/processed")
    parser.add_argument("--video-raw", default="datasets/video/raw")
    parser.add_argument("--video-frames", default="datasets/video/frames")
    parser.add_argument("--video-saida", default="datasets/video/processed")
    parser.add_argument("--tokenizer", default="datasets_storage/processed/tokenizer")
    parser.add_argument("--max-arquivos", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--frame-size", type=int, default=64)
    parser.add_argument("--manifesto", default="dataset_manifest.json")
    args = parser.parse_args()

    alvos = ["code", "images", "audio", "video"] if args.todas else [args.modalidade]
    alvos = [a for a in alvos if a]

    manifesto = DatasetManifest(RAIZ / args.manifesto)
    resultado: Dict[str, Any] = {}
    for modalidade in alvos:
        print(f"\n=== preparando {modalidade} ===")
        if modalidade == "code":
            resultado["code"] = preparar_code(args)
            manifesto.definir_status(
                "codigo_github_permissivo",
                status_preparacao="concluido",
                amostras=resultado["code"].get("arquivos"),
            )
        elif modalidade == "images":
            resultado["images"] = preparar_images(args)
            manifesto.definir_status(
                "flickr8k_test",
                status_preparacao="concluido",
                amostras=resultado["images"]["imagens"],
            )
        elif modalidade == "audio":
            resultado["audio"] = preparar_audio(args)
            manifesto.definir_status(
                "librispeech_dev_clean",
                status_preparacao="concluido",
                amostras=resultado["audio"]["falas"],
            )
        elif modalidade == "video":
            resultado["video"] = preparar_video(args)
            manifesto.definir_status(
                "wikimedia_videos",
                status_preparacao="concluido",
                amostras=resultado["video"]["videos"],
            )

    destino = RAIZ / "datasets" / "manifests" / "preparacao.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[preparação] resumo gravado em {destino}")


if __name__ == "__main__":
    main()
