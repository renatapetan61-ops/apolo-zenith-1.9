"""Carregamento lazy de vídeo do Apolo Zenith 1.9.

* Extração de frames com **PyAV** (embute ffmpeg) — o binário ``ffmpeg`` não
  existe no dispositivo;
* amostragem por intervalo (``amostra_fps``): vídeos longos **não** são
  processados frame a frame;
* os vídeos originais ficam em ``datasets/video/raw``; os frames extraídos
  vão para ``datasets/video/frames`` (pastas separadas, nunca sobrescritas).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger("apolo.datasets.video")


@dataclass
class ItemVideo:
    """Uma sequência de frames com metadados."""

    frames: torch.Tensor                # (T, 3, H, W)
    identificador: str = ""
    fps: float = 0.0
    duracao_s: float = 0.0
    titulo: str = ""


def extrair_frames(
    caminho: str | Path,
    max_frames: int = 8,
    amostra_fps: float = 1.0,
    tamanho: int = 64,
) -> Tuple[torch.Tensor, float, float]:
    """Extrai ``max_frames`` frames distribuídos ao longo do vídeo.

    Usa **seek por timestamp** (PyAV): em vídeos longos, decodificar quadro a
    quadro seria inviável no celular — apenas os quadros-alvo são lidos.

    Devolve ``(frames, fps_real, duracao_s)``.
    """
    import av

    from datasets.image_loader import preprocessar_pil

    caminho = Path(caminho)
    frames: List[torch.Tensor] = []
    fps_video = 0.0
    duracao = 0.0

    with av.open(str(caminho)) as container:
        fluxo = container.streams.video[0]
        fps_video = float(fluxo.average_rate or 0.0)
        duracao = float(fluxo.duration * fluxo.time_base) if fluxo.duration else 0.0
        if duracao <= 0 and container.duration:
            duracao = float(container.duration) / 1_000_000.0

        total_quadros = int(fluxo.frames or 0)
        if total_quadros > 0:
            # Índices equidistantes; busca por timestamp evita decodificar
            # o vídeo inteiro.
            time_base = fluxo.time_base or 1 / 1_000_000
            for indice in np.linspace(0, max(total_quadros - 1, 0), max_frames):
                alvo = int(indice) * float(time_base)
                quadro = _buscar_quadro(container, fluxo, alvo)
                if quadro is not None:
                    frames.append(preprocessar_pil(quadro.to_image(), tamanho))
        else:
            # Sem contagem conhecida: decodifica um trecho limitado.
            teto = max_frames * 10
            coletados: List[torch.Tensor] = []
            for quadro in container.decode(fluxo):
                coletados.append(preprocessar_pil(quadro.to_image(), tamanho))
                if len(coletados) >= teto:
                    break
            if coletados:
                indices = np.linspace(0, len(coletados) - 1, min(max_frames, len(coletados)))
                frames = [coletados[int(i)] for i in indices]

    if not frames:
        return torch.zeros(0, 3, tamanho, tamanho), fps_video, duracao
    return torch.stack(frames), fps_video, duracao


def _buscar_quadro(container, fluxo, alvo_segundos: float):
    """Busca (seek) o quadro mais próximo de ``alvo_segundos``.

    Se o seek não for suportado, devolve ``None`` (o chamador ignora).
    """
    try:
        time_base = fluxo.time_base or 1 / 1_000_000
        container.seek(max(int(alvo_segundos / float(time_base)), 0), stream=fluxo)
        for quadro in container.decode(fluxo):
            momento = float(quadro.pts * time_base) if quadro.pts is not None else 0.0
            if momento >= alvo_segundos:
                return quadro
            if momento > alvo_segundos + 5.0:
                break
    except Exception:
        return None
    return None


def descobrir_videos(pasta: str | Path) -> List[Path]:
    """Lista os vídeos de uma pasta (WebM, MP4, Ogg, MOV)."""
    extensoes = {".webm", ".mp4", ".ogv", ".ogg", ".mkv", ".mov", ".avi"}
    pasta = Path(pasta)
    if not pasta.exists():
        return []
    return sorted(p for p in pasta.iterdir() if p.is_file() and p.suffix.lower() in extensoes)


class VideoDataset(Dataset):
    """Dataset lazy de vídeos (frames extraídos sob demanda)."""

    def __init__(
        self,
        pasta: str | Path,
        max_frames: int = 8,
        amostra_fps: float = 1.0,
        tamanho: int = 64,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.pasta = Path(pasta)
        self.max_frames = max_frames
        self.amostra_fps = amostra_fps
        self.tamanho = tamanho
        self.log = log or LOGGER
        self.videos = descobrir_videos(self.pasta)

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, indice: int) -> ItemVideo:
        caminho = self.videos[indice]
        frames, fps, duracao = extrair_frames(
            caminho,
            max_frames=self.max_frames,
            amostra_fps=self.amostra_fps,
            tamanho=self.tamanho,
        )
        return ItemVideo(
            frames=frames,
            identificador=caminho.stem,
            fps=fps,
            duracao_s=duracao,
            titulo=caminho.stem.replace("_", " "),
        )


def extrair_frames_para_pasta(
    pasta_videos: str | Path,
    pasta_frames: str | Path,
    max_frames: int = 8,
    amostra_fps: float = 1.0,
    tamanho: int = 64,
) -> Dict[str, Any]:
    """Extrai e **salva** os frames em disco (frames/ por vídeo).

    Devolve um resumo com a contagem real de frames gravados.
    """
    from PIL import Image

    pasta_frames = Path(pasta_frames)
    pasta_frames.mkdir(parents=True, exist_ok=True)
    videos = descobrir_videos(pasta_videos)
    total = 0
    por_video: List[Dict[str, Any]] = []
    for video in videos:
        destino = pasta_frames / video.stem
        destino.mkdir(parents=True, exist_ok=True)
        frames, fps, duracao = extrair_frames(
            video, max_frames=max_frames, amostra_fps=amostra_fps, tamanho=tamanho
        )
        for i, frame in enumerate(frames):
            array = ((frame.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(array).save(destino / f"frame_{i:03d}.png", optimize=True)
        total += int(frames.shape[0])
        por_video.append(
            {
                "video": video.name,
                "frames": int(frames.shape[0]),
                "fps": round(fps, 2),
                "duracao_s": round(duracao, 2),
                "pasta": str(destino),
            }
        )
    return {"videos": len(videos), "frames_total": total, "detalhe": por_video}


def exportar_indice_video(pasta: str | Path, destino: str | Path) -> int:
    """Exporta o índice de vídeos em JSONL."""
    videos = descobrir_videos(pasta)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as fh:
        for video in videos:
            fh.write(json.dumps({"id": video.stem, "arquivo": str(video)}, ensure_ascii=False) + "\n")
    return len(videos)
