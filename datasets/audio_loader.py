"""Carregamento lazy de áudio do Apolo Zenith 1.9 (LibriSpeech).

* Percorre os arquivos FLAC **um a um** (nunca carrega o conjunto inteiro);
* calcula o mel-spectrogram sob demanda (com cache opcional em disco);
* mantém a transcrição associada ao áudio;
* os FLAC originais permanecem em ``datasets/audio/raw`` — os espectrogramas
  processados vão para ``datasets/audio/processed``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger("apolo.datasets.audio")


@dataclass
class ItemAudio:
    """Um áudio com a transcrição associada."""

    mel: torch.Tensor                  # (n_mels, frames)
    transcricao: str
    identificador: str = ""
    duracao_s: float = 0.0


def ler_flac(caminho: str | Path) -> Tuple[np.ndarray, int]:
    """Lê um arquivo de áudio (FLAC/WAV) com soundfile."""
    import soundfile as sf

    onda, taxa = sf.read(str(caminho), dtype="float32", always_2d=False)
    return onda, int(taxa)


def preparar_mel(
    caminho: str | Path,
    n_mels: int = 80,
    max_seconds: float = 3.0,
    taxa_alvo: int = 16000,
) -> Tuple[np.ndarray, float]:
    """Calcula o mel-spectrogram normalizado de um arquivo de áudio."""
    import librosa

    from audio.encoder import mel_frontend, normalizar_mel
    from audio.encoder import AudioConfig

    onda, taxa = ler_flac(caminho)
    if taxa != taxa_alvo:
        onda = librosa.resample(onda, orig_sr=taxa, target_sr=taxa_alvo)
    duracao = len(onda) / taxa_alvo
    limite = int(max_seconds * taxa_alvo)
    if len(onda) > limite:
        onda = onda[:limite]
    config = AudioConfig(n_mels=n_mels, max_seconds=max_seconds, sample_rate=taxa_alvo)
    mel = mel_frontend(onda, config)
    return normalizar_mel(mel), duracao


class LibriSpeechDataset(Dataset):
    """Dataset lazy de pares áudio-transcrição do LibriSpeech."""

    def __init__(
        self,
        pasta: str | Path,
        n_mels: int = 80,
        max_seconds: float = 3.0,
        max_itens: Optional[int] = None,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.pasta = Path(pasta)
        self.n_mels = n_mels
        self.max_seconds = max_seconds
        self.log = log or LOGGER
        self.itens: List[Dict[str, Any]] = self._indexar()
        if max_itens:
            self.itens = self.itens[:max_itens]

    def _indexar(self) -> List[Dict[str, Any]]:
        """Indexa os pares .flac/transcrição (apenas metadados).

        O LibriSpeech organiza as transcrições em ``<capitulo>.trans.txt``
        (linhas ``ID TEXTO``); os arquivos ``.flac`` são casados pelo ID.
        """
        if not self.pasta.exists():
            return []

        transcricoes: Dict[str, str] = {}
        for arquivo_txt in self.pasta.rglob("*.trans.txt"):
            try:
                for linha in arquivo_txt.read_text(encoding="utf-8", errors="replace").splitlines():
                    linha = linha.strip()
                    if not linha:
                        continue
                    identificador, _, texto = linha.partition(" ")
                    if texto:
                        transcricoes[identificador] = texto
            except OSError:
                continue
        # Fallback: arquivos .txt individuais (outros corpora).
        for arquivo_txt in self.pasta.rglob("*.txt"):
            if arquivo_txt.name.endswith(".trans.txt"):
                continue
            try:
                texto = arquivo_txt.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            if texto:
                transcricoes.setdefault(arquivo_txt.stem, texto)

        itens: List[Dict[str, Any]] = []
        for flac in sorted(self.pasta.rglob("*.flac")):
            texto = transcricoes.get(flac.stem, "")
            if not texto:
                continue
            try:
                duracao = _duracao(flac)
            except Exception:
                continue
            itens.append(
                {
                    "id": flac.stem,
                    "flac": str(flac),
                    "txt": str(flac.with_suffix(".txt")),
                    "transcricao": texto,
                    "duracao_s": duracao,
                }
            )
        return itens

    def __len__(self) -> int:
        return len(self.itens)

    def __getitem__(self, indice: int) -> ItemAudio:
        item = self.itens[indice]
        mel, duracao = preparar_mel(
            item["flac"], n_mels=self.n_mels, max_seconds=self.max_seconds
        )
        return ItemAudio(
            mel=torch.from_numpy(mel),
            transcricao=item["transcricao"],
            identificador=item["id"],
            duracao_s=duracao,
        )

    def indice_legendas(self) -> List[Dict[str, Any]]:
        """Índice (id → transcrição) sem carregar o áudio."""
        return [
            {"id": item["id"], "transcricao": item["transcricao"], "duracao_s": item["duracao_s"]}
            for item in self.itens
        ]


def _duracao(caminho: Path) -> float:
    import soundfile as sf

    info = sf.info(str(caminho))
    return float(info.frames) / float(info.samplerate or 16000)


def exportar_indice_audio(
    pasta: str | Path,
    destino: str | Path,
    max_itens: Optional[int] = None,
) -> int:
    """Exporta o índice de transcrições em JSONL (streaming)."""
    dataset = LibriSpeechDataset(pasta, max_itens=max_itens)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as fh:
        for item in dataset.indice_legendas():
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(dataset)
