"""Pipeline de dados multimodais do Apolo Zenith 1.9.

Subpacotes:

* ``text_loader``   — corpus textual (limpeza, PII, deduplicação, empacotamento);
* ``code_loader``   — corpus de código (coleta por linguagem, tokenização);
* ``image_loader``  — imagens lazy (Parquet por row group) + augmentação;
* ``audio_loader``  — áudio lazy (FLAC → mel-spectrogram sob demanda);
* ``video_loader``  — vídeo lazy (PyAV, amostragem por intervalo);
* ``catalog``       — catálogo de datasets com licenças e URLs;
* ``downloader``    — download verificado (SHA-256, blocos, espaço livre);
* ``manifest``      — ``dataset_manifest.json``.
"""

from datasets.audio_loader import ItemAudio, LibriSpeechDataset
from datasets.code_loader import coletar_codigo, preparar_corpus_codigo
from datasets.image_loader import (
    ItemImagem,
    ParquetImagemLegendaDataset,
    PastaImagemDataset,
    collate_imagens,
)
from datasets.manifest import DatasetManifest, EntradaManifesto
from datasets.text_loader import (
    PackedTokenDataset,
    carregar_tokens,
    deduplicar,
    dividir_documentos,
    normalizar_texto,
    qualidade_texto,
    remover_pii,
    salvar_tokens,
)
from datasets.video_loader import ItemVideo, VideoDataset, extrair_frames

__all__ = [
    "PackedTokenDataset",
    "carregar_tokens",
    "salvar_tokens",
    "normalizar_texto",
    "remover_pii",
    "qualidade_texto",
    "deduplicar",
    "dividir_documentos",
    "coletar_codigo",
    "preparar_corpus_codigo",
    "ItemImagem",
    "ParquetImagemLegendaDataset",
    "PastaImagemDataset",
    "LazyJsonlDataset",
    "collate_imagens",
    "ItemAudio",
    "LibriSpeechDataset",
    "ItemVideo",
    "VideoDataset",
    "extrair_frames",
    "DatasetManifest",
    "EntradaManifesto",
]
