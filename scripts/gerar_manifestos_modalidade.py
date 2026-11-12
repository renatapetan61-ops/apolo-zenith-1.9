"""Gera os manifestos por modalidade (``datasets/<modalidade>/manifests/``).

Cada arquivo resume, com **valores reais**, o que foi baixado, preparado e
utilizado naquela modalidade:

* ``resumo.json``     – contagem de amostras, tamanhos e formatos;
* ``fontes.json``     – origem, URL e licença de cada fonte;
* ``uso.json``        – como os dados foram usados no treinamento.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from datasets.manifest import DatasetManifest  # noqa: E402

USO = {
    "text": (
        "Tokenizado com o BPE do projeto e empacotado em sequências de 64 tokens "
        "(train.bin/val.bin/test.bin). Treinou o Language Core (1.901.696 params)."
    ),
    "code": (
        "Tokenizado com o mesmo BPE e empacotado; 13 linguagens marcadas no "
        "texto. Treinou a especialização de código do Language Core."
    ),
    "images": (
        "Parquet lido por row group; imagens redimensionadas para 64x64 e "
        "normalizadas; 5 legendas por imagem. Treino contrastivo (InfoNCE) do "
        "ViT + projetor contra os embeddings de texto do Language Core."
    ),
    "audio": (
        "FLAC lidos um a um; mel-spectrogram (80 bandas, 16 kHz) calculado sob "
        "demanda e normalizado por arquivo. Treino contrastivo do encoder de "
        "áudio + projetor contra as transcrições."
    ),
    "video": (
        "Frames extraídos com PyAV por amostragem de índice (vídeos longos não "
        "são processados frame a frame); 8 frames por vídeo, 64x64. Treino do "
        "encoder temporal + cabeça de ordem temporal."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Manifestos por modalidade")
    parser.add_argument("--manifesto", default="dataset_manifest.json")
    args = parser.parse_args()

    manifesto = DatasetManifest(RAIZ / args.manifesto)
    entradas = manifesto.entradas.values()

    for modalidade in ("text", "code", "images", "audio", "video"):
        pasta = RAIZ / "datasets" / modalidade / "manifests"
        pasta.mkdir(parents=True, exist_ok=True)
        da_modalidade = [e for e in entradas if e.modalidade == modalidade]

        resumo: Dict[str, Any] = {
            "modalidade": modalidade,
            "datasets": [
                {
                    "nome": e.nome,
                    "origem": e.origem,
                    "url_oficial": e.url_oficial,
                    "licenca": e.licenca,
                    "formato": e.formato,
                    "amostras": e.amostras,
                    "bytes_baixado": e.tamanho_bytes,
                    "bytes_em_disco": e.tamanho_em_disco_bytes,
                    "status_download": e.status_download,
                    "status_preparacao": e.status_preparacao,
                    "status_treino": e.status_treino,
                }
                for e in da_modalidade
            ],
            "uso_no_treinamento": USO.get(modalidade, ""),
        }
        (pasta / "resumo.json").write_text(
            json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        fontes = {
            "modalidade": modalidade,
            "fontes": [
                {
                    "nome": e.nome,
                    "origem": e.origem,
                    "url_oficial": e.url_oficial,
                    "licenca": e.licenca,
                    "detalhe": e.subfontes,
                }
                for e in da_modalidade
            ],
        }
        (pasta / "fontes.json").write_text(
            json.dumps(fontes, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[manifestos] {modalidade}: {len(da_modalidade)} dataset(s) -> {pasta}")


if __name__ == "__main__":
    main()
