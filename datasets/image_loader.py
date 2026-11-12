"""Carregamento lazy de imagens do Apolo Zenith 1.9.

Estratégia (compatível com o limite de RAM do celular):

* o Parquet do Flickr8k é lido **por row group** (10 grupos de 100 linhas),
  nunca inteiro na memória;
* cada imagem é decodificada sob demanda com Pillow, redimensionada e
  normalizada;
* os arquivos originais (``datasets/images/raw``) **nunca** são alterados;
* o índice de legendas é gravado em JSONL para permitir streaming.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger("apolo.datasets.image")


@dataclass
class ItemImagem:
    """Uma imagem com suas legendas."""

    imagem: torch.Tensor                 # (3, H, W) normalizada
    legendas: List[str]
    identificador: str = ""

    @property
    def legenda(self) -> str:
        return self.legendas[0] if self.legendas else ""


def normalizar_imagem(array: np.ndarray) -> np.ndarray:
    """Normaliza para ``[-1, 1]`` (float32, HxWx3)."""
    array = array.astype(np.float32) / 255.0
    return (array - 0.5) / 0.5


def preprocessar_pil(imagem, tamanho: int = 64, augmentar: bool = False, rng=None) -> torch.Tensor:
    """Redimensiona (quadrado) e normaliza uma imagem PIL.

    Com ``augmentar=True`` aplica o conjunto padrão de augmentações de
    visão (recorte aleatório, espelhamento, jitter de cor, escala de
    cinza).  A augmentação é essencial: sem ela o encoder **memoriza** os
    pixels do conjunto pequeno e não generaliza (verificado: 100% no
    treino, 0% na validação).
    """
    from PIL import Image

    if imagem.mode != "RGB":
        imagem = imagem.convert("RGB")
    if augmentar:
        import numpy as _np

        gerador = rng if rng is not None else _np.random.default_rng()
        largura, altura = imagem.size
        lado = min(largura, altura)
        # Recorte aleatório (com margem de 20% a 100% do lado menor).
        escala = float(gerador.uniform(0.6, 1.0))
        novo = max(8, int(lado * escala))
        x0 = int(gerador.integers(0, max(1, largura - novo + 1)))
        y0 = int(gerador.integers(0, max(1, altura - novo + 1)))
        imagem = imagem.crop((x0, y0, x0 + novo, y0 + novo))
        if bool(gerador.random() < 0.5):
            imagem = imagem.transpose(Image.FLIP_LEFT_RIGHT)
        # Jitter de brilho/contraste.
        array = _np.asarray(imagem, dtype=_np.float32)
        brilho = float(gerador.uniform(0.8, 1.2))
        contraste = float(gerador.uniform(0.8, 1.2))
        array = (array - 128.0) * contraste + 128.0
        array = array * brilho
        if bool(gerador.random() < 0.15):
            cinza = array.mean(axis=2, keepdims=True)
            array = cinza + (array - cinza) * float(gerador.uniform(0.0, 0.6))
        imagem = Image.fromarray(_np.clip(array, 0, 255).astype(_np.uint8))
    imagem = imagem.resize((tamanho, tamanho), Image.BILINEAR)
    array = np.asarray(imagem, dtype=np.float32)
    tensor = torch.from_numpy(normalizar_imagem(array)).permute(2, 0, 1)
    return tensor


class Flickr8kDataset(Dataset):
    """Dataset lazy de pares imagem-legenda (Flickr8k em Parquet)."""

    def __init__(
        self,
        caminho_parquet: str | Path,
        tamanho: int = 64,
        max_itens: Optional[int] = None,
        log: Optional[logging.Logger] = None,
    ) -> None:
        import pyarrow.parquet as pq

        self.caminho = Path(caminho_parquet)
        self.tamanho = tamanho
        self.log = log or LOGGER
        self._arquivo = pq.ParquetFile(self.caminho)
        self.num_linhas = self._arquivo.metadata.num_rows
        self.max_itens = min(self.num_linhas, max_itens) if max_itens else self.num_linhas
        self._grupo_atual = -1
        self._tabela = None
        self._linhas: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return self.max_itens

    def _carregar_grupo(self, indice: int) -> None:
        """Carrega apenas o row group que contém o índice."""
        grupo = indice // 100
        if grupo == self._grupo_atual:
            return
        tabela = self._arquivo.read_row_group(grupo, columns=["image"])
        self._tabela = tabela
        self._grupo_atual = grupo
        self._linhas = []

    def _linha(self, indice: int) -> Dict[str, Any]:
        grupo = indice // 100
        if grupo != self._grupo_atual:
            self._grupo_atual = -1
            self._carregar_grupo(indice)
        local = indice - self._grupo_atual * 100
        if not self._linhas:
            self._linhas = self._tabela.to_pylist()
        return self._linhas[local]

    def __getitem__(self, indice: int) -> ItemImagem:
        from PIL import Image

        linha = self._linha(indice)
        bruto = linha["image"]["bytes"]
        with Image.open(io.BytesIO(bruto)) as imagem:
            tensor = preprocessar_pil(imagem, self.tamanho)
        return ItemImagem(imagem=tensor, legendas=[], identificador=linha["image"].get("path", ""))


class ParquetImagemLegendaDataset(Dataset):
    """Dataset lazy que já extrai as legendas dos mesmos Parquets.

    Aceita **um ou mais** arquivos Parquet (shards).  O índice global é
    mapeado para (arquivo, linha) e apenas o row group necessário é lido,
    mantendo o consumo de memória baixo mesmo com milhares de imagens.
    """

    def __init__(
        self,
        caminho_parquet: str | Path | Sequence[str | Path],
        tamanho: int = 64,
        max_itens: Optional[int] = None,
        colunas_legenda: Optional[List[str]] = None,
        log: Optional[logging.Logger] = None,
        augmentar: bool = False,
        seed: int = 1337,
    ) -> None:
        import pyarrow.parquet as pq

        if isinstance(caminho_parquet, (str, Path)):
            caminhos = [Path(caminho_parquet)]
        else:
            caminhos = [Path(c) for c in caminho_parquet]
        self.caminhos = caminhos
        self.tamanho = tamanho
        self.log = log or LOGGER
        self.augmentar = augmentar
        self._rng = np.random.default_rng(seed)
        self._arquivos = [pq.ParquetFile(c) for c in caminhos]
        self.colunas_legenda = colunas_legenda or [
            nome for nome in self._arquivos[0].schema_arrow.names if nome.startswith("caption")
        ]
        # Índice global -> (arquivo, linha), agrupado por row group.
        self._mapa: List[Tuple[int, int]] = []
        self._grupos: List[Tuple[int, int, int]] = []   # (arquivo, grupo, início)
        for indice_arquivo, arquivo in enumerate(self._arquivos):
            total = arquivo.metadata.num_rows
            for grupo in range(arquivo.metadata.num_row_groups):
                inicio = grupo * 100
                fim = min(inicio + 100, total)
                self._grupos.append((indice_arquivo, grupo, inicio))
                for linha in range(inicio, fim):
                    self._mapa.append((indice_arquivo, linha))
        self.num_linhas = len(self._mapa)
        self.max_itens = min(self.num_linhas, max_itens) if max_itens else self.num_linhas
        self._grupo_atual = (-1, -1)
        self._tabela = None

    def ordem_por_grupo(self, semente: int = 1337) -> List[int]:
        """Ordem de acesso **agrupada por row group** (embaralhada).

        Acessar um Parquet fora de ordem obriga a reler o row group inteiro
        (dezenas de MB) a cada item — no celular isso domina o tempo de
        treino.  Esta ordem embaralha *dentro* dos grupos, mantendo a leitura
        sequencial.
        """
        gerador = np.random.default_rng(semente)
        ordem: List[int] = []
        grupos = list(self._grupos)
        gerador.shuffle(grupos)
        for indice_arquivo, grupo, inicio in grupos:
            bloco = list(range(inicio, min(inicio + 100, self.num_linhas)))
            gerador.shuffle(bloco)
            ordem.extend(bloco)
        return ordem[: self.max_itens]

    def __len__(self) -> int:
        return self.max_itens

    def __getitem__(self, indice: int) -> ItemImagem:
        from PIL import Image

        arquivo_idx, linha_global = self._mapa[indice]
        arquivo = self._arquivos[arquivo_idx]
        grupo = linha_global // 100
        if self._grupo_atual != (arquivo_idx, grupo):
            self._tabela = arquivo.read_row_group(grupo)
            self._grupo_atual = (arquivo_idx, grupo)
        local = linha_global - grupo * 100
        linha = self._tabela.slice(local, 1).to_pylist()[0]
        bruto = linha["image"]["bytes"]
        with Image.open(io.BytesIO(bruto)) as imagem:
            tensor = preprocessar_pil(
                imagem, self.tamanho, augmentar=self.augmentar, rng=self._rng
            )
        legendas = [str(linha[c]) for c in self.colunas_legenda if linha.get(c)]
        return ItemImagem(
            imagem=tensor,
            legendas=legendas,
            identificador=linha["image"].get("path", ""),
        )


class PastaImagemDataset(Dataset):
    """Dataset lazy de imagens em uma pasta (ex.: Wikimedia Commons)."""

    def __init__(
        self,
        pasta: str | Path,
        tamanho: int = 64,
        log: Optional[logging.Logger] = None,
    ) -> None:
        from PIL import Image

        self.pasta = Path(pasta)
        self.tamanho = tamanho
        self.log = log or LOGGER
        extensoes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
        self.arquivos = sorted(
            p for p in self.pasta.iterdir() if p.is_file() and p.suffix.lower() in extensoes
        )

    def __len__(self) -> int:
        return len(self.arquivos)

    def __getitem__(self, indice: int) -> ItemImagem:
        from PIL import Image

        caminho = self.arquivos[indice]
        with Image.open(caminho) as imagem:
            tensor = preprocessar_pil(imagem, self.tamanho)
        return ItemImagem(imagem=tensor, legendas=[], identificador=caminho.name)


def collate_imagens(lote: List[ItemImagem]) -> Dict[str, Any]:
    """Collate para o DataLoader: agrupa imagens e legendas.

    Cada imagem traz **todas** as suas legendas (o Flickr8k tem 5 por
    imagem); o trainer sorteia uma por época, multiplicando os pares de
    treino sem duplicar as imagens.
    """
    return {
        "imagens": torch.stack([item.imagem for item in lote]),
        "legendas": [item.legenda for item in lote],
        "todas_legendas": [item.legendas for item in lote],
        "identificadores": [item.identificador for item in lote],
    }


def exportar_legendas_jsonl(
    caminho_parquet: str | Path,
    destino: str | Path,
    max_itens: Optional[int] = None,
) -> int:
    """Exporta as legendas em JSONL (streaming, sem carregar as imagens)."""
    import pyarrow.parquet as pq

    arquivo = pq.ParquetFile(caminho_parquet)
    total = arquivo.metadata.num_rows
    limite = min(total, max_itens) if max_itens else total
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    escritos = 0
    with destino.open("w", encoding="utf-8") as fh:
        indice = 0
        for grupo in range(arquivo.metadata.num_row_groups):
            tabela = arquivo.read_row_group(
                grupo, columns=[c for c in arquivo.schema_arrow.names if c.startswith("caption")]
            )
            for linha in tabela.to_pylist():
                if indice >= limite:
                    break
                legendas = [
                    str(linha[c]) for c in linha if str(linha[c] or "").strip()
                ]
                if legendas:
                    fh.write(json.dumps({"id": indice, "legendas": legendas}, ensure_ascii=False) + "\n")
                    escritos += 1
                indice += 1
            if indice >= limite:
                break
    return escritos
