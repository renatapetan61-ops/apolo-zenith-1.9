"""Carregamento lazy de datasets do Apolo Zenith 1.9.

Objetivos:

* **não** carregar o dataset inteiro na RAM;
* manter um cache limitado por **bytes** (não por número de itens);
* validar a integridade **SHA-256** contra um arquivo de manifesto;
* **ignorar arquivos corrompidos**, registrando o motivo no log;
* permitir retomada (skip) de itens já processados.

Duas implementações são fornecidas:

* :class:`LazyShardDataset` – shards NumPy (``.npy``) ou binários (``.bin``);
* :class:`LazyJsonlDataset` – arquivos JSONL lidos linha a linha (streaming).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("apolo.datasets.lazy")

#: Tamanho padrão do bloco de leitura (streaming) – 1 MiB.
BLOCO_LEITURA = 1024 * 1024


# --------------------------------------------------------------------------- #
# Integridade
# --------------------------------------------------------------------------- #
def integridade_sha256(caminho: str | Path, bloco: int = BLOCO_LEITURA) -> str:
    """SHA-256 do arquivo, lido em blocos (não carrega tudo na RAM)."""
    digest = hashlib.sha256()
    with Path(caminho).open("rb") as fh:
        while True:
            dados = fh.read(bloco)
            if not dados:
                break
            digest.update(dados)
    return digest.hexdigest()


@dataclass
class ResultadoIntegridade:
    """Resultado da verificação de integridade de um arquivo."""

    caminho: str
    ok: bool
    sha256: Optional[str] = None
    esperado: Optional[str] = None
    motivo: str = ""

    def para_dict(self) -> Dict[str, Any]:
        return {
            "caminho": self.caminho,
            "ok": self.ok,
            "sha256": self.sha256,
            "esperado": self.esperado,
            "motivo": self.motivo,
        }


def _sha_esperado(caminho: Path) -> Optional[str]:
    """Lê o SHA-256 esperado de ``<arquivo>.sha256`` ou de um manifesto JSON."""
    sidecar = caminho.with_suffix(caminho.suffix + ".sha256")
    if sidecar.exists():
        try:
            texto = sidecar.read_text(encoding="utf-8").strip().split()[0]
            return texto.lower()
        except (OSError, IndexError):
            return None
    manifesto = caminho.parent / "manifest.json"
    if manifesto.exists():
        try:
            dados = json.loads(manifesto.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        entradas = dados.get("arquivos", dados) if isinstance(dados, dict) else {}
        if isinstance(entradas, dict):
            valor = entradas.get(caminho.name)
            if isinstance(valor, dict):
                return str(valor.get("sha256", "")).lower() or None
            if isinstance(valor, str):
                return valor.lower()
        if isinstance(entradas, list):
            for item in entradas:
                if isinstance(item, dict) and item.get("nome") == caminho.name:
                    return str(item.get("sha256", "")).lower() or None
    return None


def verificar_arquivo(
    caminho: str | Path,
    sha_esperado: Optional[str] = None,
    calcular: bool = True,
) -> ResultadoIntegridade:
    """Verifica se o arquivo existe e (opcionalmente) se o SHA-256 confere."""
    caminho = Path(caminho)
    if not caminho.exists():
        return ResultadoIntegridade(str(caminho), False, motivo="arquivo inexistente")
    if not caminho.is_file():
        return ResultadoIntegridade(str(caminho), False, motivo="não é um arquivo")
    esperado = (sha_esperado or _sha_esperado(caminho) or "").lower() or None
    if esperado is None or not calcular:
        return ResultadoIntegridade(
            str(caminho), True, esperado=esperado, motivo="sem hash esperado; aceito"
        )
    try:
        atual = integridade_sha256(caminho)
    except OSError as exc:
        return ResultadoIntegridade(str(caminho), False, esperado=esperado, motivo=f"erro de leitura: {exc}")
    if atual != esperado:
        return ResultadoIntegridade(
            str(caminho),
            False,
            sha256=atual,
            esperado=esperado,
            motivo="SHA-256 divergente",
        )
    return ResultadoIntegridade(str(caminho), True, sha256=atual, esperado=esperado, motivo="SHA-256 ok")


# --------------------------------------------------------------------------- #
# Dataset de shards
# --------------------------------------------------------------------------- #
@dataclass
class Shard:
    """Um shard do dataset."""

    caminho: Path
    sha256: Optional[str] = None
    tamanho_bytes: int = 0
    itens: int = 0
    valido: bool = True
    motivo: str = ""


class LazyShardDataset:
    """Dataset de shards carregado sob demanda, com cache limitado por bytes.

    Cada *item* é um array NumPy (``.npy``) ou um ``bytes`` (``.bin``).  Shards
    corrompidos são **ignorados** (com registro no log) em vez de derrubar o
    treino.
    """

    def __init__(
        self,
        caminhos: Sequence[str | Path],
        cache_max_mb: int = 256,
        validar_sha256: bool = True,
        formato: Optional[str] = None,
        log: Optional[logging.Logger] = None,
        pular_indice: int = 0,
    ) -> None:
        self.log = log or LOGGER
        self.cache_max_bytes = max(0, int(cache_max_mb) * 1024 * 1024)
        self.validar_sha256 = validar_sha256
        self.formato = formato
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self._cache_bytes = 0
        self.shards: List[Shard] = []
        self._carregar_shards(caminhos)
        self._pular_indice = max(0, int(pular_indice))

    # ------------------------------------------------------------------ #
    def _carregar_shards(self, caminhos: Sequence[str | Path]) -> None:
        for bruto in caminhos:
            caminho = Path(bruto)
            if not caminho.exists():
                self.shards.append(
                    Shard(caminho=caminho, valido=False, motivo="arquivo inexistente")
                )
                self.log.warning("Shard ignorado (inexistente): %s", caminho)
                continue
            resultado = (
                verificar_arquivo(caminho)
                if self.validar_sha256
                else ResultadoIntegridade(str(caminho), True)
            )
            if not resultado.ok:
                self.shards.append(
                    Shard(
                        caminho=caminho,
                        sha256=resultado.sha256,
                        tamanho_bytes=caminho.stat().st_size if caminho.exists() else 0,
                        valido=False,
                        motivo=resultado.motivo,
                    )
                )
                self.log.error(
                    "Shard corrompido ignorado: %s (%s)", caminho, resultado.motivo
                )
                continue
            tamanho = caminho.stat().st_size
            itens = self._contar_itens(caminho)
            self.shards.append(
                Shard(
                    caminho=caminho,
                    sha256=resultado.sha256,
                    tamanho_bytes=tamanho,
                    itens=itens,
                    valido=True,
                )
            )

    def _contar_itens(self, caminho: Path) -> int:
        """Número de itens do shard (1 para arquivo único)."""
        if self.formato == "jsonl" or caminho.suffix.lower() == ".jsonl":
            total = 0
            try:
                with caminho.open("rb") as fh:
                    for _ in fh:
                        total += 1
            except OSError:
                return 0
            return total
        if caminho.suffix.lower() == ".npy":
            try:
                arr = np.load(caminho, mmap_mode="r")
                return int(arr.shape[0]) if arr.ndim > 0 else 1
            except (OSError, ValueError):
                return 0
        return 1

    # ------------------------------------------------------------------ #
    @property
    def shards_validos(self) -> List[Shard]:
        return [s for s in self.shards if s.valido]

    def __len__(self) -> int:
        return sum(s.itens for s in self.shards_validos) - self._pular_indice

    def _carregar(self, shard: Shard) -> Any:
        """Carrega um shard usando o cache LRU limitado por bytes."""
        chave = str(shard.caminho)
        if chave in self._cache:
            self._cache.move_to_end(chave)
            return self._cache[chave]

        sufixo = shard.caminho.suffix.lower()
        if sufixo == ".npy":
            dados: Any = np.load(shard.caminho)
        elif sufixo in (".jsonl", ".ndjson"):
            with shard.caminho.open("r", encoding="utf-8", errors="replace") as fh:
                dados = [json.loads(linha) for linha in fh if linha.strip()]
        else:
            dados = shard.caminho.read_bytes()

        tamanho = _tamanho_objeto(dados)
        self._cache[chave] = dados
        self._cache_bytes += tamanho
        while self._cache_bytes > self.cache_max_bytes and len(self._cache) > 1:
            _, removido = self._cache.popitem(last=False)
            self._cache_bytes -= _tamanho_objeto(removido)
        return dados

    def item(self, indice: int) -> Any:
        """Obtém o item ``indice`` (índice global sobre os shards válidos)."""
        indice += self._pular_indice
        if indice < 0:
            raise IndexError("índice negativo")
        acumulado = 0
        for shard in self.shards_validos:
            if indice < acumulado + shard.itens:
                dados = self._carregar(shard)
                if shard.itens == 1:
                    return dados
                if isinstance(dados, list):
                    return dados[indice - acumulado]
                return dados[indice - acumulado]
            acumulado += shard.itens
        raise IndexError(f"índice {indice} fora do intervalo (0..{acumulado - 1})")

    def __getitem__(self, indice: int) -> Any:
        return self.item(indice)

    def __iter__(self) -> Iterator[Any]:
        for i in range(len(self)):
            yield self.item(i)

    def limpar_cache(self) -> int:
        """Limpa o cache; devolve os bytes liberados."""
        liberado = self._cache_bytes
        self._cache.clear()
        self._cache_bytes = 0
        return liberado

    def resumo(self) -> Dict[str, Any]:
        return {
            "shards_total": len(self.shards),
            "shards_validos": len(self.shards_validos),
            "shards_ignorados": len(self.shards) - len(self.shards_validos),
            "itens": len(self),
            "bytes_cache": self._cache_bytes,
            "cache_max_bytes": self.cache_max_bytes,
            "corrompidos": [
                {"caminho": str(s.caminho), "motivo": s.motivo}
                for s in self.shards
                if not s.valido
            ],
        }


def _tamanho_objeto(dados: Any) -> int:
    """Tamanho aproximado de um objeto em memória (bytes)."""
    if isinstance(dados, np.ndarray):
        return int(dados.nbytes)
    if isinstance(dados, (bytes, bytearray, memoryview)):
        return len(dados)
    if isinstance(dados, list):
        return sum(_tamanho_objeto(d) for d in dados)
    if isinstance(dados, dict):
        return sum(_tamanho_objeto(v) for v in dados.values())
    if isinstance(dados, str):
        return len(dados.encode("utf-8", errors="ignore"))
    return 64


# --------------------------------------------------------------------------- #
# Dataset JSONL em streaming
# --------------------------------------------------------------------------- #
class LazyJsonlDataset:
    """Dataset JSONL lido linha a linha, sem carregar o arquivo inteiro.

    Mantém um índice de offsets (construído uma vez) para acesso aleatório
    sem carregar o arquivo na RAM.
    """

    def __init__(
        self,
        caminho: str | Path,
        pular_linhas: int = 0,
        log: Optional[logging.Logger] = None,
        validar_sha256: bool = False,
    ) -> None:
        self.caminho = Path(caminho)
        self.log = log or LOGGER
        self.pular_linhas = max(0, int(pular_linhas))
        self._offsets: List[int] = []
        self._offset_por_linha: Dict[int, Optional[int]] = {}
        self._corrompidos: List[int] = []
        self._corrompidos_set: set[int] = set()
        self._indice_ok: List[int] = []
        self._indice_arquivo = self.caminho
        self._construir_indice()

    def _construir_indice(self) -> None:
        if not self.caminho.exists():
            self.log.error("Arquivo JSONL inexistente: %s", self.caminho)
            return
        offset = 0
        with self.caminho.open("rb") as fh:
            for numero, linha in enumerate(fh):
                tamanho = len(linha)
                if linha.strip():
                    # Offset do início de cada linha física (não vazia).
                    self._offset_por_linha[numero] = offset
                    self._offsets.append(offset)
                    try:
                        json.loads(linha.decode("utf-8", errors="replace"))
                        self._indice_ok.append(numero)
                    except json.JSONDecodeError:
                        self._corrompidos.append(numero)
                        self._corrompidos_set.add(numero)
                        self.log.warning(
                            "Linha JSONL corrompida ignorada: %s linha %d", self.caminho, numero + 1
                        )
                offset += tamanho

    def __len__(self) -> int:
        return max(0, len(self._indice_ok) - self.pular_linhas)

    def __getitem__(self, indice: int) -> Dict[str, Any]:
        if indice < 0:
            indice += len(self)
        if indice < 0 or indice >= len(self):
            raise IndexError(f"índice {indice} fora do intervalo (0..{len(self) - 1})")
        posicao = self._indice_ok[indice + self.pular_linhas]
        with self.caminho.open("rb") as fh:
            fh.seek(self._offset_por_linha[posicao])
            linha = fh.readline()
        return json.loads(linha.decode("utf-8", errors="replace"))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        with self.caminho.open("r", encoding="utf-8", errors="replace") as fh:
            emitida = 0
            for numero, linha in enumerate(fh):
                if not linha.strip():
                    continue
                if numero in self._corrompidos_set:
                    continue
                if emitida < self.pular_linhas:
                    emitida += 1
                    continue
                emitida += 1
                yield json.loads(linha)

    def resumo(self) -> Dict[str, Any]:
        return {
            "arquivo": str(self.caminho),
            "linhas_validas": len(self._indice_ok),
            "linhas_corrompidas": len(self._corrompidos),
            "pular_linhas": self.pular_linhas,
        }
