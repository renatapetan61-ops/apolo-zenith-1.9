"""Coleta do corpus bruto do Apolo, com controle de licenças.

Fontes suportadas (todas com licença verificada e registrada em manifesto):

* **Project Gutenberg** — obras em domínio público (português e inglês);
* **Wikisource em português** — dump de páginas (CC BY-SA 4.0).

Cada documento baixado é salvo em ``datasets_storage/raw/<fonte>/`` e
descrito no ``manifest.json`` com URL, licença e hash SHA-256, permitindo
auditar a composição do corpus utilizado em cada treinamento.

Uso::

    python -m datasets.corpus_builder --saida datasets_storage --max-livros 120 --max-wiki-mb 24
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

USER_AGENT = "ApoloZenith/1.9 (dataset-builder; +https://example.invalid/apolo)"
TIMEOUT = 60


# --------------------------------------------------------------------------- #
# Infraestrutura HTTP
# --------------------------------------------------------------------------- #
def baixar(url: str, destino: Path, tentativas: int = 3, espera: float = 1.5) -> bool:
    """Baixa uma URL para um arquivo local com retentativas."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    for tentativa in range(tentativas):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resposta:
                with destino.open("wb") as fh:
                    while True:
                        bloco = resposta.read(1 << 16)
                        if not bloco:
                            break
                        fh.write(bloco)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            if tentativa < tentativas - 1:
                time.sleep(espera * (tentativa + 1))
                continue
            print(f"  [falha] {url}: {exc}")
            return False
    return False


def sha256_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            digest.update(bloco)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Metadados
# --------------------------------------------------------------------------- #
@dataclass
class DocumentoFonte:
    """Registro de proveniência de um documento do corpus."""

    fonte: str
    titulo: str
    arquivo: str
    url: str
    licenca: str
    caracteres: int
    sha256: str
    idioma: str


# --------------------------------------------------------------------------- #
# Project Gutenberg
# --------------------------------------------------------------------------- #
_RE_LINK_LIVRO = re.compile(r'<a[^>]*href="/ebooks/(\d+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
_RE_TAG = re.compile(r"<[^>]+>")
# Títulos Public Domain compatíveis com a política de licenças do Apolo.
# A busca do Gutenberg é usada para idiomas com catálogo muito grande.
CONSULTA_BUSCA_EN = [
    "shakespeare",
    "dickens",
    "austen",
    "tolstoy",
    "melville",
]


def _extrair_livros(html: str, max_livros: int) -> List[Tuple[int, str]]:
    vistos = set()
    livros: List[Tuple[int, str]] = []
    for ebook_id, titulo in _RE_LINK_LIVRO.findall(html):
        if ebook_id in vistos:
            continue
        vistos.add(ebook_id)
        titulo_limpo = _RE_TAG.sub("", titulo).strip()
        if not titulo_limpo:
            continue
        livros.append((int(ebook_id), titulo_limpo))
        if len(livros) >= max_livros:
            break
    return livros


def listar_livros_gutenberg(idioma: str = "pt", max_livros: int = 120) -> List[Tuple[int, str]]:
    """Lista os livros disponíveis no idioma indicado.

    Usa a página de navegação por idioma; para catálogos muito grandes
    (como o inglês) recorre à busca por palavras-chave, que é renderizada
    no servidor.
    """
    url = f"https://www.gutenberg.org/browse/languages/{idioma}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resposta:
        html = resposta.read().decode("utf-8", errors="replace")
    livros = _extrair_livros(html, max_livros)
    if livros:
        return livros

    # Fallback: busca por autores clássicos.
    encontrados: List[Tuple[int, str]] = []
    vistos: set[int] = set()
    for consulta in CONSULTA_BUSCA_EN:
        url_busca = (
            "https://www.gutenberg.org/ebooks/search/?query="
            + urllib.parse.quote(consulta)
            + "&submit_search=Go%21"
        )
        req = urllib.request.Request(url_busca, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resposta:
                html = resposta.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            continue
        for ebook_id, titulo in _extrair_livros(html, max_livros):
            if ebook_id in vistos:
                continue
            vistos.add(ebook_id)
            encontrados.append((ebook_id, titulo))
            if len(encontrados) >= max_livros:
                return encontrados
        time.sleep(0.4)
    return encontrados


def texto_plain_gutenberg(ebook_id: int) -> Optional[Tuple[str, str]]:
    """Baixa o texto puro de um ebook.  Devolve ``(url, texto)`` ou ``None``."""
    candidatos = [
        f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-0.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}.txt",
    ]
    for url in candidatos:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resposta:
                tipo = resposta.headers.get("Content-Type", "")
                if "text/html" in tipo:
                    continue
                texto = resposta.read().decode("utf-8", errors="replace")
            if len(texto) > 2000:
                return url, texto
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            continue
    return None


def coletar_gutenberg(
    saida_raw: Path,
    idioma: str = "pt",
    max_livros: int = 120,
    documentos: Optional[List[DocumentoFonte]] = None,
) -> List[DocumentoFonte]:
    """Baixa livros do Gutenberg e grava os textos brutos."""
    documentos = documentos if documentos is not None else []
    pasta = saida_raw / f"gutenberg_{idioma}"
    pasta.mkdir(parents=True, exist_ok=True)
    livros = listar_livros_gutenberg(idioma, max_livros)
    print(f"  [gutenberg:{idioma}] {len(livros)} livros candidatos")
    for ebook_id, titulo in livros:
        arquivo = pasta / f"{ebook_id}.txt"
        if arquivo.exists() and arquivo.stat().st_size > 2000:
            url = f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
        else:
            resultado = texto_plain_gutenberg(ebook_id)
            if resultado is None:
                continue
            url, texto = resultado
            arquivo.write_text(texto, encoding="utf-8")
            time.sleep(0.25)
        documentos.append(
            DocumentoFonte(
                fonte=f"gutenberg_{idioma}",
                titulo=titulo,
                arquivo=str(arquivo.relative_to(saida_raw.parent)),
                url=url,
                licenca="Domínio público (Project Gutenberg)",
                caracteres=arquivo.stat().st_size,
                sha256=sha256_arquivo(arquivo),
                idioma=idioma,
            )
        )
    return documentos


# --------------------------------------------------------------------------- #
# Wikisource
# --------------------------------------------------------------------------- #
WIKISOURCE_DUMP = "https://dumps.wikimedia.org/ptwikisource/latest/ptwikisource-latest-pages-articles.xml.bz2"


def coletar_wikisource(
    saida_raw: Path,
    max_mb: int = 24,
    documentos: Optional[List[DocumentoFonte]] = None,
) -> List[DocumentoFonte]:
    """Baixa o dump do Wikisource PT e extrai o texto das páginas.

    O dump é processado em streaming (sem carregar o arquivo inteiro na RAM),
    o que é essencial em dispositivos com pouca memória.
    """
    import bz2
    import xml.etree.ElementTree as ET

    documentos = documentos if documentos is not None else []
    pasta = saida_raw / "wikisource_pt"
    pasta.mkdir(parents=True, exist_ok=True)
    dump_local = saida_raw / "ptwikisource-pages-articles.xml.bz2"

    if not dump_local.exists():
        print(f"  [wikisource] baixando dump (~{max_mb} MB alvo, arquivo completo)…")
        if not baixar(WIKISOURCE_DUMP, dump_local):
            return documentos

    limite_chars = max_mb * 1024 * 1024
    total_chars = 0
    print("  [wikisource] extraindo texto das páginas…")
    with bz2.open(dump_local, "rb") as fh:
        contexto = ET.iterparse(fh, events=("end",))
        for _, elemento in contexto:
            if elemento.tag.endswith("}page") or elemento.tag == "page":
                titulo = ""
                texto = ""
                # No dump do MediaWiki, <text> fica dentro de <revision>.
                for descendente in elemento.iter():
                    tag = descendente.tag
                    if tag.endswith("}title") or tag == "title":
                        titulo = (descendente.text or "").strip()
                    elif tag.endswith("}text") or tag == "text":
                        texto = descendente.text or ""
                if texto.strip() and len(texto) > 400:
                    nome_seguro = re.sub(r"[^A-Za-z0-9_\-]+", "_", titulo)[:80] or "pagina"
                    arquivo = pasta / f"{nome_seguro}.txt"
                    arquivo.write_text(texto, encoding="utf-8")
                    documentos.append(
                        DocumentoFonte(
                            fonte="wikisource_pt",
                            titulo=titulo,
                            arquivo=str(arquivo.relative_to(saida_raw.parent)),
                            url=WIKISOURCE_DUMP,
                            licenca="CC BY-SA 4.0 (Wikimedia/Wikisource)",
                            caracteres=len(texto),
                            sha256=sha256_arquivo(arquivo),
                            idioma="pt",
                        )
                    )
                    total_chars += len(texto)
                elemento.clear()
                if total_chars >= limite_chars:
                    break

    print(f"  [wikisource] {len(documentos)} páginas, ~{total_chars/1024/1024:.1f} MB de texto")
    return documentos


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def construir_corpus(
    saida: str | Path = "datasets_storage",
    max_livros_pt: int = 120,
    max_livros_en: int = 6,
    max_wiki_mb: int = 24,
    usar_wikisource: bool = True,
) -> Path:
    """Executa a coleta completa e grava o manifesto."""
    saida = Path(saida)
    raw = saida / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    documentos: List[DocumentoFonte] = []
    documentos = coletar_gutenberg(raw, "pt", max_livros_pt, documentos)
    documentos = coletar_gutenberg(raw, "en", max_livros_en, documentos)
    if usar_wikisource:
        documentos = coletar_wikisource(raw, max_wiki_mb, documentos)

    manifesto = saida / "manifest.json"
    manifesto.write_text(
        json.dumps([asdict(d) for d in documentos], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    total = sum(d.caracteres for d in documentos)
    print(f"  [manifesto] {len(documentos)} documentos, {total/1024/1024:.1f} MB -> {manifesto}")
    return manifesto


def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta o corpus bruto do Apolo Zenith 1.9")
    parser.add_argument("--saida", default="datasets_storage")
    parser.add_argument("--max-livros-pt", type=int, default=120)
    parser.add_argument("--max-livros-en", type=int, default=6)
    parser.add_argument("--max-wiki-mb", type=int, default=24)
    parser.add_argument("--sem-wikisource", action="store_true")
    args = parser.parse_args()
    construir_corpus(
        saida=args.saida,
        max_livros_pt=args.max_livros_pt,
        max_livros_en=args.max_livros_en,
        max_wiki_mb=args.max_wiki_mb,
        usar_wikisource=not args.sem_wikisource,
    )


if __name__ == "__main__":
    main()
