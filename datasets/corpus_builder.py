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

USER_AGENT = "ApoloZenith/1.9 (dataset-builder; +https://github.com/renatapetan61-ops/apolo-zenith-1.9)"
TIMEOUT = 20


# --------------------------------------------------------------------------- #
# Infraestrutura HTTP
# --------------------------------------------------------------------------- #
def _ler_url_texto(url: str, tentativas: int = 5, espera: float = 2.0) -> Optional[str]:
    """Lê uma página HTTP com retry para 429/5xx e falhas transitórias."""
    for tentativa in range(tentativas):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resposta:
                return resposta.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            transitório = exc.code == 429 or 500 <= exc.code <= 599
            if not transitório or tentativa == tentativas - 1:
                print(f"  [falha HTTP {exc.code}] {url}: {exc.reason}")
                return None
            retry_after = exc.headers.get("Retry-After")
            try:
                # Um servidor indisponível não deve bloquear o CI por minutos.
                # Respeitamos Retry-After, mas com teto curto e backoff limitado.
                atraso = min(float(retry_after), 5.0) if retry_after else min(espera * (2 ** tentativa), 5.0)
            except (TypeError, ValueError):
                atraso = min(espera * (2 ** tentativa), 5.0)
            print(f"  [retry {tentativa + 1}/{tentativas - 1}] HTTP {exc.code}; aguardando {atraso:.1f}s")
            time.sleep(atraso)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if tentativa == tentativas - 1:
                print(f"  [falha de rede] {url}: {exc}")
                return None
            atraso = espera * (2 ** tentativa)
            print(f"  [retry {tentativa + 1}/{tentativas - 1}] falha de rede; aguardando {atraso:.1f}s")
            time.sleep(atraso)
    return None


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


def ler_texto_http(url: str, tentativas: int = 5) -> str:
    """Lê página HTTP com retry para falhas transitórias (429/5xx/rede)."""
    ultimo_erro: Optional[BaseException] = None
    for tentativa in range(tentativas):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resposta:
                return resposta.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            ultimo_erro = exc
            if not (exc.code == 429 or 500 <= exc.code <= 599):
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            ultimo_erro = exc
        if tentativa < tentativas - 1:
            time.sleep(min(2 ** tentativa, 20))
    if ultimo_erro is not None:
        raise ultimo_erro
    raise RuntimeError("Falha HTTP sem resposta para " + url)


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
    html = _ler_url_texto(url, tentativas=3, espera=1.0)
    livros = _extrair_livros(html, max_livros) if html else []
    # Se o host estiver retornando HTTP 5xx, não multiplique a indisponibilidade
    # por cinco buscas redundantes. O fallback Wikipedia será usado pelo pipeline.
    if livros or idioma != "en" or html is None:
        return livros

    # Fallback: busca por autores clássicos somente para o catálogo inglês.
    encontrados: List[Tuple[int, str]] = []
    vistos: set[int] = set()
    for consulta in CONSULTA_BUSCA_EN:
        url_busca = (
            "https://www.gutenberg.org/ebooks/search/?query="
            + urllib.parse.quote(consulta)
            + "&submit_search=Go%21"
        )
        html = _ler_url_texto(url_busca, tentativas=3, espera=1.0)
        if not html:
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
            with urllib.request.urlopen(req, timeout=8) as resposta:
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
# Wikipedia — fallback independente do Project Gutenberg
# --------------------------------------------------------------------------- #
WIKIPEDIA_BUSCAS = {
    "pt": ["literatura brasileira", "história do Brasil", "ciência no Brasil", "geografia do Brasil", "língua portuguesa", "biologia", "física", "matemática", "meio ambiente", "tecnologia", "cultura brasileira", "saúde pública"],
    "en": ["world history", "science", "literature", "geography", "biology", "physics", "mathematics", "environment", "technology", "public health", "computer science", "language"],
}


def coletar_wikipedia(
    saida_raw: Path,
    idioma: str,
    max_documentos: int,
    documentos: Optional[List[DocumentoFonte]] = None,
) -> List[DocumentoFonte]:
    """Obtém artigos CC BY-SA 4.0 por busca temática e completa com páginas aleatórias.

    A busca temática oferece cobertura dirigida; o gerador aleatório evita que
    consultas muito sobrepostas deixem o corpus abaixo do tamanho solicitado.
    Cada artigo mantém título, URL canônica, idioma, licença e hash SHA-256.
    """
    documentos = documentos if documentos is not None else []
    pasta = saida_raw / f"wikipedia_{idioma}"
    pasta.mkdir(parents=True, exist_ok=True)
    host = "pt.wikipedia.org" if idioma == "pt" else "en.wikipedia.org"
    vistos: set[int] = set()
    adicionados = 0

    def registrar_paginas(payload: dict) -> int:
        nonlocal adicionados
        novos = 0
        paginas = payload.get("query", {}).get("pages", {})
        for pagina in sorted(paginas.values(), key=lambda item: int(item.get("pageid", 0))):
            page_id = int(pagina.get("pageid", 0))
            titulo = str(pagina.get("title", "")).strip()
            texto = str(pagina.get("extract", "")).strip()[:50000]
            if not page_id or page_id in vistos or len(texto) < 1200:
                continue
            vistos.add(page_id)
            arquivo = pasta / f"{page_id}.txt"
            arquivo.write_text(texto, encoding="utf-8")
            url_artigo = f"https://{host}/wiki/{urllib.parse.quote(titulo.replace(' ', '_'))}"
            documentos.append(DocumentoFonte(
                fonte=f"wikipedia_{idioma}", titulo=titulo,
                arquivo=str(arquivo.relative_to(saida_raw.parent)),
                url=url_artigo,
                licenca="CC BY-SA 4.0 (Wikipedia; atribuição ao artigo e histórico)",
                caracteres=len(texto), sha256=sha256_arquivo(arquivo), idioma=idioma,
            ))
            adicionados += 1
            novos += 1
            if adicionados >= max_documentos:
                break
        return novos

    for consulta in WIKIPEDIA_BUSCAS.get(idioma, WIKIPEDIA_BUSCAS["en"]):
        params = urllib.parse.urlencode({
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": consulta, "gsrnamespace": 0,
            "gsrlimit": min(max_documentos - adicionados, 10),
            "prop": "extracts", "explaintext": 1, "exchars": 50000,
        })
        bruto = _ler_url_texto(f"https://{host}/w/api.php?{params}", tentativas=2, espera=1.0)
        if bruto:
            try:
                registrar_paginas(json.loads(bruto))
            except json.JSONDecodeError:
                print(f"  [wikipedia:{idioma}] JSON inválido para {consulta!r}")
        if adicionados >= max_documentos:
            break
        time.sleep(0.2)

    # Consultas de texto frequentemente retornam artigos sobrepostos.
    # Completar com generator=random para alcançar a amostra mínima pedida.
    tentativas = 0
    max_tentativas = max(6, ((max_documentos - adicionados + 9) // 10) * 4)
    while adicionados < max_documentos and tentativas < max_tentativas:
        tentativas += 1
        params = urllib.parse.urlencode({
            "action": "query", "format": "json", "generator": "random",
            "grnnamespace": 0, "grnlimit": min(max_documentos - adicionados, 10),
            "prop": "extracts", "explaintext": 1, "exchars": 50000,
        })
        bruto = _ler_url_texto(f"https://{host}/w/api.php?{params}", tentativas=2, espera=1.0)
        if bruto:
            try:
                registrar_paginas(json.loads(bruto))
            except json.JSONDecodeError:
                print(f"  [wikipedia:{idioma}] JSON aleatório inválido")
        time.sleep(0.2)

    print(f"  [wikipedia:{idioma}] {adicionados}/{max_documentos} artigos coletados")
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
    min_documentos_idioma: int = 40,
) -> Path:
    """Executa a coleta completa e grava o manifesto."""
    saida = Path(saida)
    raw = saida / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    documentos: List[DocumentoFonte] = []
    antes = len(documentos)
    documentos = coletar_gutenberg(raw, "pt", max_livros_pt, documentos)
    pt_existentes = sum(d.idioma == "pt" for d in documentos)
    if pt_existentes < min_documentos_idioma:
        faltam = min_documentos_idioma - pt_existentes
        print(f"  [complemento] PT tem {pt_existentes}/{min_documentos_idioma} documentos; buscando {faltam} artigos na Wikipedia.")
        documentos = coletar_wikipedia(raw, "pt", faltam, documentos)

    documentos = coletar_gutenberg(raw, "en", max_livros_en, documentos)
    en_existentes = sum(d.idioma == "en" for d in documentos)
    if en_existentes < min_documentos_idioma:
        faltam = min_documentos_idioma - en_existentes
        print(f"  [complemento] EN tem {en_existentes}/{min_documentos_idioma} documentos; buscando {faltam} artigos na Wikipedia.")
        documentos = coletar_wikipedia(raw, "en", faltam, documentos)

    for idioma in ("pt", "en"):
        quantidade = sum(d.idioma == idioma for d in documentos)
        if quantidade < min_documentos_idioma:
            print(f"  [aviso] corpus {idioma} abaixo do mínimo: {quantidade}/{min_documentos_idioma}; a auditoria bloqueará o treino.")

    if not documentos:
        raise RuntimeError(
            "Nenhum livro foi coletado. O catálogo do Project Gutenberg pode estar "
            "temporariamente indisponível; execute novamente após alguns minutos."
        )
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
    parser.add_argument("--min-documentos-idioma", type=int, default=40)
    parser.add_argument("--sem-wikisource", action="store_true")
    args = parser.parse_args()
    construir_corpus(
        saida=args.saida,
        max_livros_pt=args.max_livros_pt,
        max_livros_en=args.max_livros_en,
        max_wiki_mb=args.max_wiki_mb,
        usar_wikisource=not args.sem_wikisource,
        min_documentos_idioma=args.min_documentos_idioma,
    )


if __name__ == "__main__":
    main()
