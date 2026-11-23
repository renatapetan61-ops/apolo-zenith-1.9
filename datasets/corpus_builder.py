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
                atraso = min(max(float(retry_after), 1.0), 120.0) if retry_after else min(espera * (2 ** tentativa), 60.0)
            except (TypeError, ValueError):
                atraso = min(espera * (2 ** tentativa), 60.0)
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


def _ler_json_api(url: str, idioma: str, contexto: str, tentativas: int = 8) -> Optional[dict]:
    """Consulta a API MediaWiki e repete erros JSON transitórios em HTTP 200."""
    transitorios = {"maxlag", "ratelimited", "readonly", "internal_api_error_DBConnectionError"}
    for tentativa in range(tentativas):
        bruto = _ler_url_texto(url, tentativas=3, espera=2.0)
        if bruto:
            try:
                payload = json.loads(bruto)
            except json.JSONDecodeError:
                print(f"  [wikipedia:{idioma}] JSON inválido ({contexto})")
                payload = None
            if isinstance(payload, dict):
                erro = payload.get("error") or {}
                codigo = str(erro.get("code", ""))
                if not erro:
                    return payload
                if codigo not in transitorios:
                    print(f"  [wikipedia:{idioma}] erro da API ({contexto}): {codigo or erro}")
                    return payload
                print(f"  [wikipedia:{idioma}] API {codigo} ({contexto}); tentativa {tentativa + 1}/{tentativas}")
        if tentativa < tentativas - 1:
            time.sleep(min(2 ** tentativa, 60))
    print(f"  [wikipedia:{idioma}] consulta esgotou retentativas ({contexto})")
    return None


# --------------------------------------------------------------------------- #
# Wikipedia — fallback independente do Project Gutenberg
# --------------------------------------------------------------------------- #
WIKIPEDIA_BUSCAS = {
    "pt": ["literatura brasileira", "história do Brasil", "ciência no Brasil", "geografia do Brasil", "língua portuguesa", "biologia", "física", "matemática", "meio ambiente", "tecnologia", "cultura brasileira", "saúde pública"],
    "en": ["world history", "science", "literature", "geography", "biology", "physics", "mathematics", "environment", "technology", "public health", "computer science", "language"],
}


def _limite_lotes_wikipedia(documentos_restantes: int) -> int:
    """Orçamento defensivo para lotes de extração em páginas de 50 IDs."""
    if documentos_restantes <= 0:
        return 0
    return max(1, (documentos_restantes + 49) // 50)


def coletar_wikipedia(
    saida_raw: Path, idioma: str, max_documentos: int,
    documentos: Optional[List[DocumentoFonte]] = None,
) -> List[DocumentoFonte]:
    """Coleta artigos CC BY-SA 4.0 com paginação MediaWiki allpages.

    A coleta anterior usava list=random em ciclos curtos e teto de 1.200 lotes,
    limitando o corpus a cerca de mil artigos por idioma. Esta versão obtém até
    500 IDs por página e extrai textos em grupos de 50, seguindo o cursor da API.
    """
    documentos = documentos if documentos is not None else []
    pasta = saida_raw / f"wikipedia_{idioma}"
    pasta.mkdir(parents=True, exist_ok=True)
    host = "pt.wikipedia.org" if idioma == "pt" else "en.wikipedia.org"
    # Reuse page IDs already collected by the thematic search.
    vistos: set[int] = {
        int(Path(d.arquivo).stem)
        for d in documentos
        if d.idioma == idioma and d.fonte == f"wikipedia_{idioma}"
        and Path(d.arquivo).stem.isdigit()
    }
    adicionados = sum(1 for d in documentos if d.idioma == idioma and d.fonte == f"wikipedia_{idioma}")

    def registrar_paginas(payload: dict) -> int:
        nonlocal adicionados
        novos = 0
        paginas = payload.get("query", {}).get("pages", {})
        iterable = paginas if isinstance(paginas, list) else paginas.values()
        for pagina in sorted(iterable, key=lambda item: int(item.get("pageid", item.get("id", 0)))):
            page_id = int(pagina.get("pageid", pagina.get("id", 0)))
            titulo = str(pagina.get("title", "")).strip()
            texto = str(pagina.get("extract", "")).strip()[:50000]
            if not page_id or page_id in vistos or len(texto) < 500:
                continue
            vistos.add(page_id)
            arquivo = pasta / f"{page_id}.txt"
            arquivo.write_text(texto, encoding="utf-8")
            titulo_url = urllib.parse.quote(titulo.replace(" ", "_"))
            url_artigo = f"https://{host}/wiki/{titulo_url}"
            documentos.append(DocumentoFonte(
                fonte=f"wikipedia_{idioma}", titulo=titulo,
                arquivo=str(arquivo.relative_to(saida_raw.parent)), url=url_artigo,
                licenca="CC BY-SA 4.0 (Wikipedia; atribuição ao artigo e histórico)",
                caracteres=len(texto), sha256=sha256_arquivo(arquivo), idioma=idioma,
            ))
            adicionados += 1
            novos += 1
            if adicionados >= max_documentos:
                break
        return novos

    # Busca temática inicial para favorecer conteúdo relevante.
    for consulta in WIKIPEDIA_BUSCAS.get(idioma, WIKIPEDIA_BUSCAS["en"]):
        if adicionados >= max_documentos:
            break
        restantes = max_documentos - adicionados
        params = urllib.parse.urlencode({
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": consulta, "gsrnamespace": 0, "gsrlimit": min(restantes, 50),
            "prop": "extracts", "explaintext": 1, "exlimit": 50, "maxlag": 5,
        })
        payload = _ler_json_api(f"https://{host}/w/api.php?{params}", idioma, f"busca {consulta!r}")
        if payload:
            registrar_paginas(payload)
        time.sleep(0.35)

    parametros_continuacao: dict = {}
    paginas_catalogo = 0
    falhas_catalogo_consecutivas = 0
    while adicionados < max_documentos:
        params = {
            "action": "query", "format": "json", "list": "allpages",
            "apnamespace": 0, "apfilterredir": "nonredirects",
            # Exclui stubs e páginas pequenas já no catálogo, antes de gastar
            # uma chamada de extração por lote. A auditoria continua exigindo
            # texto útil (>= 500 caracteres) e 15.000 documentos por idioma.
            "aplimit": 500, "apminsize": 5000, "maxlag": 5,
            **parametros_continuacao,
        }
        url = f"https://{host}/w/api.php?{urllib.parse.urlencode(params)}"
        catalogo = _ler_json_api(url, idioma, "catálogo allpages", tentativas=10)
        if not catalogo or catalogo.get("error"):
            falhas_catalogo_consecutivas += 1
            if falhas_catalogo_consecutivas >= 10:
                raise RuntimeError(f"Wikipedia {idioma}: API indisponível; coletados {adicionados}/{max_documentos}.")
            time.sleep(min(5 * falhas_catalogo_consecutivas, 60))
            continue
        falhas_catalogo_consecutivas = 0
        entradas = catalogo.get("query", {}).get("allpages", [])
        # Remove artigos obtidos pela busca temática antes de aplicar a cota restante.
        ids = [
            str(item.get("pageid")) for item in entradas
            if item.get("pageid") and int(item["pageid"]) not in vistos
        ]
        ids = ids[:max_documentos - adicionados]
        if ids:
            paginas_catalogo += 1
            for inicio in range(0, len(ids), 50):
                if adicionados >= max_documentos:
                    break
                lote_ids = ids[inicio:inicio + 50]
                params_texto = urllib.parse.urlencode({
                    "action": "query", "format": "json", "pageids": "|".join(lote_ids),
                    "prop": "extracts", "explaintext": 1, "exlimit": 50, "maxlag": 5,
                })
                payload = _ler_json_api(
                    f"https://{host}/w/api.php?{params_texto}", idioma, "extração allpages", tentativas=10
                )
                if not payload or payload.get("error"):
                    raise RuntimeError(
                        f"Wikipedia {idioma}: falha na extração allpages; "
                        f"coletados {adicionados}/{max_documentos}. Repita a execução."
                    )
                registrar_paginas(payload)
                time.sleep(0.35)
        # Atingir a cota encerra a coleta; não é necessário exigir cursor
        # de continuação quando já temos todos os documentos solicitados.
        if adicionados >= max_documentos:
            break
        # MediaWiki exige todos os campos de continuação retornados pela resposta.
        continuacao = catalogo.get("continue", {})
        if not isinstance(continuacao, dict) or not continuacao:
            raise RuntimeError(f"Wikipedia {idioma}: catálogo esgotado com {adicionados}/{max_documentos} documentos.")
        parametros_continuacao = {str(k): str(v) for k, v in continuacao.items()}
        if paginas_catalogo and paginas_catalogo % 10 == 0:
            print(f"  [wikipedia:{idioma}] progresso: {adicionados}/{max_documentos}; páginas de catálogo {paginas_catalogo}")
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
