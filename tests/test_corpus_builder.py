"""Testes do coletor textual e dos fallbacks de fonte."""

import importlib.util
import json
import sys
from pathlib import Path

# Carrega somente o módulo sob teste. Importar o pacote "datasets" executa
# datasets/__init__.py e puxa dependências multimodais (NumPy/PyTorch) que
# não fazem parte dos requisitos deste coletor HTTP baseado na biblioteca padrão.
_SPEC = importlib.util.spec_from_file_location(
    "_apolo_corpus_builder_under_test",
    Path(__file__).resolve().parents[1] / "datasets" / "corpus_builder.py",
)
assert _SPEC is not None and _SPEC.loader is not None
cb = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cb
_SPEC.loader.exec_module(cb)


def test_coletar_wikipedia_registra_licenca_url_e_hash(tmp_path, monkeypatch):
    texto = "Conteúdo de referência para o corpus textual. " * 80
    resposta = {"query": {"pages": {"123": {
        "pageid": 123, "title": "Literatura brasileira", "extract": texto
    }}}}
    monkeypatch.setattr(cb, "_ler_url_texto", lambda *args, **kwargs: json.dumps(resposta))
    monkeypatch.setattr(cb.time, "sleep", lambda *_: None)
    raw = tmp_path / "datasets_storage" / "raw"
    documentos = cb.coletar_wikipedia(raw, "pt", 1, [])
    assert len(documentos) == 1
    documento = documentos[0]
    assert documento.fonte == "wikipedia_pt"
    assert documento.idioma == "pt"
    assert documento.licenca.startswith("CC BY-SA 4.0")
    assert documento.url == "https://pt.wikipedia.org/wiki/Literatura_brasileira"
    assert documento.sha256
    assert (tmp_path / "datasets_storage" / documento.arquivo).read_text() == texto.strip()


def test_coletor_wikipedia_usa_filtro_de_tamanho_por_idioma(tmp_path, monkeypatch):
    from urllib.parse import parse_qs, urlparse

    texto = "Artigo enciclopédico completo para o corpus. " * 30

    for idioma, tamanho_esperado in (("pt", "1000"), ("en", "5000")):
        chamadas = []

        def resposta(url, *args, **kwargs):
            params = parse_qs(urlparse(url).query)
            chamadas.append(params)
            if params.get("generator") == ["search"]:
                return json.dumps({"query": {"pages": {}}})
            if params.get("list") == ["allpages"]:
                return json.dumps({"query": {"allpages": [
                    {"pageid": 456, "title": "Substantial article"}
                ]}})
            if "pageids" in params:
                return json.dumps({"query": {"pages": {"456": {
                    "pageid": 456, "title": "Substantial article", "extract": texto,
                }}}})
            raise AssertionError(f"Consulta inesperada: {url}")

        monkeypatch.setattr(cb, "_ler_url_texto", resposta)
        monkeypatch.setattr(cb.time, "sleep", lambda *_: None)
        docs = cb.coletar_wikipedia(tmp_path / idioma, idioma, 1, [])
        assert len(docs) == 1
        assert docs[0].idioma == idioma
        catalogo = next(p for p in chamadas if p.get("list") == ["allpages"])
        assert catalogo["apminsize"] == [tamanho_esperado]
        assert catalogo["aplimit"] == ["500"]
        assert catalogo["maxlag"] == ["5"]


def test_construir_corpus_usa_wikipedia_quando_gutenberg_nao_tem_dados(tmp_path, monkeypatch):
    def sem_gutenberg(_raw, _idioma, _limite, documentos):
        return documentos

    def wikipedia(_raw, idioma, limite, documentos):
        documentos.append(cb.DocumentoFonte(
            fonte=f"wikipedia_{idioma}",
            titulo=f"Artigo {idioma}",
            arquivo=f"raw/wikipedia_{idioma}/artigo.txt",
            url=f"https://{idioma}.wikipedia.org/wiki/Artigo",
            licenca="CC BY-SA 4.0",
            caracteres=2500,
            sha256="a" * 64,
            idioma=idioma,
        ))
        return documentos

    monkeypatch.setattr(cb, "coletar_gutenberg", sem_gutenberg)
    monkeypatch.setattr(cb, "coletar_wikipedia", wikipedia)
    monkeypatch.setattr(cb, "coletar_wikisource", lambda raw, max_mb, docs: docs)
    manifesto = cb.construir_corpus(tmp_path / "saida", max_livros_pt=1, max_livros_en=1)
    dados = json.loads(manifesto.read_text(encoding="utf-8"))
    assert [item["fonte"] for item in dados] == ["wikipedia_pt", "wikipedia_en"]
    assert all(item["licenca"] == "CC BY-SA 4.0" for item in dados)


def test_construir_corpus_complementa_gutenberg_abaixo_do_minimo(tmp_path, monkeypatch):
    def gutenberg(raw, idioma, limite, documentos):
        documentos.append(cb.DocumentoFonte(
            fonte=f"gutenberg_{idioma}", titulo=f"Livro {idioma}",
            arquivo=f"raw/gutenberg_{idioma}/livro.txt",
            url="https://example.org/livro", licenca="Domínio público",
            caracteres=3000, sha256="b"*64, idioma=idioma))
        return documentos

    def wikipedia(raw, idioma, limite, documentos):
        for n in range(limite):
            documentos.append(cb.DocumentoFonte(
                fonte=f"wikipedia_{idioma}", titulo=f"Artigo {idioma} {n}",
                arquivo=f"raw/wikipedia_{idioma}/{n}.txt",
                url=f"https://{idioma}.wikipedia.org/wiki/{n}",
                licenca="CC BY-SA 4.0", caracteres=2500,
                sha256="c"*64, idioma=idioma))
        return documentos

    monkeypatch.setattr(cb, "coletar_gutenberg", gutenberg)
    monkeypatch.setattr(cb, "coletar_wikipedia", wikipedia)
    monkeypatch.setattr(cb, "coletar_wikisource", lambda raw, max_mb, docs: docs)
    manifesto = cb.construir_corpus(tmp_path / "saida", max_livros_pt=1,
        max_livros_en=1, usar_wikisource=False, min_documentos_idioma=3)
    dados = json.loads(manifesto.read_text(encoding="utf-8"))
    assert sum(x["idioma"] == "pt" for x in dados) == 3
    assert sum(x["idioma"] == "en" for x in dados) == 3

@__import__('pytest').mark.skip(reason='substituído pelo coletor paginado allpages')
def test_coletar_wikipedia_completa_com_paginas_aleatorias(tmp_path, monkeypatch):
    from urllib.parse import urlparse, parse_qs

    def resposta(url, *args, **kwargs):
        params = parse_qs(urlparse(url).query)
        if params.get("generator") == ["search"]:
            return json.dumps({"query": {"pages": {}}})
        if params.get("list") == ["random"]:
            return json.dumps({"query": {"random": [
                {"id": str(2000 + n), "title": f"Random {n}"}
                for n in range(20)
            ]}})
        if "pageids" in params:
            ids = params["pageids"][0].split("|")
            paginas = {
                page_id: {
                    "pageid": int(page_id),
                    "title": f"Random {page_id}",
                    "extract": "Texto enciclopédico válido para treinamento. " * 15,
                }
                for page_id in ids
            }
            return json.dumps({"query": {"pages": paginas}})
        raise AssertionError(f"Consulta inesperada: {url}")

    monkeypatch.setattr(cb, "_ler_url_texto", resposta)
    monkeypatch.setattr(cb.time, "sleep", lambda *_: None)
    docs = cb.coletar_wikipedia(tmp_path / "raw", "pt", 12, [])
    assert sum(d.idioma == "pt" for d in docs) == 12
    assert all(d.licenca.startswith("CC BY-SA 4.0") for d in docs)


def test_ler_json_api_repete_erro_maxlag_em_http_200(monkeypatch):
    respostas = iter([
        json.dumps({"error": {"code": "maxlag", "info": "Waiting for replica"}}),
        json.dumps({"query": {"random": [{"id": 42}]}}),
    ])
    monkeypatch.setattr(cb, "_ler_url_texto", lambda *args, **kwargs: next(respostas))
    monkeypatch.setattr(cb.time, "sleep", lambda *_: None)
    payload = cb._ler_json_api("https://pt.wikipedia.org/w/api.php", "pt", "teste", tentativas=2)
    assert payload["query"]["random"][0]["id"] == 42



def test_limite_lotes_wikipedia_cobre_corpus_maior_sem_loop_ilimitado():
    assert cb._limite_lotes_wikipedia(300) == 6
    assert cb._limite_lotes_wikipedia(50) == 1
    assert cb._limite_lotes_wikipedia(0) == 0
    assert cb._limite_lotes_wikipedia(10000) == 200
