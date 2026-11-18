"""Testes do coletor textual e dos fallbacks de fonte."""

import json
from datasets import corpus_builder as cb


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
