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
    assert (tmp_path / "datasets_storage" / documento.arquivo).read_text() == texto


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
