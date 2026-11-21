import json
from urllib.parse import parse_qs, urlparse
from datasets import corpus_builder as cb

def test_wikipedia_collector_uses_allpages_and_extracts_in_batches(tmp_path, monkeypatch):
    def response(url, *args, **kwargs):
        params = parse_qs(urlparse(url).query)
        if params.get("generator") == ["search"]:
            return json.dumps({"query": {"pages": {}}})
        if params.get("list") == ["allpages"]:
            return json.dumps({"query": {"allpages": [
                {"pageid": 100 + i, "title": "Article " + str(i)} for i in range(20)
            ]}, "continue": {"apcontinue": "next", "continue": "-||"}})
        if "pageids" in params:
            ids = params["pageids"][0].split("|")
            return json.dumps({"query": {"pages": {
                page_id: {"pageid": int(page_id), "title": "Article " + page_id,
                          "extract": "Texto enciclopédico válido. " * 30}
                for page_id in ids
            }}})
        raise AssertionError("Unexpected MediaWiki request: " + url)

    monkeypatch.setattr(cb, "_ler_url_texto", response)
    monkeypatch.setattr(cb.time, "sleep", lambda *_: None)
    docs = cb.coletar_wikipedia(tmp_path / "raw", "pt", 12, [])
    assert len(docs) == 12
    assert all(doc.licenca.startswith("CC BY-SA 4.0") for doc in docs)
    assert len({doc.sha256 for doc in docs}) == 12
