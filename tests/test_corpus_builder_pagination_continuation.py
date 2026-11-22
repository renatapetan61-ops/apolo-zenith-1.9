import json
from urllib.parse import parse_qs, urlparse

from datasets import corpus_builder as cb


def test_wikipedia_collector_follows_allpages_continuation_cursor(tmp_path, monkeypatch):
    cursors = []
    extracted_ids = []

    def response(url, *args, **kwargs):
        params = parse_qs(urlparse(url).query)

        if params.get("generator") == ["search"]:
            return json.dumps({"query": {"pages": {}}})

        if params.get("list") == ["allpages"]:
            cursor = params.get("apcontinue", [None])[0]
            generic_continue = params.get("continue", [None])[0]
            cursors.append((cursor, generic_continue))
            if cursor is None:
                assert generic_continue is None
                start = 1
                continuation = "cursor-page-two"
            elif cursor == "cursor-page-two":
                assert generic_continue == "-||"
                start = 11
                continuation = "cursor-page-three"
            else:
                raise AssertionError("Unexpected continuation cursor: " + str(cursor))

            return json.dumps({
                "query": {
                    "allpages": [
                        {"pageid": start + i, "title": "Article " + str(start + i)}
                        for i in range(10)
                    ]
                },
                "continue": {
                    "apcontinue": continuation,
                    "continue": "-||",
                },
            })

        if "pageids" in params:
            ids = params["pageids"][0].split("|")
            extracted_ids.extend(ids)
            return json.dumps({
                "query": {
                    "pages": {
                        page_id: {
                            "pageid": int(page_id),
                            "title": "Article " + page_id,
                            "extract": (
                                "Conteudo enciclopedico unico do artigo " + page_id + ". "
                            ) * 35,
                        }
                        for page_id in ids
                    }
                }
            })

        raise AssertionError("Unexpected MediaWiki request: " + url)

    monkeypatch.setattr(cb, "_ler_url_texto", response)
    monkeypatch.setattr(cb.time, "sleep", lambda *_: None)

    docs = cb.coletar_wikipedia(tmp_path / "raw", "pt", 12, [])

    assert len(docs) == 12
    assert cursors == [(None, None), ("cursor-page-two", "-||")]
    assert extracted_ids == [str(i) for i in range(1, 13)]
    assert len({doc.sha256 for doc in docs}) == 12
