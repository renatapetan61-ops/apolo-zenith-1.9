import hashlib
import json
from scripts.audit_text_corpus import auditar

def test_auditoria_aprova_corpus_bilingue_com_integridade(tmp_path):
    root=tmp_path/"datasets_storage"; root.mkdir()
    raw=root/"raw"; raw.mkdir()
    docs=[]
    for lang in ("pt","en"):
        for i in range(3):
            rel=f"raw/{lang}_{i}.txt"
            path=root/rel
            path.write_text((f"Texto {lang} documento {i}. ")*100,encoding="utf-8")
            docs.append({"idioma":lang,"fonte":f"wikipedia_{lang}","arquivo":rel,
                "licenca":"CC BY-SA 4.0","sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    manifesto=root/"manifest.json"
    manifesto.write_text(json.dumps(docs),encoding="utf-8")
    processed=root/"processed"; processed.mkdir()
    for split,count in (("train",25000),("val",1500),("test",1500)):
        (processed/f"{split}.bin").write_bytes(b"\x00\x00"*count)
    report=auditar(manifesto,processed,min_docs=3,min_train_tokens=20000,min_eval_tokens=1000)
    assert report["passed"] is True
    assert report["documents"]["integrity_failures"] == 0

def test_auditoria_reprova_corpus_sem_cobertura_bilingue(tmp_path):
    root=tmp_path/"datasets_storage"; root.mkdir()
    (root/"raw").mkdir()
    docs=[]
    for i in range(2):
        rel=f"raw/pt_{i}.txt"; path=root/rel; path.write_text("texto",encoding="utf-8")
        docs.append({"idioma":"pt","fonte":"wikipedia_pt","arquivo":rel,
          "licenca":"CC BY-SA 4.0","sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    manifesto=root/"manifest.json"; manifesto.write_text(json.dumps(docs),encoding="utf-8")
    processed=root/"processed"; processed.mkdir()
    for split,count in (("train",10),("val",10),("test",10)):
        (processed/f"{split}.bin").write_bytes(b"\x00\x00"*count)
    report=auditar(manifesto,processed,min_docs=2,min_train_tokens=10,min_eval_tokens=5)
    assert report["passed"] is False
    assert report["gates"]["minimum_en_documents"] is False

def test_auditoria_bloqueia_hash_de_documento_alterado(tmp_path):
    root=tmp_path/"datasets_storage"; (root/"raw").mkdir(parents=True)
    rel="raw/artigo.txt"; path=root/rel; path.write_text("conteúdo original",encoding="utf-8")
    docs=[{"idioma":"pt","fonte":"wikipedia_pt","arquivo":rel,"licenca":"CC BY-SA 4.0","sha256":"0"*64},
          {"idioma":"en","fonte":"wikipedia_en","arquivo":rel,"licenca":"CC BY-SA 4.0","sha256":"0"*64}]
    manifesto=root/"manifest.json"; manifesto.write_text(json.dumps(docs),encoding="utf-8")
    processed=root/"processed"; processed.mkdir()
    for split,count in (("train",20),("val",10),("test",10)):
        (processed/f"{split}.bin").write_bytes(b"\\x00\\x00"*count)
    report=auditar(manifesto,processed,min_docs=1,min_train_tokens=10,min_eval_tokens=5)
    assert report["passed"] is False
    assert report["gates"]["manifest_files_and_hashes"] is False

def test_auditoria_exige_total_exato_quando_configurado(tmp_path):
    import hashlib, json
    from scripts.audit_text_corpus import auditar
    root=tmp_path/"data"; (root/"raw").mkdir(parents=True)
    docs=[]
    for lang in ("pt","en"):
        rel=f"raw/{lang}.txt"; path=root/rel; path.write_text("conteúdo único "+lang,encoding="utf-8")
        docs.append({"idioma":lang,"fonte":"wikipedia_"+lang,"arquivo":rel,
          "licenca":"CC BY-SA 4.0","sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    manifesto=root/"manifest.json"; manifesto.write_text(json.dumps(docs),encoding="utf-8")
    processed=root/"processed"; processed.mkdir()
    for split,count in (("train",20),("val",10),("test",10)):
        (processed/f"{split}.bin").write_bytes(b"\\x00\\x00"*count)
    report=auditar(manifesto,processed,min_docs=1,min_train_tokens=10,min_eval_tokens=5,
                   expected_total_documents=30000)
    assert report["passed"] is False
    assert report["gates"]["expected_total_documents"] is False


def test_auditoria_exige_documentos_processados_minimos(tmp_path):
    import hashlib, json
    from scripts.audit_text_corpus import auditar
    root=tmp_path/"data"; (root/"raw").mkdir(parents=True)
    docs=[]
    for lang in ("pt","en"):
        rel=f"raw/{lang}.txt"; path=root/rel; path.write_text("conteúdo "+lang,encoding="utf-8")
        docs.append({"idioma":lang,"fonte":"wikipedia_"+lang,"arquivo":rel,
          "licenca":"CC BY-SA 4.0","sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    manifesto=root/"manifest.json"; manifesto.write_text(json.dumps(docs),encoding="utf-8")
    processed=root/"processed"; processed.mkdir()
    for split,count in (("train",20),("val",10),("test",10)):
        (processed/f"{split}.bin").write_bytes(b"\\x00\\x00"*count)
    (processed/"resumo.json").write_text(json.dumps({
      "documentos":{"lidos":2,"aprovados":1,"rejeitados":1},
      "duplicadas_removidas":0
    }))
    report=auditar(manifesto,processed,min_docs=1,min_train_tokens=10,min_eval_tokens=5,
                   min_processed_documents=2)
    assert report["passed"] is False
    assert report["gates"]["minimum_processed_documents"] is False
