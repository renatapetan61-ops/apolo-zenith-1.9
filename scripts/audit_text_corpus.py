"""Audita tamanho, equilíbrio e proveniência do corpus antes do treinamento."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def auditar(manifesto: Path, processed: Path, min_docs: int=40,
            min_train_tokens: int=20000, min_eval_tokens: int=1000,
            expected_total_documents: int=0) -> dict:
    docs=json.loads(manifesto.read_text(encoding="utf-8"))
    por_idioma=Counter(str(d.get("idioma","")).lower() for d in docs)
    por_fonte=Counter(str(d.get("fonte","")) for d in docs)
    licencas=Counter(str(d.get("licenca","")) for d in docs)
    integridade=[]
    for d in docs:
        arquivo=d.get("arquivo")
        raw_root=manifesto.parent
        path=raw_root / arquivo if arquivo else None
        if not path or not path.is_file():
            integridade.append({"arquivo":arquivo,"ok":False,"motivo":"arquivo ausente"})
            continue
        digest=sha256(path)
        integridade.append({"arquivo":arquivo,"ok":digest==d.get("sha256"),
                            "sha256_calculado":digest})
    tokens={}
    for split in ("train","val","test"):
        path=processed/f"{split}.bin"
        if not path.is_file():
            tokens[split]=0
        else:
            size=path.stat().st_size
            if size%2: raise ValueError(f"{path} não possui tamanho válido para uint16")
            tokens[split]=size//2
    hashes=[str(d.get("sha256","")) for d in docs]
    duplicados=len(hashes)-len(set(hashes))
    licencas_aprovadas=all(
      ("CC BY-SA 4.0" in str(d.get("licenca","")))
      or ("Domínio público" in str(d.get("licenca","")))
      for d in docs
    )
    gates={
      "minimum_pt_documents":por_idioma.get("pt",0)>=min_docs,
      "minimum_en_documents":por_idioma.get("en",0)>=min_docs,
      "training_tokens":tokens["train"]>=min_train_tokens,
      "validation_tokens":tokens["val"]>=min_eval_tokens,
      "test_tokens":tokens["test"]>=min_eval_tokens,
      "manifest_files_and_hashes":bool(integridade) and all(x["ok"] for x in integridade),
      "unique_document_hashes":duplicados==0,
      "approved_licenses":licencas_aprovadas,
      "expected_total_documents":expected_total_documents<=0 or len(docs)==expected_total_documents,
      "declared_licenses":all(bool(x.strip()) for x in licencas),
      "both_languages_present":por_idioma.get("pt",0)>0 and por_idioma.get("en",0)>0
    }
    return {"status":"success","passed":all(gates.values()),"gates":gates,
      "thresholds":{"minimum_documents_per_language":min_docs,
        "expected_total_documents":expected_total_documents,
        "minimum_train_tokens":min_train_tokens,"minimum_val_test_tokens_each":min_eval_tokens},
      "documents":{"total":len(docs),"by_language":dict(por_idioma),"by_source":dict(por_fonte),
        "licenses":dict(licencas),"integrity_checked":len(integridade),
        "duplicate_hashes":duplicados,
        "integrity_failures":sum(not x["ok"] for x in integridade)},
      "tokens_by_split":tokens,"manifest_sha256":sha256(manifesto),
      "note":"O treinamento é bloqueado se qualquer critério falhar; o relatório preserva a proveniência."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifesto",default="datasets_storage/manifest.json")
    p.add_argument("--processado",default="datasets_storage/processed")
    p.add_argument("--min-documentos-idioma",type=int,default=40)
    p.add_argument("--min-tokens-treino",type=int,default=20000)
    p.add_argument("--min-tokens-avaliacao",type=int,default=1000)
    p.add_argument("--documentos-esperados",type=int,default=0,
                   help="exige a quantidade total exata de documentos quando maior que zero")
    p.add_argument("--saida",default="reports/corpus-audit.json")
    a=p.parse_args()
    report=auditar(Path(a.manifesto),Path(a.processado),a.min_documentos_idioma,
                   a.min_tokens_treino,a.min_tokens_avaliacao,a.documentos_esperados)
    out=Path(a.saida); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit("Corpus insuficiente ou integridade inválida; treinamento bloqueado.")

if __name__=="__main__": main()
