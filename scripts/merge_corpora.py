#!/usr/bin/env python3
"""Merge and verify the existing APOLO ZENITH PT-BR and EN corpus artifacts."""
import argparse, hashlib, json, shutil
from pathlib import Path, PurePosixPath

PER_LANGUAGE = 15000
TOTAL = 30000

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def verify(root, lang):
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"Manifesto ausente: {manifest}")
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != PER_LANGUAGE:
        raise SystemExit(f"{lang}: esperado {PER_LANGUAGE} registros")
    hashes, paths = set(), set()
    for i, rec in enumerate(records):
        if rec.get("idioma") != lang:
            raise SystemExit(f"{lang}: idioma incorreto no registro {i}")
        rel = PurePosixPath(str(rec.get("arquivo", "")))
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise SystemExit(f"{lang}: caminho inseguro: {rel}")
        name = rel.as_posix()
        if name in paths:
            raise SystemExit(f"{lang}: caminho duplicado: {name}")
        paths.add(name)
        file = root.joinpath(*rel.parts)
        if not file.is_file():
            raise SystemExit(f"{lang}: arquivo ausente: {name}")
        actual = digest(file)
        if actual != rec.get("sha256"):
            raise SystemExit(f"{lang}: SHA-256 divergente: {name}")
        if actual in hashes:
            raise SystemExit(f"{lang}: hash duplicado: {actual}")
        hashes.add(actual)
        if not rec.get("titulo") or not rec.get("url") or "CC BY-SA 4.0" not in rec.get("licenca", ""):
            raise SystemExit(f"{lang}: metadados/licença inválidos: {name}")
    return records

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pt", type=Path, required=True)
    p.add_argument("--en", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pt-run-id", required=True)
    p.add_argument("--en-run-id", required=True)
    p.add_argument("--source-commit", required=True)
    a = p.parse_args()
    pt, en = verify(a.pt, "pt"), verify(a.en, "en")
    hp, he = {x["sha256"] for x in pt}, {x["sha256"] for x in en}
    if hp & he:
        raise SystemExit(f"Hashes duplicados entre idiomas: {len(hp & he)}")
    if len(pt) + len(en) != TOTAL:
        raise SystemExit("Total combinado diferente de 30.000")
    if a.out.exists():
        shutil.rmtree(a.out)
    (a.out / "raw").mkdir(parents=True)
    for root, records in ((a.pt, pt), (a.en, en)):
        for rec in records:
            rel = PurePosixPath(rec["arquivo"])
            src, dst = root.joinpath(*rel.parts), a.out.joinpath(*rel.parts)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    combined = pt + en
    (a.out / "manifest.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "project": "APOLO ZENITH 1.9", "documents_total": len(combined),
        "documents_pt": len(pt), "documents_en": len(en),
        "pt_source_run_id": a.pt_run_id, "en_source_run_id": a.en_run_id,
        "merged_on_commit": a.source_commit,
        "integrity": "all source files SHA-256 verified; no duplicate hashes",
        "license_policy": "CC BY-SA 4.0 attribution metadata preserved per document"
    }
    (a.out / "corpus_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
