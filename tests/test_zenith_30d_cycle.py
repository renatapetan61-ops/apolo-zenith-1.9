import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.zenith_30d_cycle import iniciar, registrar_sessao, atualizar_relogio, mudar_status


def _prerequisitos(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"status": "passed", "commit": "test-commit", "checks": {"pytest": True, "cli": True}}))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "passed": True, "manifest_sha256": digest, "release_tag": "apolo-zenith-corpus-30000-test", "source_commit": "test-commit",
        "gates": {name: True for name in (
            "all_manifest_documents_processed", "minimum_processed_documents",
            "minimum_pt_documents", "minimum_en_documents", "training_tokens",
            "validation_tokens", "test_tokens", "manifest_files_and_hashes",
            "unique_document_hashes", "approved_licenses", "expected_total_documents",
            "declared_licenses", "both_languages_present"
        )},
        "thresholds": {"minimum_train_tokens": 1000000, "minimum_val_test_tokens_each": 10000},
        "tokens_by_split": {"train": 1000000, "val": 10000, "test": 10000},
        "documents": {
            "total": 30000, "by_language": {"pt": 15000, "en": 15000},
            "processed_final_documents": 27000, "integrity_checked": 30000,
            "integrity_failures": 0, "duplicate_hashes": 0,
            "licenses": {"CC BY-SA 4.0": 15000, "Domínio público": 15000}
        }
    }))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name, content in (("model.pt", b"model"), ("optimizer.pt", b"optimizer"),
                          ("scheduler.pt", b"scheduler")):
        (checkpoint / name).write_bytes(content)
    (checkpoint / "meta.json").write_text(json.dumps({"step": 12000}))
    return preflight, audit, manifest, checkpoint


def test_ciclo_so_inicia_com_preflight_corpus_e_checkpoint_validos(tmp_path):
    preflight, audit, manifest, checkpoint = _prerequisitos(tmp_path)
    state = tmp_path / "state.json"
    now = datetime(2026, 11, 18, tzinfo=timezone.utc)
    result = iniciar(state, preflight, audit, manifest, checkpoint, now)
    assert result["status"] == "running"
    assert result["day_index"] == 1
    assert result["last_step"] == 12000
    assert result["ends_at"] == (now + timedelta(days=30)).isoformat()


def test_ciclo_nao_inicia_com_auditoria_reprovada(tmp_path):
    preflight, audit, manifest, checkpoint = _prerequisitos(tmp_path)
    payload = json.loads(audit.read_text())
    payload["passed"] = False
    audit.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="auditoria"):
        iniciar(tmp_path / "state.json", preflight, audit, manifest, checkpoint)


def test_sessao_exige_checkpoint_com_passos_maiores(tmp_path):
    preflight, audit, manifest, checkpoint = _prerequisitos(tmp_path)
    state = tmp_path / "state.json"
    start = datetime(2026, 11, 18, tzinfo=timezone.utc)
    iniciar(state, preflight, audit, manifest, checkpoint, start)
    meta = json.loads((checkpoint / "meta.json").read_text())
    meta["step"] = 13000
    (checkpoint / "meta.json").write_text(json.dumps(meta))
    result = registrar_sessao(state, checkpoint, "123", 1000, now=start + timedelta(hours=6))
    assert result["last_step"] == 13000
    assert result["completed_sessions"] == 1


def test_ciclo_so_completa_depois_dos_30_dias(tmp_path):
    preflight, audit, manifest, checkpoint = _prerequisitos(tmp_path)
    state = tmp_path / "state.json"
    start = datetime(2026, 11, 18, tzinfo=timezone.utc)
    iniciar(state, preflight, audit, manifest, checkpoint, start)
    before = atualizar_relogio(json.loads(state.read_text()), start + timedelta(days=29, hours=23))
    assert before["status"] == "running"
    after = atualizar_relogio(before, start + timedelta(days=30))
    assert after["status"] == "completed"
    assert after["day_index"] == 30


def test_pausar_e_retomar_preserva_deadline(tmp_path):
    preflight, audit, manifest, checkpoint = _prerequisitos(tmp_path)
    state = tmp_path / "state.json"
    start = datetime(2026, 11, 18, tzinfo=timezone.utc)
    iniciar(state, preflight, audit, manifest, checkpoint, start)
    paused = mudar_status(state, "pause", start + timedelta(hours=1))
    assert paused["status"] == "paused"
    resumed = mudar_status(state, "resume", start + timedelta(hours=2))
    assert resumed["status"] == "running"
    assert resumed["ends_at"] == (start + timedelta(days=30)).isoformat()


def test_sessao_parcial_nao_avanca_estado(tmp_path):
    preflight, audit, manifest, checkpoint = _prerequisitos(tmp_path)
    state = tmp_path / "state.json"
    start = datetime(2026, 11, 18, tzinfo=timezone.utc)
    iniciar(state, preflight, audit, manifest, checkpoint, start)
    meta = json.loads((checkpoint / "meta.json").read_text())
    meta["step"] = 12500
    (checkpoint / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="sessão incompleta"):
        registrar_sessao(state, checkpoint, "124", 1000, now=start + timedelta(hours=6))
    saved = json.loads(state.read_text())
    assert saved["last_step"] == 12000
    assert saved["completed_sessions"] == 0


def test_ciclo_pausado_tambem_expira_ao_completar_30_dias(tmp_path):
    preflight, audit, manifest, checkpoint = _prerequisitos(tmp_path)
    state = tmp_path / "state.json"
    start = datetime(2026, 11, 18, tzinfo=timezone.utc)
    iniciar(state, preflight, audit, manifest, checkpoint, start)
    mudar_status(state, "pause", start + timedelta(hours=1))
    result = atualizar_relogio(json.loads(state.read_text()), start + timedelta(days=30))
    assert result["status"] == "completed"


def test_ciclo_registra_lacuna_de_monitoramento_acima_de_12_horas(tmp_path):
    preflight, audit, manifest, checkpoint = _prerequisitos(tmp_path)
    state = tmp_path / "state.json"
    start = datetime(2026, 11, 18, tzinfo=timezone.utc)
    iniciar(state, preflight, audit, manifest, checkpoint, start)
    result = atualizar_relogio(json.loads(state.read_text()), start + timedelta(hours=18))
    gaps = [event for event in result["events"] if event["type"] == "monitoring_gap"]
    assert len(gaps) == 1
    assert gaps[0]["gap_hours"] == 18.0


def test_ciclo_recusa_preflight_ou_corpus_de_outro_commit(tmp_path):
    preflight, audit, manifest, checkpoint = _prerequisitos(tmp_path)
    with pytest.raises(ValueError, match="outro commit"):
        iniciar(tmp_path / "state.json", preflight, audit, manifest, checkpoint,
                expected_commit="different-commit")
