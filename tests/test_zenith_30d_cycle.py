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
    preflight.write_text(json.dumps({"status": "passed", "checks": {"pytest": True, "cli": True}}))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "passed": True, "manifest_sha256": digest,
        "documents": {"total": 30000, "by_language": {"pt": 15000, "en": 15000}}
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
