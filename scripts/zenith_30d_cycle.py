"""Estado auditável do ciclo incremental de treinamento de 30 dias.

O relógio só começa após preflight e auditoria do corpus aprovados. Cada sessão
registra o checkpoint e os passos alcançados; uma sessão interrompida não
avança o estado sem checkpoint íntegro.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DURACAO_DIAS = 30
DOCUMENTOS_ESPERADOS = 30000
MIN_DOCUMENTOS_IDIOMA = 15000


def agora_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_data(valor: str) -> datetime:
    data = datetime.fromisoformat(valor.replace("Z", "+00:00"))
    if data.tzinfo is None:
        raise ValueError("timestamp precisa conter fuso horário")
    return data.astimezone(timezone.utc)


def salvar_json_atomico(caminho: Path, dados: dict[str, Any]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    temporario.write_text(json.dumps(dados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporario.replace(caminho)


def carregar_json(caminho: Path) -> dict[str, Any]:
    if not caminho.is_file():
        raise FileNotFoundError(f"arquivo não encontrado: {caminho}")
    return json.loads(caminho.read_text(encoding="utf-8"))


def sha256_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def sha256_diretorio(pasta: Path) -> str:
    digest = hashlib.sha256()
    arquivos = sorted(p for p in pasta.rglob("*") if p.is_file())
    if not arquivos:
        raise ValueError(f"diretório vazio: {pasta}")
    for arquivo in arquivos:
        digest.update(str(arquivo.relative_to(pasta)).encode("utf-8"))
        digest.update(b"\0")
        with arquivo.open("rb") as entrada:
            for bloco in iter(lambda: entrada.read(1024 * 1024), b""):
                digest.update(bloco)
    return digest.hexdigest()


def validar_checkpoint(pasta: Path) -> dict[str, Any]:
    obrigatorios = ("model.pt", "optimizer.pt", "scheduler.pt", "meta.json")
    ausentes = [nome for nome in obrigatorios if not (pasta / nome).is_file()]
    if ausentes:
        raise ValueError("checkpoint incompleto; ausentes: " + ", ".join(ausentes))
    meta = carregar_json(pasta / "meta.json")
    if int(meta.get("step", -1)) < 0:
        raise ValueError("checkpoint sem contador de passos válido")
    if (pasta / "model.pt").stat().st_size == 0:
        raise ValueError("pesos do checkpoint estão vazios")
    return meta


def validar_pre_requisitos(preflight_path: Path, audit_path: Path, manifest_path: Path) -> dict[str, Any]:
    preflight = carregar_json(preflight_path)
    audit = carregar_json(audit_path)
    if preflight.get("status") != "passed" or not preflight.get("checks"):
        raise ValueError("preflight não aprovado")
    if not all(value is True for value in preflight["checks"].values()):
        raise ValueError("há verificações de preflight não aprovadas")
    if audit.get("passed") is not True:
        raise ValueError("auditoria do corpus não aprovada")
    docs = audit.get("documents", {})
    por_idioma = docs.get("by_language", {})
    if int(docs.get("total", 0)) != DOCUMENTOS_ESPERADOS:
        raise ValueError("corpus não contém exatamente 30.000 documentos")
    if int(por_idioma.get("pt", 0)) < MIN_DOCUMENTOS_IDIOMA:
        raise ValueError("corpus não atingiu 15.000 documentos em português")
    if int(por_idioma.get("en", 0)) < MIN_DOCUMENTOS_IDIOMA:
        raise ValueError("corpus não atingiu 15.000 documentos em inglês")
    if not manifest_path.is_file():
        raise ValueError("manifesto de proveniência ausente")
    manifest_hash = sha256_arquivo(manifest_path)
    if audit.get("manifest_sha256") != manifest_hash:
        raise ValueError("hash do manifesto não corresponde à auditoria")
    return {
        "preflight_run_id": preflight.get("run_id"),
        "preflight_commit": preflight.get("commit"),
        "corpus_documents": int(docs["total"]),
        "documents_by_language": por_idioma,
        "manifest_sha256": manifest_hash,
        "corpus_audit_passed": True,
    }


def iniciar(state_path: Path, preflight_path: Path, audit_path: Path,
            manifest_path: Path, checkpoint_path: Path, now: datetime | None = None) -> dict[str, Any]:
    if state_path.exists():
        atual = carregar_json(state_path)
        if atual.get("status") in ("running", "paused"):
            raise ValueError("já existe um ciclo ativo ou pausado; não iniciar outro")
    requisitos = validar_pre_requisitos(preflight_path, audit_path, manifest_path)
    meta = validar_checkpoint(checkpoint_path)
    inicio = (now or agora_utc()).astimezone(timezone.utc)
    fim = inicio + timedelta(days=DURACAO_DIAS)
    estado = {
        "schema_version": 1,
        "status": "running",
        "duration_days": DURACAO_DIAS,
        "started_at": inicio.isoformat(),
        "ends_at": fim.isoformat(),
        "last_seen_at": inicio.isoformat(),
        "day_index": 1,
        "completed_sessions": 0,
        "last_step": int(meta["step"]),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_diretorio(checkpoint_path),
        "final_evaluation_done": False,
        "final_evaluation": None,
        "corpus": requisitos,
        "events": [{
            "type": "cycle_started",
            "at": inicio.isoformat(),
            "initial_step": int(meta["step"]),
            "checkpoint_sha256": sha256_diretorio(checkpoint_path),
        }],
    }
    salvar_json_atomico(state_path, estado)
    return estado


def atualizar_relogio(estado: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    instante = (now or agora_utc()).astimezone(timezone.utc)
    estado["last_seen_at"] = instante.isoformat()
    inicio = parse_data(estado["started_at"])
    fim = parse_data(estado["ends_at"])
    if estado.get("status") == "running":
        if instante >= fim:
            estado["status"] = "completed"
            estado["day_index"] = DURACAO_DIAS
            estado["completed_at"] = instante.isoformat()
            estado["events"].append({"type": "duration_elapsed", "at": instante.isoformat()})
        else:
            dias = math.floor(max(0.0, (instante - inicio).total_seconds()) / 86400) + 1
            estado["day_index"] = min(DURACAO_DIAS, dias)
    return estado


def registrar_sessao(state_path: Path, checkpoint_path: Path, run_id: str,
                     steps: int, evaluation_path: Path | None = None,
                     now: datetime | None = None) -> dict[str, Any]:
    estado = carregar_json(state_path)
    estado = atualizar_relogio(estado, now)
    if estado.get("status") != "running":
        raise ValueError("o ciclo não está em estado running")
    meta = validar_checkpoint(checkpoint_path)
    passo = int(meta["step"])
    if passo <= int(estado.get("last_step", -1)):
        raise ValueError("checkpoint não avançou em relação à sessão anterior")
    avaliacao = carregar_json(evaluation_path) if evaluation_path and evaluation_path.is_file() else {}
    evento = {
        "type": "training_session",
        "at": (now or agora_utc()).astimezone(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "steps_requested": int(steps),
        "step_before": int(estado["last_step"]),
        "step_after": passo,
        "checkpoint_sha256": sha256_diretorio(checkpoint_path),
    }
    for chave in ("melhor_perda_validacao", "tokens_processados", "tokens_por_segundo", "tempo_s"):
        if chave in avaliacao:
            evento[chave] = avaliacao[chave]
    estado["events"].append(evento)
    estado["completed_sessions"] = int(estado.get("completed_sessions", 0)) + 1
    estado["last_step"] = passo
    estado["checkpoint_path"] = str(checkpoint_path)
    estado["checkpoint_sha256"] = evento["checkpoint_sha256"]
    estado = atualizar_relogio(estado, now)
    salvar_json_atomico(state_path, estado)
    return estado


def mudar_status(state_path: Path, comando: str, now: datetime | None = None) -> dict[str, Any]:
    estado = carregar_json(state_path)
    estado = atualizar_relogio(estado, now)
    if comando == "pause":
        if estado.get("status") != "running":
            raise ValueError("só é possível pausar um ciclo running")
        estado["status"] = "paused"
    elif comando == "resume":
        if estado.get("status") != "paused":
            raise ValueError("só é possível retomar um ciclo paused")
        if (now or agora_utc()) >= parse_data(estado["ends_at"]):
            estado["status"] = "completed"
        else:
            estado["status"] = "running"
    salvar_json_atomico(state_path, estado)
    return estado


def main() -> None:
    parser = argparse.ArgumentParser(description="Controle auditável do ciclo de treinamento de 30 dias")
    sub = parser.add_subparsers(dest="command", required=True)
    p_start = sub.add_parser("start")
    p_start.add_argument("--state", required=True)
    p_start.add_argument("--preflight", required=True)
    p_start.add_argument("--corpus-audit", required=True)
    p_start.add_argument("--manifest", required=True)
    p_start.add_argument("--checkpoint", required=True)
    p_record = sub.add_parser("record")
    p_record.add_argument("--state", required=True)
    p_record.add_argument("--checkpoint", required=True)
    p_record.add_argument("--run-id", required=True)
    p_record.add_argument("--steps", type=int, required=True)
    p_record.add_argument("--evaluation", default=None)
    p_status = sub.add_parser("tick")
    p_status.add_argument("--state", required=True)
    p_change = sub.add_parser("pause")
    p_change.add_argument("--state", required=True)
    p_resume = sub.add_parser("resume")
    p_resume.add_argument("--state", required=True)
    p_show = sub.add_parser("status")
    p_show.add_argument("--state", required=True)
    args = parser.parse_args()

    if args.command == "start":
        result = iniciar(Path(args.state), Path(args.preflight), Path(args.corpus_audit),
                         Path(args.manifest), Path(args.checkpoint))
    elif args.command == "record":
        result = registrar_sessao(Path(args.state), Path(args.checkpoint), args.run_id,
                                  args.steps, Path(args.evaluation) if args.evaluation else None)
    elif args.command == "tick":
        caminho = Path(args.state)
        result = atualizar_relogio(carregar_json(caminho))
        salvar_json_atomico(caminho, result)
    elif args.command in ("pause", "resume"):
        result = mudar_status(Path(args.state), args.command)
    else:
        caminho = Path(args.state)
        result = atualizar_relogio(carregar_json(caminho))
        salvar_json_atomico(caminho, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
