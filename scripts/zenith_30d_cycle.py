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


def localizar_checkpoint_recuperavel(raiz: Path) -> tuple[Path, dict[str, Any]]:
    """Seleciona o checkpoint completo de maior passo, ignorando gravações truncadas."""
    if not raiz.is_dir():
        raise FileNotFoundError(f"diretório de checkpoints não encontrado: {raiz}")
    candidatos: list[tuple[int, Path, dict[str, Any]]] = []
    for pasta in raiz.iterdir():
        if not pasta.is_dir() or pasta.name.endswith((".tmp", ".old", ".copytmp")):
            continue
        try:
            meta = validar_checkpoint(pasta)
            # A retomada de treino exige pesos, AdamW e scheduler, não só inferência.
            if not all((pasta / nome).is_file() and (pasta / nome).stat().st_size > 0
                       for nome in ("model.pt", "optimizer.pt", "scheduler.pt", "meta.json")):
                continue
            candidatos.append((int(meta["step"]), pasta, meta))
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            continue
    if not candidatos:
        raise ValueError(f"nenhum checkpoint completo e recuperável em {raiz}")
    passo, pasta, meta = max(candidatos, key=lambda item: (item[0], item[1].name))
    return pasta, meta


def validar_pre_requisitos(preflight_path: Path, audit_path: Path, manifest_path: Path,
                           expected_commit: str | None = None,
                           benchmark_path: Path | None = None) -> dict[str, Any]:
    preflight = carregar_json(preflight_path)
    audit = carregar_json(audit_path)
    if preflight.get("status") != "passed" or not preflight.get("checks"):
        raise ValueError("preflight não aprovado")
    if not all(value is True for value in preflight["checks"].values()):
        raise ValueError("há verificações de preflight não aprovadas")
    if expected_commit and preflight.get("commit") != expected_commit:
        raise ValueError("preflight foi executado em outro commit; repetir validação no HEAD atual")
    if audit.get("passed") is not True:
        raise ValueError("auditoria do corpus não aprovada")
    benchmark_path = benchmark_path or audit_path.with_name("benchmark.json")
    benchmark = carregar_json(benchmark_path)
    if benchmark.get("status") != "passed" or benchmark.get("device") != "cpu":
        raise ValueError("benchmark real de CPU não aprovado")
    if expected_commit and benchmark.get("source_commit") != expected_commit:
        raise ValueError("benchmark foi executado em outro commit; reconstruir corpus e benchmark")
    if int(benchmark.get("steps", 0)) != 1000:
        raise ValueError("benchmark precisa comprovar exatamente 1.000 passos")
    if int(benchmark.get("evaluation_count", 0)) < 10:
        raise ValueError("benchmark precisa conter pelo menos 10 avaliações")
    elapsed = float(benchmark.get("elapsed_seconds", 0))
    if not (0 < elapsed <= 300 * 60):
        raise ValueError("benchmark excedeu a janela operacional de 300 minutos")
    if expected_commit and audit.get("source_commit") != expected_commit:
        raise ValueError("corpus foi construído em outro commit; reconstruir no HEAD atual")
    docs = audit.get("documents", {})
    por_idioma = docs.get("by_language", {})
    if int(docs.get("total", 0)) != DOCUMENTOS_ESPERADOS:
        raise ValueError("corpus não contém exatamente 30.000 documentos")
    if int(por_idioma.get("pt", 0)) < MIN_DOCUMENTOS_IDIOMA:
        raise ValueError("corpus não atingiu 15.000 documentos em português")
    if int(por_idioma.get("en", 0)) < MIN_DOCUMENTOS_IDIOMA:
        raise ValueError("corpus não atingiu 15.000 documentos em inglês")

    # Revalidar cada gate, sem confiar apenas no campo agregado "passed".
    gates = audit.get("gates")
    obrigatorios = (
        "all_manifest_documents_processed", "minimum_processed_documents",
        "minimum_pt_documents", "minimum_en_documents", "training_tokens",
        "validation_tokens", "test_tokens", "manifest_files_and_hashes",
        "unique_document_hashes", "approved_licenses", "expected_total_documents",
        "declared_licenses", "both_languages_present",
    )
    if not isinstance(gates, dict):
        raise ValueError("auditoria sem gates individuais; reconstruir o corpus")
    falhas = [nome for nome in obrigatorios if gates.get(nome) is not True]
    if falhas:
        raise ValueError("gates individuais reprovados: " + ", ".join(falhas))
    if int(docs.get("processed_final_documents", 0)) < 27000:
        raise ValueError("menos de 27.000 documentos processados")
    if int(docs.get("integrity_checked", 0)) != DOCUMENTOS_ESPERADOS:
        raise ValueError("hashes não verificados para os 30.000 documentos")
    if int(docs.get("integrity_failures", -1)) != 0:
        raise ValueError("há hashes divergentes ou arquivos ausentes")
    if int(docs.get("duplicate_hashes", -1)) != 0:
        raise ValueError("há hashes duplicados")
    tokens = audit.get("tokens_by_split", {})
    thresholds = audit.get("thresholds", {})
    if int(tokens.get("train", 0)) < max(1000000, int(thresholds.get("minimum_train_tokens", 0))):
        raise ValueError("tokens de treinamento abaixo do mínimo")
    minimo_avaliacao = max(10000, int(thresholds.get("minimum_val_test_tokens_each", 0)))
    if int(tokens.get("val", 0)) < minimo_avaliacao or int(tokens.get("test", 0)) < minimo_avaliacao:
        raise ValueError("tokens de validação/teste abaixo do mínimo")
    licencas = docs.get("licenses", {})
    if not licencas or any(not str(nome).strip() or int(qtd) <= 0 for nome, qtd in licencas.items()):
        raise ValueError("contagem de licenças ausente ou inválida")
    if not manifest_path.is_file():
        raise ValueError("manifesto de proveniência ausente")
    release_tag = str(audit.get("release_tag", "")).strip()
    if not release_tag.startswith("apolo-zenith-corpus-30000-"):
        raise ValueError("corpus sem release imutável identificado")
    manifest_hash = sha256_arquivo(manifest_path)
    if audit.get("manifest_sha256") != manifest_hash:
        raise ValueError("hash do manifesto não corresponde à auditoria")
    return {
        "preflight_run_id": preflight.get("run_id"),
        "preflight_commit": preflight.get("commit"),
        "corpus_source_commit": audit.get("source_commit"),
        "corpus_documents": int(docs["total"]),
        "documents_by_language": por_idioma,
        "manifest_sha256": manifest_hash,
        "release_tag": release_tag,
        "corpus_audit_passed": True,
        "cpu_benchmark": {
            "status": benchmark["status"],
            "steps": int(benchmark["steps"]),
            "elapsed_seconds": elapsed,
            "evaluation_count": int(benchmark["evaluation_count"]),
        },
    }


def iniciar(state_path: Path, preflight_path: Path, audit_path: Path,
            manifest_path: Path, checkpoint_path: Path, now: datetime | None = None,
            expected_commit: str | None = None,
            benchmark_path: Path | None = None) -> dict[str, Any]:
    if state_path.exists():
        atual = carregar_json(state_path)
        if atual.get("status") in ("running", "paused"):
            raise ValueError("já existe um ciclo ativo ou pausado; não iniciar outro")
    requisitos = validar_pre_requisitos(preflight_path, audit_path, manifest_path, expected_commit, benchmark_path)
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
    ultimo_ping = estado.get("last_seen_at")
    if ultimo_ping and estado.get("status") in ("running", "paused"):
        lacuna_horas = max(0.0, (instante - parse_data(ultimo_ping)).total_seconds() / 3600.0)
        estado["max_observed_gap_hours"] = max(float(estado.get("max_observed_gap_hours", 0.0)), lacuna_horas)
        if lacuna_horas > 12.0:
            estado["events"].append({
                "type": "monitoring_gap",
                "at": instante.isoformat(),
                "gap_hours": round(lacuna_horas, 2),
                "threshold_hours": 12.0,
            })
    estado["last_seen_at"] = instante.isoformat()
    inicio = parse_data(estado["started_at"])
    fim = parse_data(estado["ends_at"])
    if estado.get("status") in ("running", "paused"):
        if instante >= fim:
            estado["status"] = "completed"
            estado["day_index"] = DURACAO_DIAS
            estado["completed_at"] = instante.isoformat()
            estado["events"].append({"type": "duration_elapsed", "at": instante.isoformat()})
        elif estado.get("status") == "running":
            dias = math.floor(max(0.0, (instante - inicio).total_seconds()) / 86400) + 1
            estado["day_index"] = min(DURACAO_DIAS, dias)
    return estado


def registrar_sessao(state_path: Path, checkpoint_path: Path, run_id: str,
                     steps: int, evaluation_path: Path | None = None,
                     now: datetime | None = None) -> dict[str, Any]:
    estado = carregar_json(state_path)
    if estado.get("status") != "running":
        raise ValueError("o ciclo não está em estado running")
    meta = validar_checkpoint(checkpoint_path)
    passo = int(meta["step"])
    passo_anterior = int(estado.get("last_step", -1))
    if passo != passo_anterior + int(steps):
        raise ValueError(
            f"sessão incompleta: esperado passo {passo_anterior + int(steps)}, recebido {passo}"
        )
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
    if estado.get("status") == "completed":
        salvar_json_atomico(state_path, estado)
        return estado
    if comando == "pause":
        if estado.get("status") != "running":
            raise ValueError("só é possível pausar um ciclo running")
        estado["status"] = "paused"
    elif comando == "resume":
        if estado.get("status") != "paused":
            raise ValueError("só é possível retomar um ciclo paused")
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
    p_start.add_argument("--expected-commit", default=None)
    p_start.add_argument("--benchmark", required=True)
    p_recover = sub.add_parser("recover")
    p_recover.add_argument("--root", required=True)
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
    p_mark = sub.add_parser("mark-evaluated")
    p_mark.add_argument("--state", required=True)
    p_mark.add_argument("--report", required=True)
    args = parser.parse_args()

    if args.command == "recover":
        pasta, meta = localizar_checkpoint_recuperavel(Path(args.root))
        print(json.dumps({"path": str(pasta), "step": int(meta["step"])}, ensure_ascii=False))
        return
    if args.command == "start":
        result = iniciar(Path(args.state), Path(args.preflight), Path(args.corpus_audit),
                         Path(args.manifest), Path(args.checkpoint), expected_commit=args.expected_commit,
                         benchmark_path=Path(args.benchmark))
    elif args.command == "record":
        result = registrar_sessao(Path(args.state), Path(args.checkpoint), args.run_id,
                                  args.steps, Path(args.evaluation) if args.evaluation else None)
    elif args.command == "tick":
        caminho = Path(args.state)
        result = atualizar_relogio(carregar_json(caminho))
        salvar_json_atomico(caminho, result)
    elif args.command in ("pause", "resume"):
        result = mudar_status(Path(args.state), args.command)
    elif args.command == "mark-evaluated":
        caminho = Path(args.state)
        result = carregar_json(caminho)
        relatorio = carregar_json(Path(args.report))
        result["final_evaluation_done"] = True
        result["final_evaluation"] = {
            "status": relatorio.get("status"),
            "decision": relatorio.get("decision"),
            "training_authorization": relatorio.get("training_authorization"),
            "report_path": args.report,
            "marked_at": agora_utc().isoformat(),
        }
        result["events"].append({"type": "final_evaluation_completed", "at": result["final_evaluation"]["marked_at"], "decision": relatorio.get("decision")})
        salvar_json_atomico(caminho, result)
    else:
        caminho = Path(args.state)
        result = atualizar_relogio(carregar_json(caminho))
        salvar_json_atomico(caminho, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
