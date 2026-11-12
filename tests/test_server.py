"""Testes do servidor de treinamento e da detecção de hardware."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from server.api import criar_servidor
from server.hardware_detection import (
    coletar_relatorio,
    detectar_cpu,
    detectar_discos,
    detectar_gpus,
    detectar_memoria,
    estimar_memoria_treino,
    recomendar_estrategia,
)
from server.job_manager import JobManager
from server.monitoring import ler_metricas, metricas_sistema


# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #
def test_deteccao_cpu() -> None:
    cpu = detectar_cpu()
    assert cpu.nucleos_logicos >= 1
    assert cpu.arquitetura


def test_deteccao_memoria() -> None:
    memoria = detectar_memoria()
    assert memoria.total_gb > 0
    assert 0 <= memoria.disponivel_gb <= memoria.total_gb + 1


def test_deteccao_discos() -> None:
    discos = detectar_discos(["/"])
    assert discos
    assert discos[0].total_gb > 0


def test_deteccao_gpus_nao_quebra_sem_gpu() -> None:
    gpus = detectar_gpus()
    assert isinstance(gpus, list)


def test_relatorio_completo() -> None:
    relatorio = coletar_relatorio()
    dados = relatorio.para_dict()
    assert "cpu" in dados and "memoria" in dados and "discos" in dados
    assert dados["torch"]


def test_estimativa_de_memoria() -> None:
    estimativa = estimar_memoria_treino(1_000_000)
    gb = 1024**3
    # pesos (4 bytes/param) + gradientes (4) + AdamW (8) = 16 bytes/param
    assert estimativa["pesos_gb"] == pytest.approx(4_000_000 / gb, rel=1e-4)
    assert estimativa["adamw_gb"] == pytest.approx(8_000_000 / gb, rel=1e-4)
    assert estimativa["total_gb"] > estimativa["pesos_gb"]


def test_recomendacao_considera_hardware() -> None:
    relatorio = coletar_relatorio()
    recomendacao = recomendar_estrategia(relatorio, 2_000_000)
    assert recomendacao["estrategia"]["modo"] in {"single", "ddp", "fsdp", "cpu"}
    assert recomendacao["recomendacoes"]


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
def test_job_manager_executa_processo(tmp_path: Path) -> None:
    gerenciador = JobManager(tmp_path / "jobs", caminho_lock=tmp_path / "treino.lock")
    job = gerenciador.iniciar("teste", [__import__("sys").executable, "-c", "print('apolo')"])
    for _ in range(50):
        gerenciador.atualizar(job.id)
        if gerenciador.jobs[job.id].status in {"concluido", "falhou"}:
            break
        time.sleep(0.1)
    assert gerenciador.jobs[job.id].status == "concluido"
    assert gerenciador.jobs[job.id].codigo_saida == 0
    assert "apolo" in gerenciador.logs(job.id)


def test_job_manager_lista_e_cancela(tmp_path: Path) -> None:
    gerenciador = JobManager(tmp_path / "jobs", caminho_lock=tmp_path / "treino.lock")
    job = gerenciador.iniciar(
        "longo", [__import__("sys").executable, "-c", "import time; time.sleep(30)"]
    )
    assert len(gerenciador.listar()) == 1
    assert gerenciador.cancelar(job.id) is True
    assert gerenciador.jobs[job.id].status == "cancelado"


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def servidor():
    srv = criar_servidor("127.0.0.1", 0, raiz=Path(__file__).resolve().parents[1])
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    porta = srv.server_address[1]
    yield f"http://127.0.0.1:{porta}"
    srv.shutdown()
    srv.server_close()


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20) as resposta:
        return json.loads(resposta.read().decode("utf-8"))


def _post(url: str, dados: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(dados).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as resposta:
        return json.loads(resposta.read().decode("utf-8"))


def test_api_health(servidor: str) -> None:
    dados = _get(f"{servidor}/api/health")
    assert dados["status"] == "ok"
    assert dados["versao"] == "1.9"


def test_api_hardware(servidor: str) -> None:
    dados = _get(f"{servidor}/api/hardware")
    assert "cpu" in dados
    assert "recomendacao" in dados


def test_api_metrics_e_models(servidor: str) -> None:
    metricas = _get(f"{servidor}/api/metrics")
    assert "sistema" in metricas and "treino" in metricas
    modelos = _get(f"{servidor}/api/models")
    assert "checkpoints" in modelos


def test_api_jobs_ciclo_completo(servidor: str) -> None:
    import sys

    criado = _post(
        f"{servidor}/api/jobs",
        {"nome": "teste-api", "comando": [sys.executable, "-c", "print('ok')"]},
    )
    job_id = criado["job"]["id"]
    for _ in range(50):
        detalhe = _get(f"{servidor}/api/jobs/{job_id}")
        if detalhe["job"]["status"] in {"concluido", "falhou"}:
            break
        time.sleep(0.1)
    assert detalhe["job"]["status"] == "concluido"
    assert "ok" in detalhe["log"]


def test_api_rota_desconhecida(servidor: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{servidor}/api/inexistente")
    assert exc.value.code == 404


def test_metricas_sistema_sao_plausiveis() -> None:
    dados = metricas_sistema()
    assert "timestamp" in dados
    if dados.get("cpu_percent") is not None:
        assert 0 <= dados["cpu_percent"] <= 100


def test_leitor_de_metricas(tmp_path: Path) -> None:
    arquivo = tmp_path / "m.jsonl"
    arquivo.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    assert len(ler_metricas(arquivo)) == 2
    assert ler_metricas(tmp_path / "inexistente.jsonl") == []
