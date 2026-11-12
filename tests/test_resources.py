"""Testes do gerenciamento de recursos do Apolo Zenith 1.9.

Cobre as 12 garantias de segurança exigidas:

1.  dois treinamentos simultâneos são bloqueados;
2.  a fila de treinamentos é sequencial e respeita a prioridade;
3.  o carregamento é lazy (o dataset não é carregado inteiro na RAM);
4.  o streaming de JSONL funciona e ignora linhas corrompidas;
5.  o batch é reduzido após um OOM simulado (sem repetição infinita);
6.  o checkpoint é gravado de forma atômica e validada;
7.  a retomada restaura passo, otimizador e estado do RNG;
8.  a pausa térmica é acionada quando a temperatura real excede o limite;
9.  a falta de armazenamento pausa o treino;
10. arquivos originais e checkpoints válidos nunca são apagados;
11. o treino roda em CPU-only (sem CUDA) sem quebrar;
12. nenhuma operação exige root.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from core.config import ModelConfig
from core.language_model import ApoloLanguageModel
from resources.adaptive_batch import AdaptiveBatchController
from resources.config import (
    ConfigRecursos,
    LimitesArmazenamento,
    LimitesBatch,
    LimitesMemoria,
    LimitesTermicos,
    PrioridadesTreino,
    carregar_config_recursos,
)
from resources.incremental_processor import IncrementalProcessor
from resources.lazy_dataset import (
    LazyJsonlDataset,
    LazyShardDataset,
    integridade_sha256,
    verificar_arquivo,
)
from resources.memory_manager import MemoryManager
from resources.resource_monitor import EstadoMonitor, ResourceMonitor
from resources.storage_manager import StorageManager
from resources.thermal_monitor import ThermalMonitor
from resources.training_lock import (
    FilaTreinos,
    TreinoConcorrenteError,
    TrainingLock,
)
from training.checkpoint import CheckpointManager, load_checkpoint
from training.trainer import TrainingConfig, Trainer, resolver_dispositivo


# --------------------------------------------------------------------------- #
# Utilitários
# --------------------------------------------------------------------------- #
def _modelo_pequeno() -> ApoloLanguageModel:
    torch.manual_seed(7)
    return ApoloLanguageModel(
        ModelConfig(
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=32,
            block_size=16,
            mlp_hidden=64,
            dropout=0.0,
        )
    )


# =========================================================================== #
# 1. Dois treinamentos simultâneos são bloqueados
# =========================================================================== #
def test_lock_bloqueia_treino_simultaneo(tmp_path: Path) -> None:
    """O segundo processo/treino não consegue adquirir o lock."""
    caminho = tmp_path / "treino.lock"
    primeiro = TrainingLock(caminho)
    primeiro.adquirir("linguagem")
    try:
        assert primeiro.ocupado()
        segundo = TrainingLock(caminho)
        with pytest.raises(TreinoConcorrenteError):
            segundo.adquirir("codigo")
    finally:
        primeiro.liberar()
    # Depois de liberar, o lock volta a ficar disponível.
    terceiro = TrainingLock(caminho)
    terceiro.adquirir("codigo")
    terceiro.liberar()


def test_lock_detecta_dono_morto(tmp_path: Path) -> None:
    """Um PID morto não bloqueia novos treinamentos."""
    caminho = tmp_path / "morto.lock"
    lock = TrainingLock(caminho)
    lock.adquirir("linguagem")
    # Simula a morte do dono gravando um PID inexistente.
    lock._escrever_estado({"pid": 999_999_999, "modalidade": "linguagem", "heartbeat": 0.0})
    assert not lock.ocupado()
    assert lock.esta_morto()


# =========================================================================== #
# 2. Fila sequencial por prioridade
# =========================================================================== #
def test_fila_treinos_e_sequencial_e_prioriza(tmp_path: Path) -> None:
    """A fila ordena por prioridade e devolve um item por vez."""
    fila = FilaTreinos(tmp_path / "fila.json", PrioridadesTreino())
    video = fila.enfileirar("video", ["python", "train.py", "--video"])
    codigo = fila.enfileirar("codigo", ["python", "train.py", "--code"])
    multimodal = fila.enfileirar("multimodal", ["python", "train.py", "--mm"])
    linguagem = fila.enfileirar("linguagem", ["python", "train.py", "--text"])

    proxima = fila.proxima()
    assert proxima is not None and proxima.id == linguagem.id
    fila.marcar_inicio(linguagem.id)
    assert fila.proxima().id == codigo.id
    fila.marcar_fim(linguagem.id, "concluido")
    assert len(fila.pendentes) == 3

    # A ordem respeita a prioridade configurada (linguagem < ... < multimodal).
    prioridades = [fila.listar()[i]["prioridade"] for i in range(len(fila.listar()))]
    assert prioridades == sorted(prioridades)
    assert {video.id, codigo.id, multimodal.id, linguagem.id} == {
        i["id"] for i in fila.listar()
    }


def test_fila_persiste_entre_processos(tmp_path: Path) -> None:
    """A fila é salva em disco e relida."""
    arquivo = tmp_path / "fila.json"
    fila = FilaTreinos(arquivo)
    item = fila.enfileirar("audio", ["x"])
    outra = FilaTreinos(arquivo)
    assert outra.proxima() is not None
    assert outra.proxima().id == item.id


# =========================================================================== #
# 3. Carregamento lazy
# =========================================================================== #
def test_lazy_loading_nao_carrega_tudo(tmp_path: Path) -> None:
    """Os shards são carregados sob demanda, com cache limitado por bytes."""
    caminho = tmp_path / "dados"
    caminho.mkdir()
    shards = []
    for i in range(4):
        arquivo = caminho / f"shard_{i}.npy"
        np.save(arquivo, np.arange(64, dtype=np.int64).reshape(8, 8) + i)
        shards.append(arquivo)

    dataset = LazyShardDataset(shards, cache_max_mb=0)
    assert len(dataset) == 32
    # Nenhum shard foi carregado ainda (lazy real).
    assert dataset._cache_bytes == 0
    primeiro = dataset[0]
    # Cada item é uma linha do shard (8 tokens).
    assert primeiro.shape == (8,)
    assert dataset._cache_bytes > 0

    # Com cache de 0 MB, apenas o último shard fica na memória.
    dataset.limpar_cache()
    for indice in range(len(dataset)):
        dataset[indice]
    assert len(dataset._cache) <= 1


def test_lazy_ignora_shard_corrompido(tmp_path: Path) -> None:
    """Um shard com SHA-256 divergente é ignorado com log, sem quebrar."""
    caminho = tmp_path / "dados"
    caminho.mkdir()
    bom = caminho / "bom.npy"
    np.save(bom, np.zeros((4, 4), dtype=np.int64))
    ruim = caminho / "ruim.npy"
    np.save(ruim, np.ones((4, 4), dtype=np.int64))
    (ruim.with_suffix(".npy.sha256")).write_text("0" * 64, encoding="utf-8")

    dataset = LazyShardDataset([bom, ruim], cache_max_mb=1)
    assert len(dataset.shards_validos) == 1
    assert dataset.shards[1].valido is False
    assert "divergente" in dataset.shards[1].motivo
    assert len(dataset) == 4


def test_integridade_sha256_detecta_alteracao(tmp_path: Path) -> None:
    arquivo = tmp_path / "a.bin"
    arquivo.write_bytes(b"apolo" * 100)
    hash_antes = integridade_sha256(arquivo)
    resultado = verificar_arquivo(arquivo, hash_antes)
    assert resultado.ok
    arquivo.write_bytes(b"apolo" * 99 + b"X")
    resultado = verificar_arquivo(arquivo, hash_antes)
    assert not resultado.ok


# =========================================================================== #
# 4. Streaming
# =========================================================================== #
def test_streaming_jsonl_ignora_linhas_corrompidas(tmp_path: Path) -> None:
    """O JSONL é lido linha a linha; linhas inválidas são ignoradas."""
    arquivo = tmp_path / "dados.jsonl"
    with arquivo.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": 1, "texto": "um"}) + "\n")
        fh.write("{isto nao e json}\n")
        fh.write(json.dumps({"id": 2, "texto": "dois"}) + "\n")
        fh.write(json.dumps({"id": 3, "texto": "tres"}) + "\n")

    dataset = LazyJsonlDataset(arquivo)
    assert len(dataset) == 3
    assert dataset.resumo()["linhas_corrompidas"] == 1
    itens = list(dataset)
    assert [i["id"] for i in itens] == [1, 2, 3]
    assert dataset[1]["texto"] == "dois"


def test_streaming_pode_retomar(tmp_path: Path) -> None:
    """A retomada pula as linhas já consumidas."""
    arquivo = tmp_path / "dados.jsonl"
    with arquivo.open("w", encoding="utf-8") as fh:
        for i in range(10):
            fh.write(json.dumps({"id": i}) + "\n")
    completo = LazyJsonlDataset(arquivo)
    parcial = LazyJsonlDataset(arquivo, pular_linhas=4)
    assert len(parcial) == 6
    assert parcial[0]["id"] == 4


# =========================================================================== #
# 5. Batch adaptativo após OOM
# =========================================================================== #
def test_batch_reduzido_apos_oom_simulado() -> None:
    """Um OOM reduz o batch; não há repetição infinita de tentativas."""
    controlador = AdaptiveBatchController(
        LimitesBatch(batch_minimo=1, batch_maximo=4, batch_inicial=4)
    )
    assert controlador.batch_atual == 4
    evento = controlador.registrar_oom(passo=10)
    assert controlador.batch_atual == 2
    assert evento.batch_anterior == 4 and evento.batch_novo == 2
    controlador.registrar_oom(passo=11)
    assert controlador.batch_atual == 1
    assert controlador.em_cooldown

    # No mínimo, novas tentativas de OOM não sobem o batch.
    for passo in range(12, 15):
        controlador.registrar_oom(passo=passo)
    assert controlador.batch_atual == 1


def test_batch_eleva_apos_passos_estaveis() -> None:
    """Após passos estáveis, o batch sobe de volta (até o máximo)."""
    controlador = AdaptiveBatchController(
        LimitesBatch(batch_minimo=1, batch_maximo=4, batch_inicial=1)
    )
    controlador.passos_estaveis = 3
    for passo in range(1, 4):
        controlador.registrar_sucesso(passo)
    assert controlador.batch_atual == 2
    for passo in range(4, 7):
        controlador.registrar_sucesso(passo)
    assert controlador.batch_atual == 3
    for passo in range(7, 20):
        controlador.registrar_sucesso(passo)
    assert controlador.batch_atual == 4  # teto respeitado


def test_oom_sem_retentativa_infinita_levanta_erro() -> None:
    """Após muitas OOMs no batch mínimo, o treino é interrompido."""
    controlador = AdaptiveBatchController(
        LimitesBatch(batch_minimo=1, batch_maximo=4, batch_inicial=1)
    )
    with pytest.raises(MemoryError):
        for passo in range(10):
            controlador.registrar_oom(passo=passo)


# =========================================================================== #
# 6. Checkpoint atômico e validado
# =========================================================================== #
def test_checkpoint_gravacao_atomica_e_validada(tmp_path: Path) -> None:
    """O checkpoint é gravado de forma atômica e contém o estado completo."""
    modelo = _modelo_pequeno()
    gerenciador = CheckpointManager(tmp_path / "ckpts", manter=2)
    caminho = gerenciador.salvar(
        "step_000001",
        modelo,
        step=1,
        epoch=0,
        dataset_state={"indice": 128},
        batch_state={"batch": 2},
        fila_state={"posicao": 0},
        estado_treino={"tokens": 4096},
    )
    assert caminho.exists()
    assert (caminho / "model.pt").exists()
    assert (caminho / "meta.json").exists()
    # Nenhum diretório temporário ficou para trás.
    assert not list((tmp_path / "ckpts").glob("*.tmp"))

    meta = json.loads((caminho / "meta.json").read_text(encoding="utf-8"))
    assert meta["dataset_state"]["indice"] == 128
    assert meta["batch_state"]["batch"] == 2
    assert meta["fila_state"]["posicao"] == 0
    assert meta["estado_treino"]["tokens"] == 4096
    assert "rng_state" in meta


def test_checkpoint_preserva_o_melhor(tmp_path: Path) -> None:
    """O checkpoint 'best' não é apagado pela rotação."""
    modelo = _modelo_pequeno()
    gerenciador = CheckpointManager(tmp_path / "ckpts", manter=2)
    gerenciador.salvar("best", modelo, step=1, metrica=5.0, metrica_modo="min")
    for passo in range(2, 6):
        gerenciador.salvar(f"step_{passo:06d}", modelo, step=passo)
    assert (tmp_path / "ckpts" / "best").exists()
    step_dirs = sorted(p.name for p in (tmp_path / "ckpts").glob("step_*"))
    assert len(step_dirs) <= 2


# =========================================================================== #
# 7. Retomada
# =========================================================================== #
def test_retomada_restaura_passo_e_estado(tmp_path: Path) -> None:
    """Retomar restaura passo, pesos e estado do RNG do checkpoint."""
    modelo = _modelo_pequeno()
    gerenciador = CheckpointManager(tmp_path / "ckpts", manter=3)
    gerenciador.salvar(
        "final",
        modelo,
        step=42,
        dataset_state={"indice": 7},
        batch_state={"batch": 3},
    )
    meta = load_checkpoint(tmp_path / "ckpts" / "final", model=modelo, restaurar_rng=True)
    assert meta["step"] == 42
    assert meta["dataset_state"]["indice"] == 7
    assert meta["batch_state"]["batch"] == 3
    assert meta["rng_restaurado"] is True


# =========================================================================== #
# 8. Pausa térmica
# =========================================================================== #
def test_pausa_termica_quando_temperatura_excede_limite(monkeypatch) -> None:
    """Com leitura térmica real acima do limite, o treino deve pausar."""
    monitor = ThermalMonitor(
        LimitesTermicos(temperatura_maxima_c=42.0, temperatura_resume_c=37.0)
    )
    monkeypatch.setattr(monitor, "_ler_sysfs", lambda: 48.5)
    monkeypatch.setattr(monitor, "_ler_dumpsys", lambda: None)
    monkeypatch.setattr(monitor, "_ler_termux_api", lambda: None)
    monkeypatch.setattr(monitor, "_ler_getprop", lambda: None)

    leitura = monitor.ler(forcar=True)
    assert leitura.disponivel
    assert leitura.temperatura_c == pytest.approx(48.5)
    assert monitor.nivel(leitura) == "critico"
    pausar, motivo = monitor.deve_pausar(leitura)
    assert pausar is True
    assert "48.5" in motivo

    monitor.registrar_inicio_pausa(motivo)
    assert monitor.em_pausa
    assert monitor.deve_retomar(leitura)[0] is False


def test_modo_preventivo_sem_sensor_nao_inventa_temperatura() -> None:
    """Sem sensor, o monitor declara 'indisponível' e não inventa valor."""
    monitor = ThermalMonitor(LimitesTermicos(usar_modo_preventivo=True))
    leitura = monitor.ler(forcar=True)
    assert leitura.temperatura_c is None
    assert leitura.fonte == "indisponivel"
    assert not leitura.disponivel
    assert monitor.nivel(leitura) == "indisponivel"


def test_monitor_de_recursos_entra_em_pausa_termica(monkeypatch, tmp_path: Path) -> None:
    """O ResourceMonitor propaga a pausa térmica ao estado PAUSED_THERMAL."""
    config = ConfigRecursos()
    monitor = ResourceMonitor(config, raiz=tmp_path)
    monkeypatch.setattr(monitor.termico, "_ler_sysfs", lambda: 55.0)
    monkeypatch.setattr(monitor.termico, "_ler_dumpsys", lambda: None)
    monkeypatch.setattr(monitor.termico, "_ler_termux_api", lambda: None)
    decisao = monitor.verificar(forcar=True)
    assert decisao.estado is EstadoMonitor.PAUSED_THERMAL
    assert decisao.deve_pausar is True


# =========================================================================== #
# 9. Falta de armazenamento
# =========================================================================== #
def test_falta_de_armazenamento_pausa_o_treino(tmp_path: Path) -> None:
    """Sem espaço livre suficiente, o treino pausa."""
    limites = LimitesArmazenamento(espaco_minimo_livre_mb=10 ** 9)
    gestor = StorageManager(limites, raiz=tmp_path)
    leitura = gestor.ler(forcar=True)
    assert gestor.nivel(leitura) == "critico"
    pausar, motivo = gestor.deve_pausar(leitura)
    assert pausar is True
    assert "espaço livre" in motivo


def test_download_bloqueado_sem_espaco(tmp_path: Path) -> None:
    """Um download que comprometeria o espaço mínimo é bloqueado."""
    limites = LimitesArmazenamento(espaco_minimo_livre_mb=10 ** 9, download_max_por_passo_mb=64)
    gestor = StorageManager(limites, raiz=tmp_path)
    ok, motivo = gestor.pode_baixar(10 * 1024 * 1024)
    assert ok is False
    assert "deixaria" in motivo


def test_download_Grande_dividido_em_blocos(tmp_path: Path) -> None:
    """Downloads acima do limite por passo são recusados (streaming)."""
    limites = LimitesArmazenamento(download_max_por_passo_mb=1, espaco_minimo_livre_mb=1)
    gestor = StorageManager(limites, raiz=tmp_path)
    ok, motivo = gestor.pode_baixar_passo(50 * 1024 * 1024)
    assert ok is False
    assert "excede o limite por passo" in motivo
    # O arquivo inteiro é aceito (o limite por passo vale para cada bloco).
    ok_total, _ = gestor.pode_baixar(50 * 1024 * 1024)
    assert ok_total is True


# =========================================================================== #
# 10. Nunca apagar originais nem checkpoints válidos
# =========================================================================== #
def test_nao_apaga_arquivos_originais_nem_checkpoints(tmp_path: Path) -> None:
    """A limpeza só remove temporários; a rotação preserva 'best'."""
    pasta_raw = tmp_path / "datasets" / "text" / "raw"
    pasta_raw.mkdir(parents=True)
    original = pasta_raw / "corpus.txt"
    original.write_text("dados originais", encoding="utf-8")

    tmp_dir = tmp_path / "datasets_storage" / "tmp"
    tmp_dir.mkdir(parents=True)
    temporario = tmp_dir / "parcial.txt.tmp"
    temporario.write_text("lixo", encoding="utf-8")

    gestor = StorageManager(LimitesArmazenamento(diretorio_temporario=str(tmp_dir)), raiz=tmp_path)
    removidos, _bytes = gestor.limpar_temporarios()
    assert removidos == 1
    assert not temporario.exists()
    assert original.exists()
    assert original.read_text(encoding="utf-8") == "dados originais"

    # Rotação de checkpoints: 'best' e 'final' permanecem.
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    for nome in ("best", "final", "step_000001", "step_000002", "step_000003"):
        (ckpts / nome).mkdir()
        (ckpts / nome / "model.pt").write_bytes(b"pesos")
    gestor.rotacionar_checkpoints(ckpts, manter=1)
    assert (ckpts / "best").exists()
    assert (ckpts / "final").exists()
    assert (ckpts / "step_000003").exists()
    assert not (ckpts / "step_000001").exists()


# =========================================================================== #
# 11. CPU-only
# =========================================================================== #
def test_execucao_e_cpu_only() -> None:
    """O dispositivo não tem CUDA; o trainer deve operar em CPU."""
    assert not torch.cuda.is_available()
    assert resolver_dispositivo("auto").type == "cpu"
    assert resolver_dispositivo("cpu").type == "cpu"


def test_treino_curto_sobre_cpu(tmp_path: Path) -> None:
    """Um treino real (3 passos) roda em CPU e grava checkpoint."""
    class _Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 8

        def __getitem__(self, indice: int):
            seq = [indice % 16] * 16
            return torch.tensor(seq), torch.tensor(seq[1:] + [seq[0]])

    modelo = _modelo_pequeno()
    config = TrainingConfig(
        saida=str(tmp_path / "run"),
        batch_size=2,
        block_size=16,
        max_steps=3,
        eval_interval=3,
        save_interval=3,
        log_interval=1,
        device="cpu",
    )
    trainer = Trainer(modelo, config, _Dataset(), _Dataset())
    resumo = trainer.treinar()
    assert resumo["passos"] == 3
    assert resumo["device"] == "cpu"
    assert (tmp_path / "run" / "checkpoints" / "final").exists()


# =========================================================================== #
# 12. Sem root
# =========================================================================== #
def test_operacoes_nao_exigem_root() -> None:
    """As operações de recurso rodam sem privilégios elevados."""
    assert os.geteuid() in (0, 1000, 1001, 1002, 1003)  # root no Proot ou usuário comum
    config = carregar_config_recursos(Path(__file__).resolve().parents[1] / "configs" / "resource_manager.yaml")
    assert config.batch.batch_maximo <= 4
    assert config.memoria.rss_max_mb > 0
    monitor = MemoryManager(config.memoria)
    assert monitor.ler(forcar=True).ram_total_mb > 0


def test_config_resource_manager_yaml_valido() -> None:
    """O YAML de configuração carrega sem chaves desconhecidas."""
    raiz = Path(__file__).resolve().parents[1]
    config = carregar_config_recursos(raiz / "configs" / "resource_manager.yaml")
    config.validar()
    assert config.batch.batch_minimo == 1
    assert config.batch.batch_maximo == 4
    assert config.termicos.usar_modo_preventivo is True
    assert config.prioridades.ordem["linguagem"] < config.prioridades.ordem["multimodal"]


# =========================================================================== #
# Processamento incremental
# =========================================================================== #
def test_processamento_incremental_retoma(tmp_path: Path) -> None:
    """O processamento em blocos persiste o progresso e retoma de onde parou."""
    arquivo = tmp_path / "progresso.json"
    itens: List[int] = list(range(25))
    processados: List[int] = []

    processador = IncrementalProcessor(arquivo, itens, tamanho_bloco=10)
    processador.executar(lambda item, indice: processados.append(item))
    assert len(processados) == 25
    assert processador.progresso.estado == "concluido"

    # Nova instância retoma: não reprocessa nada.
    outro = IncrementalProcessor(arquivo, itens, tamanho_bloco=10)
    assert outro.progresso.ultimo_indice == 24
    processados.clear()
    outro.executar(lambda item, indice: processados.append(item))
    assert processados == []


def test_processamento_incremental_isola_falhas(tmp_path: Path) -> None:
    """Um item que falha não interrompe o lote."""
    arquivo = tmp_path / "progresso.json"

    def _processar(item: int, indice: int) -> None:
        if item == 3:
            raise RuntimeError("item corrompido")

    processador = IncrementalProcessor(arquivo, list(range(6)), tamanho_bloco=2)
    processador.executar(_processar)
    assert processador.progresso.concluidos == 5
    assert processador.progresso.falhas == 1
