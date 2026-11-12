"""Gerenciamento de recursos do Apolo Zenith 1.9.

Este pacote centraliza as proteções de segurança do dispositivo móvel:

* :mod:`resources.config`             – limites configuráveis (``configs/resource_manager.yaml``);
* :mod:`resources.thermal_monitor`    – leitura térmica real + modo preventivo;
* :mod:`resources.memory_manager`     – controle de RAM (sistema, RSS e RLIMIT);
* :mod:`resources.storage_manager`    – controle de disco e retenção de checkpoints;
* :mod:`resources.resource_monitor`   – orquestração dos estados de execução;
* :mod:`resources.training_lock`      – exclusividade de treino + fila por prioridade;
* :mod:`resources.adaptive_batch`     – batch adaptativo (1–4) com reação a OOM;
* :mod:`resources.lazy_dataset`       – carregamento lazy com cache limitado por bytes;
* :mod:`resources.incremental_processor` – processamento incremental com progresso persistido.

Regra central: **nunca mais de um treinamento por vez** e **nunca ultrapassar
os limites configurados**.
"""

from resources.adaptive_batch import AdaptiveBatchController, EventoBatch
from resources.config import ConfigRecursos, carregar_config_recursos
from resources.incremental_processor import IncrementalProcessor, ProgressoIncremental
from resources.lazy_dataset import (
    LazyJsonlDataset,
    LazyShardDataset,
    integridade_sha256,
    verificar_arquivo,
)
from resources.memory_manager import LeituraMemoria, MemoryManager
from resources.resource_monitor import DecisaoMonitor, EstadoMonitor, ResourceMonitor
from resources.storage_manager import LeituraArmazenamento, StorageManager
from resources.thermal_monitor import LeituraTermica, ThermalMonitor
from resources.training_lock import (
    FilaTreinos,
    TreinoConcorrenteError,
    TreinoSolicitacao,
    TrainingLock,
)

__all__ = [
    "AdaptiveBatchController",
    "ConfigRecursos",
    "DecisaoMonitor",
    "EstadoMonitor",
    "EventoBatch",
    "FilaTreinos",
    "IncrementalProcessor",
    "LeituraArmazenamento",
    "LeituraMemoria",
    "LeituraTermica",
    "LazyJsonlDataset",
    "LazyShardDataset",
    "MemoryManager",
    "ProgressoIncremental",
    "ResourceMonitor",
    "StorageManager",
    "ThermalMonitor",
    "TreinoConcorrenteError",
    "TreinoSolicitacao",
    "TrainingLock",
    "carregar_config_recursos",
    "integridade_sha256",
    "verificar_arquivo",
]
