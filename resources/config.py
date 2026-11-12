"""Configuração central de gerenciamento de recursos do Apolo Zenith 1.9.

Todos os limites são **configuráveis** e vivem em ``configs/resource_manager.yaml``.
Os valores padrão são conservadores para um celular Android e devem ser
revisados após a inspeção do dispositivo (``python main.py recursos``).

A configuração é carregada de forma tolerante: campos ausentes usam o padrão,
campos desconhecidos geram erro (para não haver erro de digitação silencioso).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class LimitesTermicos:
    """Limites térmicos e política de pausa."""

    temperatura_maxima_c: float = 42.0          # acima disso: PAUSA_TERMICA
    temperatura_resume_c: float = 37.0          # abaixo disso: retoma
    temperatura_aviso_c: float = 39.0           # WARNING
    zona_preferida: Optional[str] = None        # ex.: "thermal_zone89"
    usar_modo_preventivo: bool = True           # ativo quando não há sensor
    pausa_preventiva_min: float = 5.0           # duração da pausa preventiva
    ciclo_trabalho_min: float = 30.0            # min de trabalho antes da pausa preventiva
    intervalo_leitura_s: float = 30.0
    carga_cpu_proxy_limite: float = 0.92        # apenas informativo (proxy, não pausa)


@dataclass
class LimitesMemoria:
    """Limites de RAM do processo e do sistema."""

    ram_max_utilizacao_percent: float = 80.0    # % do total do sistema
    ram_minima_disponivel_mb: int = 512         # abaixo disso: CRITICAL
    rss_max_mb: int = 2048                      # RSS do processo
    margem_livre_mb: int = 256
    gc_interval_steps: int = 50
    liberar_entre_blocos: bool = True


@dataclass
class LimitesArmazenamento:
    """Limites de espaço em disco."""

    espaco_minimo_livre_mb: int = 2048
    cache_max_mb: int = 256
    temporarios_max_mb: int = 512
    checkpoints_max_mb: int = 3072
    download_max_por_passo_mb: int = 64
    reservar_para_sistema_mb: int = 1024
    diretorio_temporario: str = "datasets_storage/tmp"
    manter_checkpoints: int = 3


@dataclass
class LimitesBatch:
    """Limites de batch e sequência (conservadores para CPU móvel)."""

    batch_minimo: int = 1
    batch_maximo: int = 4
    batch_inicial: int = 1
    seq_len_minima: int = 32
    seq_len_maxima: int = 64
    seq_len_inicial: int = 64
    grad_accum_steps: int = 4
    num_workers: int = 0
    pin_memory: bool = False
    permitir_aumento_automatico: bool = True


@dataclass
class LimitesMonitoramento:
    """Frequência de verificações e de salvamento."""

    intervalo_monitoramento_s: float = 15.0
    intervalo_checkpoint_steps: int = 100
    checkpoint_por_bloco: bool = True
    intervalo_pausa_verificacao_s: float = 60.0
    max_tentativas_oom: int = 3
    cooldown_apos_oom_steps: int = 20


@dataclass
class PrioridadesTreino:
    """Ordem de prioridade da fila de treinamentos (nunca simultâneos)."""

    ordem: Dict[str, int] = field(
        default_factory=lambda: {
            "linguagem": 1,
            "codigo": 2,
            "visao": 3,
            "audio": 4,
            "video": 5,
            "multimodal": 6,
        }
    )
    permitir_pausa_para_trocar: bool = True


@dataclass
class ConfigRecursos:
    """Configuração completa do gerenciamento de recursos."""

    termicos: LimitesTermicos = field(default_factory=LimitesTermicos)
    memoria: LimitesMemoria = field(default_factory=LimitesMemoria)
    armazenamento: LimitesArmazenamento = field(default_factory=LimitesArmazenamento)
    batch: LimitesBatch = field(default_factory=LimitesBatch)
    monitoramento: LimitesMonitoramento = field(default_factory=LimitesMonitoramento)
    prioridades: PrioridadesTreino = field(default_factory=PrioridadesTreino)
    diretorio_lock: str = "logs/apolo_training.lock"
    arquivo_estado: str = "training_state.json"
    log_hardware: str = "logs/hardware.log"
    log_eventos: str = "logs/resource_events.jsonl"
    ativa: bool = True

    # ------------------------------------------------------------------ #
    def para_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validar(self) -> None:
        """Valida invariantes de segurança (falha rápido, não em produção)."""
        if self.batch.batch_minimo < 1:
            raise ValueError("batch_minimo precisa ser >= 1")
        if self.batch.batch_maximo < self.batch.batch_minimo:
            raise ValueError("batch_maximo deve ser >= batch_minimo")
        if not (self.batch.batch_minimo <= self.batch.batch_inicial <= self.batch.batch_maximo):
            raise ValueError("batch_inicial precisa estar entre batch_minimo e batch_maximo")
        if self.batch.seq_len_minima > self.batch.seq_len_maxima:
            raise ValueError("seq_len_minima precisa ser <= seq_len_maxima")
        if self.memoria.ram_minima_disponivel_mb <= 0:
            raise ValueError("ram_minima_disponivel_mb precisa ser positivo")
        if self.armazenamento.espaco_minimo_livre_mb < 0:
            raise ValueError("espaco_minimo_livre_mb não pode ser negativo")
        if self.termicos.temperatura_resume_c > self.termicos.temperatura_maxima_c:
            raise ValueError("temperatura_resume_c precisa ser <= temperatura_maxima_c")


def _aplicar(dc: Any, dados: Dict[str, Any], prefixo: str = "") -> Any:
    """Aplica um dicionário sobre uma dataclass, validando os campos."""
    validos = {f.name for f in fields(dc)}
    desconhecidos = set(dados) - validos
    if desconhecidos:
        raise ValueError(
            f"Chave(s) desconhecida(s) em '{prefixo or type(dc).__name__}': {sorted(desconhecidos)}"
        )
    for chave, valor in dados.items():
        atual = getattr(dc, chave)
        if isinstance(atual, (LimitesTermicos, LimitesMemoria, LimitesArmazenamento,
                              LimitesBatch, LimitesMonitoramento, PrioridadesTreino)):
            _aplicar(atual, valor or {}, f"{prefixo}.{chave}" if prefixo else chave)
        else:
            setattr(dc, chave, valor)
    return dc


def carregar_config_recursos(caminho: str | Path | None = None) -> ConfigRecursos:
    """Carrega ``configs/resource_manager.yaml`` (ou usa os padrões)."""
    config = ConfigRecursos()
    if caminho is None:
        return config
    caminho = Path(caminho)
    if not caminho.exists():
        return config
    dados = yaml.safe_load(caminho.read_text(encoding="utf-8")) or {}
    raiz = dados.get("recursos", dados)
    _aplicar(config, raiz)
    config.validar()
    return config
