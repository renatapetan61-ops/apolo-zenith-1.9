"""Estimativas de custo de treinamento do Apolo Zenith 1.9.

Converte a configuração do experimento em estimativas de tempo, memória e
tamanho de checkpoint.  O tempo só é estimado quando existe **medição real**
de vazão (tokens/s) — caso contrário, o sistema informa que a estimativa
indisponível, em vez de inventar um número.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from core.config import ModelConfig
from server.hardware_detection import estimar_memoria_treino
from training.trainer import TrainingConfig


@dataclass
class EstimativaTreino:
    """Estimativa de custo de um experimento de treino."""

    parametros: int
    tokens_por_passo: int
    passos: int
    tokens_total: int
    tempo_estimado_s: Optional[float]
    tokens_por_segundo: Optional[float]
    memoria: Dict[str, float]
    observacoes: list[str]

    def para_dict(self) -> Dict:
        return {
            "parametros": self.parametros,
            "tokens_por_passo": self.tokens_por_passo,
            "passos": self.passos,
            "tokens_total": self.tokens_total,
            "tempo_estimado_s": self.tempo_estimado_s,
            "tempo_estimado_h": round(self.tempo_estimado_s / 3600, 3) if self.tempo_estimado_s else None,
            "tokens_por_segundo": self.tokens_por_segundo,
            "memoria": self.memoria,
            "observacoes": self.observacoes,
        }


def estimar_treino(
    modelo: ModelConfig,
    treino: TrainingConfig,
    tokens_por_segundo: Optional[float] = None,
    bytes_por_token_ativacao: int = 0,
) -> EstimativaTreino:
    """Estima o custo de um experimento.

    ``tokens_por_segundo`` deve vir de uma **medição real** (o Trainer
    registra ``tokens_s`` no log JSONL).  Sem medição, ``tempo_estimado_s``
    é ``None``.
    """
    parametros = modelo.n_parameters
    tokens_por_passo = treino.batch_size * treino.grad_accum_steps * treino.block_size
    tokens_total = tokens_por_passo * treino.max_steps

    observacoes: list[str] = []
    tempo = None
    if tokens_por_segundo and tokens_por_segundo > 0:
        tempo = tokens_total / tokens_por_segundo
        observacoes.append(
            f"Estimativa baseada em {tokens_por_segundo:.0f} tokens/s medidos."
        )
    else:
        observacoes.append(
            "Sem medição de vazão: o tempo não pode ser estimado honestamente. "
            "Rode alguns passos do Trainer e leia 'tokens_s' em logs/*.jsonl."
        )

    memoria = estimar_memoria_treino(
        parametros,
        ativa_bytes_por_token=bytes_por_token_ativacao,
        tokens_por_passo=tokens_por_passo,
    )
    if parametros < 10_000_000:
        observacoes.append(
            "Modelo pequeno (<10M params): apropriado para protótipo em CPU; "
            "paraxis de desempenho real, use o servidor externo com GPUs."
        )
    if memoria["total_gb"] > 24:
        observacoes.append(
            "Memória estimada acima de 24 GB: prefira FSDP, gradient accumulation "
            "e ativação checkpointing."
        )
    observacoes.append(
        f"Epochs equivalentes sobre um corpus de 12,76M tokens: "
        f"{tokens_total / 12_759_787:.2f}"
    )

    return EstimativaTreino(
        parametros=parametros,
        tokens_por_passo=tokens_por_passo,
        passos=treino.max_steps,
        tokens_total=tokens_total,
        tempo_estimado_s=tempo,
        tokens_por_segundo=tokens_por_segundo,
        memoria=memoria,
        observacoes=observacoes,
    )
