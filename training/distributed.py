"""Treinamento distribuído do Apolo (DDP / FSDP) e detecção de estratégia.

O módulo **não assume** que o servidor externo tem GPUs ou suporte a
distribuído.  A função :func:`detectar_estrategia` inspeciona o hardware real
e só recomenda DDP/FSDP quando há GPUs detectadas e elas suportam o
comunicador necessário.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist


@dataclass
class EstrategiaDistribuida:
    """Estratégia de paralelismo escolhida com base no hardware detectado."""

    modo: str                    # "single" | "ddp" | "fsdp" | "cpu"
    world_size: int
    num_gpus: int
    backend: str
    observacoes: list[str]

    def para_dict(self) -> Dict[str, Any]:
        return {
            "modo": self.modo,
            "world_size": self.world_size,
            "num_gpus": self.num_gpus,
            "backend": self.backend,
            "observacoes": list(self.observacoes),
        }


def ambiente_distribuido() -> bool:
    """Indica se as variáveis de ambiente de execução paralela estão presentes."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def rank_atual() -> int:
    return int(os.environ.get("RANK", 0))


def world_size_atual() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def local_rank_atual() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def inicializar_distribuido(backend: Optional[str] = None) -> bool:
    """Inicializa ``torch.distributed`` se o ambiente permitir.

    Retorna ``True`` quando o processo realmente entrou em modo distribuído.
    """
    if not ambiente_distribuido():
        return False
    if dist.is_initialized():
        return True
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank_atual())
    return True


def envolver_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Envolve o modelo em ``DistributedDataParallel``."""
    if not dist.is_initialized():
        raise RuntimeError("process group não inicializado")
    device_ids = [local_rank_atual()] if device.type == "cuda" else None
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=device_ids, output_device=local_rank_atual() if device.type == "cuda" else None
    )


def envolver_fsdp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Envolve o modelo em FSDP (sharding de parâmetros, gradientes e estados).

    Requer PyTorch >= 2.0 e CUDA para o modo ``FULL_SHARD``; caso contrário
    levanta erro explícito em vez de degradar silenciosamente.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    if not dist.is_initialized():
        raise RuntimeError("process group não inicializado")
    if device.type != "cuda":
        raise RuntimeError("FSDP com estratégia FULL_SHARD requer GPUs CUDA")
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank_atual(),
        use_orig_params=True,
    )


def detectar_estrategia(num_gpus: Optional[int] = None, vram_gb: Optional[float] = None) -> EstrategiaDistribuida:
    """Escolhe a estratégia de treino compatível com o hardware disponível."""
    observacoes: list[str] = []
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    if vram_gb is None and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)

    if num_gpus == 0:
        observacoes.append("Nenhuma GPU detectada: treino em CPU (otimizador AdamW em fp32).")
        return EstrategiaDistribuida("cpu", 1, 0, "gloo", observacoes)

    if num_gpus == 1:
        observacoes.append("Uma GPU detectada: treino single-GPU com Mixed Precision (bf16/fp16).")
        return EstrategiaDistribuida("single", 1, num_gpus, "nccl", observacoes)

    observacoes.append(f"{num_gpus} GPUs detectadas.")
    if vram_gb is not None and vram_gb < 24:
        observacoes.append(
            f"VRAM por GPU ({vram_gb:.1f} GB) limitada: preferir DDP com grad accumulation "
            "e ativação checkpointing em vez de FSDP."
        )
        return EstrategiaDistribuida("ddp", num_gpus, num_gpus, "nccl", observacoes)

    observacoes.append("VRAM suficiente: FSDP recomendado para modelos grandes.")
    return EstrategiaDistribuida("fsdp", num_gpus, num_gpus, "nccl", observacoes)


def media_global(valor: torch.Tensor) -> torch.Tensor:
    """Média de um tensor entre todos os processos do grupo."""
    if not dist.is_initialized():
        return valor
    tensor = valor.detach().clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / dist.get_world_size()


def barreira() -> None:
    """Sincroniza todos os processos (usado antes de salvar checkpoints)."""
    if dist.is_initialized():
        dist.barrier()
