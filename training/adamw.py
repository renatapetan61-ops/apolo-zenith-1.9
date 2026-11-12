"""Otimizador AdamW do Apolo (decoupled weight decay).

Implementação própria, baseada em *AdamW decoupled* (Loshchilov & Hutter,
2019), em que o decaimento de pesos é aplicado **diretamente** aos parâmetros,
e não através do gradiente:

    p <- p - lr * (adam_step + weight_decay * p)

Isso é diferente do Adam+L2 (weight decay somado ao gradiente) e é o que
garante desempenho estável em LLMs.

A implementação é numericamente equivalente ao ``torch.optim.AdamW`` e a
equivalência é verificada em ``tests/test_optimizer.py``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """AdamW com decaimento desacoplado, bias correction e modo AMSGrad opcional."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        amsgrad: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"lr inválido: {lr}")
        if eps < 0.0:
            raise ValueError(f"eps inválido: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"beta1 inválido: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"beta2 inválido: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):  # type: ignore[override]
        """Executa um passo de otimização."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW não suporta gradientes esparsos")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                state["step"] += 1
                t = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Atualiza os momentos em precisão fp32 (estabilidade em fp16/bf16).
                g = grad.float()
                exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                if amsgrad:
                    max_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_sq, exp_avg_sq, out=max_sq)
                    denom_base = max_sq
                else:
                    denom_base = exp_avg_sq

                # Decaimento de pesos desacoplado (não passa pelo gradiente).
                # Aplicado antes do passo Adam, como em torch.optim.AdamW.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t

                denom = (denom_base.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg.to(p.dtype), denom.to(p.dtype), value=-step_size)

        return loss


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
    use_torch: bool = False,
) -> Optimizer:
    """Cria o otimizador AdamW com grupos de parâmetros corretos.

    Matrizes recebem decaimento; normalizações, biases e embeddings não
    recebem (prática padrão no treinamento de LLMs).
    """
    if hasattr(model, "parameter_groups_for_decay"):
        grupos = model.parameter_groups_for_decay(weight_decay)
    else:
        decay, sem_decay = [], []
        for nome, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2 and not nome.endswith(".bias"):
                decay.append(param)
            else:
                sem_decay.append(param)
        grupos = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": sem_decay, "weight_decay": 0.0},
        ]
    if use_torch:
        return torch.optim.AdamW(grupos, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return AdamW(grupos, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
