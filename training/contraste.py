"""Objetivo contrastivo (InfoNCE) e avaliação de recuperação.

Usado pelos trainers de visão, áudio e vídeo para alinhar as modalidades
com os embeddings do Language Core.

A loss InfoNCE é real e padronizada:

.. math::

    L = -\\frac{1}{B}\\sum_i \\log \\frac{e^{s(x_i, y_i)/\\tau}}
                                       {\\sum_j e^{s(x_i, y_j)/\\tau}}

com similaridade de cosseno e temperatura ``tau``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def normalizar(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normaliza os embeddings em L2 (para similaridade de cosseno)."""
    return F.normalize(x, dim=-1, eps=eps)


def centralizar(x: torch.Tensor) -> torch.Tensor:
    """Remove o componente comum (anisotropia) de um conjunto de embeddings.

    Embeddings de LM têm uma direção dominante (similaridade de cosseno
    média ≈ 0,87), o que torna a loss contrastiva quase sem gradiente.
    Centralizar (subtrair a média do lote) preserva a informação
    diferencial e é o mesmo truque usado em literatura de embeddings.
    """
    if x.shape[0] < 2:
        return x
    return x - x.mean(dim=0, keepdim=True)


def perda_contrastiva(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    temperatura: float = 0.07,
    centralizar_embeddings: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Loss InfoNCE simétrica entre dois conjuntos de embeddings.

    Devolve ``(perda, matriz_de_similaridade)``.
    """
    a = normalizar(embeddings_a)
    b = normalizar(embeddings_b)
    if centralizar_embeddings:
        a = normalizar(centralizar(a))
        b = normalizar(centralizar(b))
    similaridade = a @ b.t() / temperatura
    alvo = torch.arange(a.shape[0], device=a.device)
    perda_a = F.cross_entropy(similaridade, alvo)
    perda_b = F.cross_entropy(similaridade.t(), alvo)
    return (perda_a + perda_b) / 2.0, similaridade


@torch.no_grad()
def acuracia_recuperacao(similaridade: torch.Tensor) -> Dict[str, float]:
    """Métricas de recuperação (image→texto e texto→image)."""
    alvo = torch.arange(similaridade.shape[0], device=similaridade.device)
    i2t = (similaridade.argmax(dim=1) == alvo).float().mean().item()
    t2i = (similaridade.argmax(dim=0) == alvo).float().mean().item()
    return {
        "recuperacao_para_texto": round(i2t, 4),
        "recuperacao_para_imagem": round(t2i, 4),
        "r1": round((i2t + t2i) / 2.0, 4),
        "mrr": round(_mrr(similaridade, alvo), 4),
    }


def _mrr(similaridade: torch.Tensor, alvo: torch.Tensor) -> float:
    """Reciprocal Mean Rank médio (nas duas direções)."""
    ordem = similaridade.argsort(dim=1, descending=True)
    pos = (ordem == alvo.unsqueeze(1)).float().argmax(dim=1) + 1
    i2t = (1.0 / pos.float()).mean().item()
    ordem_t = similaridade.t().argsort(dim=1, descending=True)
    pos_t = (ordem_t == alvo.unsqueeze(1)).float().argmax(dim=1) + 1
    t2i = (1.0 / pos_t.float()).mean().item()
    return (i2t + t2i) / 2.0


@torch.no_grad()
def avaliar_recuperacao(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    temperatura: float = 0.07,
    lote: int = 128,
    centralizar_embeddings: bool = True,
) -> Dict[str, float]:
    """Avalia a recuperação em blocos (sem estourar a RAM)."""
    total = embeddings_a.shape[0]
    acertos_i2t = 0
    acertos_t2i = 0
    ranks_i2t: List[float] = []
    for inicio in range(0, total, lote):
        fim = min(inicio + lote, total)
        a = normalizar(embeddings_a[inicio:fim])
        b = normalizar(embeddings_b[inicio:fim])
        if centralizar_embeddings:
            a = normalizar(centralizar(a))
            b = normalizar(centralizar(b))
        similaridade = a @ b.t() / temperatura
        alvo = torch.arange(fim - inicio, device=a.device)
        acertos_i2t += int((similaridade.argmax(dim=1) == alvo).sum().item())
        acertos_t2i += int((similaridade.argmax(dim=0) == alvo).sum().item())
        ordem = similaridade.argsort(dim=1, descending=True)
        pos = (ordem == alvo.unsqueeze(1)).float().argmax(dim=1) + 1
        ranks_i2t.extend((1.0 / pos.float()).tolist())
    chance = 1.0 / max(total, 1)
    return {
        "recuperacao_para_texto": round(acertos_i2t / max(total, 1), 4),
        "recuperacao_para_imagem": round(acertos_t2i / max(total, 1), 4),
        "r1": round((acertos_i2t + acertos_t2i) / (2 * max(total, 1)), 4),
        "mrr": round(sum(ranks_i2t) / max(len(ranks_i2t), 1), 4),
        "chance": round(chance, 4),
        "amostras": total,
    }
