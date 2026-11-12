"""Métricas de avaliação do Apolo Zenith 1.9.

Implementa as métricas usadas pelos benchmarks de linguagem, raciocínio e
programação.  Todas são calculadas sobre predicções reais do modelo — nunca
sobre valores simulados.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence


# --------------------------------------------------------------------------- #
# Linguagem
# --------------------------------------------------------------------------- #
def perplexity(perda: float) -> float:
    """Perplexidade a partir da perda de entropia cruzada média."""
    if perda <= 0:
        return 1.0
    return math.exp(min(perda, 30.0))


def bits_per_byte(perda: float, bytes_por_token: float = 4.0) -> float:
    """Bits por byte (métrica usada em compressão de texto)."""
    if bytes_por_token <= 0:
        raise ValueError("bytes_por_token precisa ser positivo")
    return (perda / (math.log(2) * bytes_por_token))


def acuracia_token(predicoes: Sequence[int], alvos: Sequence[int]) -> float:
    """Acurácia de tokens (argmax correto)."""
    if not predicoes:
        return 0.0
    corretos = sum(1 for p, a in zip(predicoes, alvos) if p == a)
    return corretos / len(predicoes)


# --------------------------------------------------------------------------- #
# Tarefas
# --------------------------------------------------------------------------- #
def normalizar_resposta(texto: str) -> str:
    """Normaliza a resposta para comparação (minúsculas, sem pontuação)."""
    texto = texto.strip().lower()
    texto = re.sub(r"[`\"'’‘]", "", texto)
    texto = re.sub(r"[.,;:!?]+$", "", texto)
    texto = re.sub(r"^\s*(a resposta é|resposta:|answer:)\s*", "", texto)
    return texto.strip()


def exact_match(predicao: str, referencia: str) -> float:
    """Correspondência exata após normalização."""
    return 1.0 if normalizar_resposta(predicao) == normalizar_resposta(referencia) else 0.0


def f1_token(predicao: str, referencia: str) -> float:
    """F1 sobre tokens (métrica de QA)."""
    p = normalizar_resposta(predicao).split()
    r = normalizar_resposta(referencia).split()
    if not p or not r:
        return float(p == r)
    comum = Counter(p) & Counter(r)
    n_comum = sum(comum.values())
    if n_comum == 0:
        return 0.0
    precisao = n_comum / len(p)
    recall = n_comum / len(r)
    return 2 * precisao * recall / (precisao + recall)


def extrair_primeiro_numero(texto: str) -> Optional[str]:
    """Extrai o primeiro número da resposta (para GSM8K/MATH)."""
    m = re.search(r"-?\d+(?:[.,]\d+)?", texto.replace(" ", ""))
    if not m:
        return None
    return m.group(0).replace(",", ".")


def acuracia_numerica(predicao: str, referencia: str) -> float:
    """Compara o primeiro número extraído das respostas."""
    a = extrair_primeiro_numero(predicao)
    b = extrair_primeiro_numero(referencia)
    if a is None or b is None:
        return exact_match(predicao, referencia)
    try:
        return 1.0 if abs(float(a) - float(b)) < 1e-6 else 0.0
    except ValueError:
        return 0.0


# --------------------------------------------------------------------------- #
# Programação (pass@k)
# --------------------------------------------------------------------------- #
def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimador unbiased de pass@k (Chen et al., 2021).

    ``n`` = número de amostras geradas, ``c`` = quantas passaram nos testes,
    ``k`` = quantas tentativas são consideradas.
    """
    if n <= 0 or k <= 0 or c < 0 or c > n:
        raise ValueError("parâmetros inválidos para pass@k")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / (n - i) for i in range(c))


def agregar_metricas(valores: Iterable[float]) -> Dict[str, float]:
    """Agrega uma lista de valores em média/desvio/mínimo/máximo."""
    lista = [v for v in valores if isinstance(v, (int, float))]
    if not lista:
        return {"media": 0.0, "n": 0}
    media = sum(lista) / len(lista)
    var = sum((v - media) ** 2 for v in lista) / len(lista)
    return {
        "media": round(media, 6),
        "desvio": round(math.sqrt(var), 6),
        "minimo": round(min(lista), 6),
        "maximo": round(max(lista), 6),
        "n": len(lista),
    }
