"""Registro de métricas do Apolo (JSONL + histórico em memória).

Formato: um JSON por linha, com timestamp, o que permite acompanhar o
progresso com ``tail -f`` no servidor ou diretamente pelo cliente mobile.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetricsTracker:
    """Acumula e persiste métricas de treino/validação."""

    def __init__(self, caminho: Optional[str | Path] = None, manter_historico: int = 10_000) -> None:
        self.caminho = Path(caminho) if caminho else None
        if self.caminho is not None:
            self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.historico: List[Dict[str, Any]] = []
        self.limite = manter_historico
        self._inicio = time.time()

    def registrar(self, **metricas: Any) -> Dict[str, Any]:
        """Registra uma linha de métricas e a persiste em JSONL."""
        registro: Dict[str, Any] = {"timestamp": time.time(), "elapsed_s": round(time.time() - self._inicio, 3)}
        for chave, valor in metricas.items():
            if isinstance(valor, float) and (math.isnan(valor) or math.isinf(valor)):
                continue
            registro[chave] = valor
        self.historico.append(registro)
        if len(self.historico) > self.limite:
            self.historico = self.historico[-self.limite :]
        if self.caminho is not None:
            with self.caminho.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(registro, ensure_ascii=False) + "\n")
        return registro

    def ultimas(self, n: int = 10) -> List[Dict[str, Any]]:
        return self.historico[-n:]

    def melhor(self, chave: str, modo: str = "min") -> Optional[float]:
        valores = [r[chave] for r in self.historico if chave in r and isinstance(r[chave], (int, float))]
        if not valores:
            return None
        return min(valores) if modo == "min" else max(valores)


def perplexity(perda: float) -> float:
    """Converte perda de entropia cruzada em perplexidade."""
    try:
        return math.exp(min(perda, 30.0))
    except OverflowError:  # pragma: no cover - proteção numérica
        return float("inf")
