"""Cliente móvel do Apolo Zenith 1.9.

O celular funciona como **painel de controle e cliente de inferência**:
envia comandos ao servidor externo de treinamento, acompanha métricas, gerencia
datasets/checkpoints e executa inferência local com modelos compactados.

Nenhum treinamento pesado é executado no dispositivo — este módulo apenas
fala HTTP com o servidor (usando apenas a biblioteca padrão).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Resposta:
    """Resposta padronizada da API."""

    ok: bool
    dados: Any
    erro: Optional[str] = None
    status: int = 0


class ClienteServidorApolo:
    """Cliente HTTP do servidor de treinamento."""

    def __init__(self, host: str = "127.0.0.1", porta: int = 8080, timeout: int = 30) -> None:
        self.base = f"http://{host}:{porta}/api"
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    def _requisicao(self, metodo: str, rota: str, corpo: Optional[Dict] = None) -> Resposta:
        url = f"{self.base}{rota}"
        dados = json.dumps(corpo).encode("utf-8") if corpo is not None else None
        req = urllib.request.Request(url, data=dados, method=metodo)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resposta:
                payload = resposta.read().decode("utf-8")
                try:
                    return Resposta(True, json.loads(payload), status=resposta.status)
                except json.JSONDecodeError:
                    return Resposta(True, payload, status=resposta.status)
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            try:
                detalhe = json.loads(payload)
            except json.JSONDecodeError:
                detalhe = payload
            return Resposta(False, detalhe, erro=f"HTTP {exc.code}", status=exc.code)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return Resposta(False, None, erro=str(exc))

    # ------------------------------------------------------------------ #
    # Operações
    # ------------------------------------------------------------------ #
    def health(self) -> Resposta:
        return self._requisicao("GET", "/health")

    def hardware(self) -> Resposta:
        return self._requisicao("GET", "/hardware")

    def metricas(self, arquivo: str = "checkpoints/apolo-lm-tiny/metricas.jsonl", n: int = 50) -> Resposta:
        return self._requisicao("GET", f"/metrics?arquivo={arquivo}&n={n}")

    def modelos(self) -> Resposta:
        return self._requisicao("GET", "/models")

    def jobs(self) -> Resposta:
        return self._requisicao("GET", "/jobs")

    def iniciar_job(self, comando: List[str], nome: str = "treino") -> Resposta:
        return self._requisicao("POST", "/jobs", {"comando": comando, "nome": nome})

    def status_job(self, job_id: str) -> Resposta:
        return self._requisicao("GET", f"/jobs/{job_id}")

    def cancelar_job(self, job_id: str) -> Resposta:
        return self._requisicao("POST", f"/jobs/{job_id}/cancelar")

    def inferir(
        self,
        checkpoint: str,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.8,
        seed: int = 42,
    ) -> Resposta:
        return self._requisicao(
            "POST",
            "/inferir",
            {
                "checkpoint": checkpoint,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed,
            },
        )
