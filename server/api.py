"""API HTTP do servidor de treinamento do Apolo Zenith 1.9.

Implementada com a biblioteca padrão do Python (``http.server``) para não
impor dependências no servidor mínimo.  O cliente mobile e a CLI de
administração speaks o mesmo protocolo JSON.

Endpoints
---------
``GET  /api/health``                – estado do serviço
``GET  /api/hardware``              – relatório de hardware detectado
``GET  /api/metrics``               – métricas do sistema + últimos registros de treino
``GET  /api/models``                – checkpoints disponíveis
``GET  /api/jobs``                  – lista de jobs de treino
``POST /api/jobs``                  – inicia um job de treino
``GET  /api/jobs/<id>``             – detalhe + log de um job
``POST /api/jobs/<id>/cancelar``    – cancela um job
``POST /api/inferir``               – inferência de texto em um checkpoint
"""

from __future__ import annotations

import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from server.hardware_detection import coletar_relatorio, recomendar_estrategia
from server.job_manager import JobManager
from server.monitoring import ler_metricas, listar_checkpoints, metricas_sistema


class EstadoServidor:
    """Estado compartilhado entre as threads do servidor."""

    def __init__(self, raiz: str | Path = ".", usar_deteccao: bool = True) -> None:
        self.raiz = Path(raiz).resolve()
        self.jobs = JobManager(
            self.raiz / "logs" / "jobs",
            caminho_lock=self.raiz / "logs" / "jobs" / "apolo_training.lock",
        )
        self.modelos: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._relatorio = None
        if usar_deteccao:
            try:
                self._relatorio = coletar_relatorio()
            except Exception:  # pragma: no cover - detecção é opcional
                self._relatorio = None

    def relatorio(self, recarregar: bool = False):
        if recarregar or self._relatorio is None:
            self._relatorio = coletar_relatorio()
        return self._relatorio

    def carregar_modelo(self, checkpoint: str):
        """Carrega (com cache) um checkpoint para inferência."""
        from core.language_model import ApoloLanguageModel

        with self._lock:
            if checkpoint in self.modelos:
                return self.modelos[checkpoint]
        caminho = Path(checkpoint)
        if not caminho.is_absolute():
            caminho = self.raiz / caminho
        modelo = ApoloLanguageModel.load(caminho, map_location="cpu")
        modelo.eval()
        with self._lock:
            self.modelos[checkpoint] = modelo
        return modelo

    def carregar_tokenizer(self, checkpoint: str):
        from tokenizer.tokenizer import BPETokenizer

        with self._lock:
            if checkpoint in self.tokenizers:
                return self.tokenizers[checkpoint]
        caminho = Path(checkpoint)
        if not caminho.is_absolute():
            caminho = self.raiz / caminho
        tokenizer = BPETokenizer.load(caminho / "tokenizer")
        with self._lock:
            self.tokenizers[checkpoint] = tokenizer
        return tokenizer


class ManipuladorApolo(BaseHTTPRequestHandler):
    """Rotas HTTP do servidor Apolo."""

    estado: EstadoServidor = None  # type: ignore[assignment]
    versao: str = "1.9"

    # ------------------------------------------------------------------ #
    # Utilidades
    # ------------------------------------------------------------------ #
    def _responder(self, codigo: int, dados: Any) -> None:
        corpo = json.dumps(dados, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def _corpo_json(self) -> Dict:
        tamanho = int(self.headers.get("Content-Length", 0))
        if tamanho <= 0:
            return {}
        bruto = self.rfile.read(tamanho)
        try:
            return json.loads(bruto.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def log_message(self, formato: str, *args) -> None:  # pragma: no cover - log do servidor
        print(f"[api] {self.address_string()} {formato % args}")

    # ------------------------------------------------------------------ #
    # Rotas
    # ------------------------------------------------------------------ #
    def do_GET(self) -> None:  # noqa: N802 (nome exigido pelo http.server)
        rota = urlparse(self.path)
        caminho = rota.path.rstrip("/") or "/"
        query = parse_qs(rota.query)
        try:
            if caminho in ("/", "/api", "/api/health"):
                self._responder(200, {"status": "ok", "servico": "apolo-zenith", "versao": self.versao})
            elif caminho == "/api/hardware":
                relatorio = self.estado.relatorio()
                dados = relatorio.para_dict()
                dados["recomendacao"] = recomendar_estrategia(relatorio, 2_000_000)
                self._responder(200, dados)
            elif caminho == "/api/metrics":
                padrao = "checkpoints/apolo-lm-tiny/metricas.jsonl"
                arquivo = query.get("arquivo", [padrao])[0]
                self._responder(
                    200,
                    {
                        "sistema": metricas_sistema(),
                        "treino": ler_metricas(self.estado.raiz / arquivo, int(query.get("n", ["50"])[0])),
                    },
                )
            elif caminho == "/api/models":
                self._responder(200, {"checkpoints": listar_checkpoints(self.estado.raiz / "checkpoints")})
            elif caminho == "/api/jobs":
                self._responder(200, {"jobs": self.estado.jobs.listar()})
            elif caminho.startswith("/api/jobs/"):
                job_id = caminho.split("/")[3]
                job = self.estado.jobs.obter(job_id)
                if job is None:
                    self._responder(404, {"erro": "job não encontrado"})
                else:
                    self._responder(200, {"job": job.para_dict(), "log": self.estado.jobs.logs(job_id, 200)})
            else:
                self._responder(404, {"erro": f"rota desconhecida: {caminho}"})
        except Exception as exc:  # pragma: no cover - proteção do servidor
            self._responder(500, {"erro": str(exc), "traceback": traceback.format_exc()})

    def do_POST(self) -> None:  # noqa: N802
        rota = urlparse(self.path)
        caminho = rota.path.rstrip("/") or "/"
        corpo = self._corpo_json()
        try:
            if caminho == "/api/jobs":
                comando = corpo.get("comando")
                nome = corpo.get("nome", "treino")
                if not comando:
                    self._responder(400, {"erro": "campo 'comando' é obrigatório"})
                    return
                job = self.estado.jobs.iniciar(nome, comando, cwd=str(self.estado.raiz))
                self._responder(201, {"job": job.para_dict()})
            elif caminho.startswith("/api/jobs/") and caminho.endswith("/cancelar"):
                job_id = caminho.split("/")[3]
                ok = self.estado.jobs.cancelar(job_id)
                self._responder(200 if ok else 404, {"cancelado": ok})
            elif caminho == "/api/inferir":
                self._inferir(corpo)
            else:
                self._responder(404, {"erro": f"rota desconhecida: {caminho}"})
        except Exception as exc:  # pragma: no cover
            self._responder(500, {"erro": str(exc), "traceback": traceback.format_exc()})

    def _inferir(self, corpo: Dict) -> None:
        """Executa inferência de texto em um checkpoint carregado."""
        import torch

        checkpoint = corpo.get("checkpoint")
        prompt = corpo.get("prompt", "")
        if not checkpoint:
            self._responder(400, {"erro": "campo 'checkpoint' é obrigatório"})
            return
        modelo = self.estado.carregar_modelo(checkpoint)
        tokenizer = self.estado.carregar_tokenizer(checkpoint)

        ids = tokenizer.encode(prompt)
        if not ids:
            self._responder(400, {"erro": "prompt vazio"})
            return
        entrada = torch.tensor([ids[-modelo.config.block_size :]], dtype=torch.long)
        gerador = torch.Generator().manual_seed(int(corpo.get("seed", 42)))
        saida = modelo.generate(
            entrada,
            max_new_tokens=int(corpo.get("max_tokens", 128)),
            temperature=float(corpo.get("temperature", 0.8)),
            top_k=int(corpo.get("top_k", 50)),
            top_p=float(corpo.get("top_p", 0.95)),
            repetition_penalty=float(corpo.get("repetition_penalty", 1.05)),
            stop_token_ids=[tokenizer.eos_id] if tokenizer.eos_id is not None else None,
            generator=gerador,
        )
        texto = tokenizer.decode(saida[0].tolist())
        self._responder(
            200,
            {
                "checkpoint": checkpoint,
                "prompt": prompt,
                "continuacao": texto,
                "tokens_gerados": int(saida.shape[1]),
            },
        )


def criar_servidor(host: str = "0.0.0.0", porta: int = 8080, raiz: str | Path = ".") -> ThreadingHTTPServer:
    """Cria (sem iniciar) o servidor HTTP do Apolo."""
    ManipuladorApolo.estado = EstadoServidor(raiz=raiz)
    servidor = ThreadingHTTPServer((host, porta), ManipuladorApolo)
    return servidor


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Servidor de treinamento e controle do Apolo Zenith 1.9")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--porta", type=int, default=8080)
    parser.add_argument("--raiz", default=".")
    args = parser.parse_args()

    servidor = criar_servidor(args.host, args.porta, args.raiz)
    print(f"Apolo Zenith 1.9 — API do servidor em http://{args.host}:{args.porta}/api/health")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando servidor…")
    finally:
        servidor.server_close()


if __name__ == "__main__":
    main()
