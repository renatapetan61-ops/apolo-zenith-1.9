"""Inferência local (mobile) do Apolo Zenith 1.9.

Responsabilidades:

* carregar modelos compactos no dispositivo;
* aplicar quantização INT8 (e INT4 quando o backend suportar);
* limitar threads e memória para respeitar os recursos do celular;
* reutilizar o KV Cache do Language Core para respostas rápidas.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from core.config import ModelConfig
from core.language_model import ApoloLanguageModel
from tokenizer.tokenizer import BPETokenizer


@dataclass
class ConfiguracaoMobile:
    """Parâmetros de execução no dispositivo móvel."""

    threads: int = 2
    batch_size: int = 1
    quantizacao: str = "int8"        # "off" | "int8" | "int4"
    max_new_tokens: int = 128
    temperatura: float = 0.8
    top_k: int = 40
    top_p: float = 0.92
    limite_ram_mb: int = 512
    usar_cache: bool = True


def limitar_threads(n: int = 2) -> None:
    """Limita o número de threads do PyTorch (economia de energia no celular)."""
    n = max(1, int(n))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # já foi inicializado em outro ponto do processo


def quantizar_int8(model: torch.nn.Module) -> tuple[torch.nn.Module, bool]:
    """Aplica quantização dinâmica INT8 nas camadas lineares.

    Retorna ``(modelo, sucesso)``.  Em arquiteturas sem backend de quantização
    (comum em ARM sem FBGEMM), o modelo original é devolvido intacto e a
    informação é registrada — nunca se finge que a quantização ocorreu.
    """
    try:
        modelo_q = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return modelo_q, True
    except Exception as exc:  # pragma: no cover - depende do backend
        print(f"[mobile] quantização INT8 indisponível ({exc}); usando fp32.")
        return model, False


def quantizar_int4(model: torch.nn.Module) -> tuple[torch.nn.Module, bool]:
    """Tentativa de quantização INT4.

    Em PyTorch, INT4 depende de backends experimentais (``torchao``/bitsandbytes)
    e de suporte do hardware; quando indisponível, o modelo é mantido em fp32.
    """
    try:  # pragma: no cover - depende do ambiente
        from torchao.quantization import quantize_, int4_weight_only

        return quantize_(model, int4_weight_only()), True
    except Exception as exc:
        print(f"[mobile] quantização INT4 indisponível ({exc}); usando fp32/INT8.")
        return model, False


class MotorMobile:
    """Motor de inferência local para o cliente mobile."""

    def __init__(self, checkpoint: str | Path, config: ConfiguracaoMobile | None = None) -> None:
        self.config = config or ConfiguracaoMobile()
        limitar_threads(self.config.threads)
        self.checkpoint = Path(checkpoint)
        self.modelo = self._carregar()
        self.tokenizer = self._carregar_tokenizer()
        self.quantizado = False

    # ------------------------------------------------------------------ #
    def _carregar(self) -> ApoloLanguageModel:
        modelo = ApoloLanguageModel.load(self.checkpoint, map_location="cpu")
        modelo.eval()
        if self.config.quantizacao == "int8":
            modelo, ok = quantizar_int8(modelo)
            self.quantizado = ok
        elif self.config.quantizacao == "int4":
            modelo, ok = quantizar_int4(modelo)
            self.quantizado = ok
        return modelo

    def _carregar_tokenizer(self) -> BPETokenizer:
        pasta = self.checkpoint / "tokenizer"
        if pasta.exists():
            return BPETokenizer.load(pasta)
        raise FileNotFoundError(
            f"Tokenizer não encontrado em {pasta}. O checkpoint deve incluir o tokenizer."
        )

    # ------------------------------------------------------------------ #
    def gerar(self, prompt: str, max_new_tokens: Optional[int] = None, **kwargs) -> Dict:
        """Gera texto a partir de um prompt e devolve texto + métricas."""
        gc.collect()
        max_new = max_new_tokens or self.config.max_new_tokens
        ids = self.tokenizer.encode(prompt)
        if not ids:
            return {"texto": "", "tokens_gerados": 0, "tokens_s": 0.0}
        entrada = torch.tensor([ids[-self.modelo.config.block_size :]], dtype=torch.long)
        gerador = torch.Generator().manual_seed(int(kwargs.pop("seed", 42)))
        inicio = time.time()
        with torch.no_grad():
            saida = self.modelo.generate(
                entrada,
                max_new_tokens=max_new,
                temperature=kwargs.pop("temperature", self.config.temperatura),
                top_k=kwargs.pop("top_k", self.config.top_k),
                top_p=kwargs.pop("top_p", self.config.top_p),
                repetition_penalty=kwargs.pop("repetition_penalty", 1.05),
                stop_token_ids=[self.tokenizer.eos_id] if self.tokenizer.eos_id else None,
                generator=gerador,
                use_cache=kwargs.pop("use_cache", self.config.usar_cache),
            )
        decorrido = max(time.time() - inicio, 1e-6)
        texto = self.tokenizer.decode(saida[0].tolist())
        return {
            "texto": texto,
            "tokens_gerados": int(saida.shape[1]),
            "tempo_s": round(decorrido, 3),
            "tokens_s": round(saida.shape[1] / decorrido, 2),
            "quantizado": self.quantizado,
        }

    def memoria_estimada_mb(self) -> float:
        """Estimativa aproximada de RAM ocupada pelo modelo carregado."""
        total = sum(p.numel() * p.element_size() for p in self.modelo.parameters())
        return round(total / (1024**2), 2)
