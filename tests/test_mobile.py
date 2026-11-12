"""Testes do cliente mobile e da inferência local quantizada."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest
import torch

from core.config import ModelConfig
from core.language_model import ApoloLanguageModel
from mobile.inference import ConfiguracaoMobile, MotorMobile, limitar_threads, quantizar_int8
from mobile.mobile_client import ClienteServidorApolo
from server.api import criar_servidor
from tokenizer.tokenizer import BPETokenizer


@pytest.fixture(scope="module")
def checkpoint_pequeno(tmp_path_factory) -> Path:
    """Cria um checkpoint mínimo (modelo + tokenizer) para testes."""
    pasta = tmp_path_factory.mktemp("ckpt_mobile")
    tokenizer = BPETokenizer.train(
        ["Era uma vez um rei.", "O rato roeu a roupa do rei de Roma.", "Hello world."] * 20,
        vocab_size=320,
        min_frequencia=1,
    )
    tokenizer.save(pasta / "tokenizer")
    modelo = ApoloLanguageModel(
        ModelConfig(vocab_size=tokenizer.vocab_size, n_layer=2, n_head=2, n_embd=32,
                    block_size=32, mlp_hidden=64)
    )
    modelo.save(pasta)
    return pasta


def test_limitacao_de_threads() -> None:
    limitar_threads(2)
    assert torch.get_num_threads() == 2
    limitar_threads(1)


def test_quantizacao_int8_ou_fallback_transparente(checkpoint_pequeno: Path) -> None:
    """A quantização deve retornar um modelo usável, mesmo sem backend."""
    modelo = ApoloLanguageModel.load(checkpoint_pequeno, map_location="cpu")
    modelo.eval()
    entrada = torch.randint(0, modelo.config.vocab_size, (1, 8))
    with torch.no_grad():
        logits_original, _ = modelo(entrada)

    quantizado, ok = quantizar_int8(modelo)
    assert isinstance(quantizado, torch.nn.Module)
    with torch.no_grad():
        logits_quantizado, _ = quantizado(entrada)
    assert logits_quantizado.shape == (1, 8, modelo.config.vocab_size)
    if ok:
        # Na quantização dinâmica os pesos ficam em buffers empacotados,
        # não em Parameters; verificamos os módulos e a equivalência numérica.
        tem_modulo_quantizado = any(
            "quantized" in type(m).__module__ for m in quantizado.modules()
        )
        assert tem_modulo_quantizado
        assert torch.allclose(logits_quantizado, logits_original, atol=0.25)


def test_motor_mobile_gera_texto(checkpoint_pequeno: Path) -> None:
    motor = MotorMobile(checkpoint_pequeno, ConfiguracaoMobile(threads=1, quantizacao="off", max_new_tokens=8))
    resultado = motor.gerar("Era uma vez", max_new_tokens=8)
    assert "texto" in resultado
    assert resultado["tokens_gerados"] <= 8
    assert motor.memoria_estimada_mb() > 0


def test_cliente_mobile_comunica_com_servidor(tmp_path: Path) -> None:
    raiz = Path(__file__).resolve().parents[1]
    srv = criar_servidor("127.0.0.1", 0, raiz=raiz)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    porta = srv.server_address[1]
    try:
        cliente = ClienteServidorApolo(host="127.0.0.1", porta=porta, timeout=20)
        resposta = cliente.health()
        assert resposta.ok
        assert resposta.dados["status"] == "ok"

        hardware = cliente.hardware()
        assert hardware.ok
        assert "cpu" in hardware.dados

        metricas = cliente.metricas()
        assert metricas.ok

        modelos = cliente.modelos()
        assert modelos.ok and "checkpoints" in modelos.dados
    finally:
        srv.shutdown()
        srv.server_close()


def test_cliente_mobile_reporta_servidor_inexistente() -> None:
    cliente = ClienteServidorApolo(host="127.0.0.1", porta=9, timeout=2)
    resposta = cliente.health()
    assert resposta.ok is False
    assert resposta.erro
