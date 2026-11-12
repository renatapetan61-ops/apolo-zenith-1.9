"""Testes da arquitetura do Apolo Language Core."""

from __future__ import annotations

import pytest
import torch

from core.config import ModelConfig
from core.language_model import ApoloLanguageModel


@pytest.fixture()
def config_pequena() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_embd=32,
        block_size=32,
        mlp_hidden=64,
    )


@pytest.fixture()
def modelo(config_pequena: ModelConfig) -> ApoloLanguageModel:
    torch.manual_seed(42)
    m = ApoloLanguageModel(config_pequena)
    m.eval()
    return m


def test_forward_shapes(modelo: ApoloLanguageModel) -> None:
    idx = torch.randint(0, modelo.config.vocab_size, (2, 8))
    logits, loss = modelo(idx, idx)
    assert logits.shape == (2, 8, modelo.config.vocab_size)
    assert loss is not None and loss.item() > 0


def test_perda_calculada_em_cada_posicao(modelo: ApoloLanguageModel) -> None:
    """Alvos deslocados devem produzir a entropia cruzada esperada."""
    idx = torch.randint(0, modelo.config.vocab_size, (1, 6))
    logits, loss = modelo(idx, idx)
    manual = torch.nn.functional.cross_entropy(logits[0].float(), idx[0])
    assert torch.allclose(loss, manual, atol=1e-5)


def test_causalidade(modelo: ApoloLanguageModel) -> None:
    """Alterar o último token não pode alterar os logits anteriores."""
    torch.manual_seed(7)
    idx = torch.randint(0, modelo.config.vocab_size, (1, 10))
    with torch.no_grad():
        logits_a, _ = modelo(idx)
        idx_b = idx.clone()
        idx_b[0, -1] = (idx_b[0, -1] + 1) % modelo.config.vocab_size
        logits_b, _ = modelo(idx_b)
    assert torch.allclose(logits_a[0, :-1], logits_b[0, :-1], atol=1e-5)


def test_kv_cache_equivale_ao_forward_completo(modelo: ApoloLanguageModel) -> None:
    """A geração com KV Cache deve produzir os mesmos logits do forward completo."""
    from core.kv_cache import KVCache

    torch.manual_seed(11)
    idx = torch.randint(0, modelo.config.vocab_size, (1, 10))
    with torch.no_grad():
        completo, _ = modelo(idx)
        cache = KVCache(
            n_layer=modelo.config.n_layer,
            max_seq_len=modelo.config.block_size,
            batch_size=1,
            n_kv_head=modelo.config.n_kv_head,
            head_dim=modelo.config.head_dim,
        )
        logits_prefill, _ = modelo(idx[:, :9], cache=cache)
        logits_1, _ = modelo(idx[:, 9:], cache=cache)
    assert torch.allclose(logits_prefill[0, -1], completo[0, 8], atol=1e-4)
    assert torch.allclose(logits_1[0, -1], completo[0, 9], atol=1e-4)


def test_generate_respeita_block_size(modelo: ApoloLanguageModel) -> None:
    idx = torch.randint(0, modelo.config.vocab_size, (1, 4))
    with torch.no_grad():
        saida = modelo.generate(idx, max_new_tokens=5, temperature=0.0)
    assert saida.shape == (1, 5)


def test_generate_guloso_deterministico(modelo: ApoloLanguageModel) -> None:
    idx = torch.randint(0, modelo.config.vocab_size, (1, 4))
    with torch.no_grad():
        a = modelo.generate(idx, max_new_tokens=6, temperature=0.0)
        b = modelo.generate(idx, max_new_tokens=6, temperature=0.0)
    assert torch.equal(a, b)


def test_generate_com_e_sem_cache_sao_equivalentes(modelo: ApoloLanguageModel) -> None:
    """O caminho sem KV Cache deve produzir a mesma saída gulosa."""
    torch.manual_seed(21)
    idx = torch.randint(0, modelo.config.vocab_size, (1, 6))
    with torch.no_grad():
        com_cache = modelo.generate(idx, max_new_tokens=8, temperature=0.0, use_cache=True)
        sem_cache = modelo.generate(idx, max_new_tokens=8, temperature=0.0, use_cache=False)
    assert torch.equal(com_cache, sem_cache)


def test_embeddings_compartilhados(modelo: ApoloLanguageModel) -> None:
    assert modelo.config.tie_embeddings
    assert modelo.lm_head.weight is modelo.tok_emb.embedding.weight


def test_num_parametros_confere(config_pequena: ModelConfig) -> None:
    modelo = ApoloLanguageModel(config_pequena)
    total_real = sum(p.numel() for p in modelo.parameters())
    assert modelo.num_parameters() == total_real
    # A estimativa do config deve ficar próxima do valor real (±2%).
    estimativa = config_pequena.n_parameters
    assert abs(estimativa - total_real) / total_real < 0.02


def test_grupos_de_decay_separam_matrizes(modelo: ApoloLanguageModel) -> None:
    grupos = modelo.parameter_groups_for_decay(0.1)
    assert len(grupos) == 2
    assert grupos[0]["weight_decay"] == 0.1
    assert grupos[1]["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in grupos[0]["params"])
    assert all(p.dim() < 2 for p in grupos[1]["params"])


def test_save_load_roundtrip(tmp_path, modelo: ApoloLanguageModel) -> None:
    modelo.save(tmp_path)
    recarregado = ApoloLanguageModel.load(tmp_path)
    idx = torch.randint(0, modelo.config.vocab_size, (1, 8))
    with torch.no_grad():
        a, _ = modelo(idx)
        b, _ = recarregado(idx)
    assert torch.allclose(a, b, atol=1e-6)


def test_config_serializacao(tmp_path, config_pequena: ModelConfig) -> None:
    caminho = config_pequena.save(tmp_path / "config.json")
    recarregada = ModelConfig.load(caminho)
    assert recarregada.to_dict() == config_pequena.to_dict()
