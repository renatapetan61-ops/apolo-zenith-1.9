"""Testes do pipeline de dados do Apolo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from datasets.text_loader import (
    PackedTokenDataset,
    deduplicar,
    dividir_documentos,
    normalizar_texto,
    qualidade_texto,
    remover_pii,
    salvar_tokens,
    simhash,
)


def test_normalizacao_remove_cabecalho_gutenberg() -> None:
    bruto = (
        "Project Gutenberg EBook\n\n*** START OF THE PROJECT GUTENBERG EBOOK TESTE ***\n\n"
        "Texto útil do livro.\n\n*** END OF THE PROJECT GUTENBERG EBOOK TESTE ***\n"
    )
    limpo = normalizar_texto(bruto, fonte="gutenberg")
    assert "Texto útil do livro." in limpo
    assert "GUTENBERG" not in limpo


def test_normalizacao_wikitexto() -> None:
    bruto = "'''Título'''\n== Seção ==\nTexto com [[ligação|âncora]] e {{template}}."
    limpo = normalizar_texto(bruto, fonte="wikisource")
    assert "{{template}}" not in limpo
    assert "Seção" in limpo


def test_remocao_de_pii() -> None:
    texto = "Contato: joao.silva@example.com ou (11) 98765-4321. Docs em https://exemplo.com"
    limpo = remover_pii(texto)
    assert "joao.silva@example.com" not in limpo
    assert "98765-4321" not in limpo
    assert "[EMAIL]" in limpo and "[URL]" in limpo


def test_filtro_de_qualidade() -> None:
    assert qualidade_texto("texto curto")[0] is False
    bom = "Esta é uma frase completa em português com número suficiente de palavras " * 4
    aprovado, motivo = qualidade_texto(bom)
    assert aprovado and motivo == "ok"


def test_deduplicacao_exata_e_aproximada() -> None:
    base = "Frase de teste com conteúdo suficiente para passar no filtro de qualidade. " * 6
    docs = [base, base, base.replace("teste", "teste"), base + " slightly different ending here."]
    mantidos, removidos = deduplicar(docs, limiar=3)
    assert removidos >= 1
    assert len(mantidos) < len(docs)


def test_simhash_estavel() -> None:
    texto = "Aprendizado de máquina com transformers." * 3
    assert simhash(texto) == simhash(texto)
    outro = "Culinária brasileira com feijoada." * 3
    assert simhash(texto) != simhash(outro)


def test_division_deterministica() -> None:
    docs = [f"documento {i}" for i in range(100)]
    a = dividir_documentos(docs, (0.8, 0.1, 0.1), semente=7)
    b = dividir_documentos(docs, (0.8, 0.1, 0.1), semente=7)
    assert a == b
    assert len(a["train"]) == 80
    assert len(a["val"]) == 10
    assert len(a["test"]) == 10
    assert set(a["train"]) & set(a["test"]) == set()


def test_packed_dataset(tmp_path: Path) -> None:
    tokens = list(range(100)) + [0, 1, 2, 3]
    caminho = salvar_tokens(tokens, tmp_path / "dados.bin")
    dataset = PackedTokenDataset(caminho, block_size=16)
    assert len(dataset) == len(tokens) // 16
    x, y = dataset[0]
    assert x.shape == (16,) and y.shape == (16,)
    assert torch.equal(y[:-1], x[1:])
    assert int(y[-1]) == tokens[16]
