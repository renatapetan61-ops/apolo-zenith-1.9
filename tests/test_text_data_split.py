"""Regressões para divisão de corpus pequeno e arquivos de tokens vazios."""

import numpy as np

from datasets.text_loader import PackedTokenDataset, dividir_documentos


def test_divisao_de_oito_documentos_reserva_validacao_e_teste():
    documentos = [f"documento {i}" for i in range(8)]
    splits = dividir_documentos(documentos, proporcoes=(0.97, 0.015, 0.015))
    assert len(splits["train"]) == 6
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1
    assert sum(map(len, splits.values())) == len(documentos)


def test_divisao_de_dois_documentos_preserva_treino_e_validacao():
    splits = dividir_documentos(["um", "dois"])
    assert len(splits["train"]) == 1
    assert len(splits["val"]) == 1
    assert splits["test"] == []


def test_dataset_aceita_arquivo_vazio_sem_memmap(tmp_path):
    arquivo = tmp_path / "val.bin"
    arquivo.write_bytes(b"")
    dataset = PackedTokenDataset(arquivo, block_size=64)
    assert len(dataset) == 0
    assert dataset.num_tokens == 0


def test_dataset_vazio_nao_inventa_tokens(tmp_path):
    arquivo = tmp_path / "val.bin"
    np.asarray([], dtype=np.uint16).tofile(arquivo)
    dataset = PackedTokenDataset(arquivo, block_size=64)
    assert len(dataset) == 0
