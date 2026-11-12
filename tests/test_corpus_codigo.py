"""Teste de regressão do corpus de código (bug real corrigido).

Bug: o corpus de código era tokenizado com
``tokenizer.encode_dataset(texto, ...)`` — mas esse método espera uma
**lista** de textos.  Passando uma string, o iterador percorria os
**caracteres**, produzindo documentos de 1 token (BOS + caractere + EOS).
O modelo de código treinava sobre esse corpus e gerava apenas espaços.

Este teste garante que o corpus é **real**: documentos longos, BOS/EOS
raros e tokens de conteúdo presentes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from datasets.code_loader import preparar_corpus_codigo
from datasets.text_loader import carregar_tokens


def test_corpus_de_codigo_tem_documentos_reais(tmp_path: Path) -> None:
    """O corpus de código deve conter documentos de tamanho real."""
    tokenizer_dir = Path("datasets_storage/processed/tokenizer")
    if not tokenizer_dir.exists():
        tokenizer_dir = Path("tokenizers/tokenizer")
    if not tokenizer_dir.exists():
        pytest.skip("tokenizer ausente (datasets_storage/processed/tokenizer ou tokenizers/tokenizer)")

    pasta = tmp_path / "raw" / "python" / "repo"
    pasta.mkdir(parents=True)
    (pasta / "a.py").write_text(
        "def soma(a, b):\n    return a + b\n\n" * 30, encoding="utf-8"
    )
    (pasta / "b.py").write_text("class X:\n    pass\n\n" * 30, encoding="utf-8")
    (pasta / "c.py").write_text("import os\n\nfor raiz, dirs, arquivos in os.walk('.'):\n    pass\n\n" * 25, encoding="utf-8")
    (pasta / "d.py").write_text("def f(x):\n    if x > 0:\n        return x\n    return -x\n\n" * 25, encoding="utf-8")
    (pasta / "e.py").write_text("CONST = 42\n\ndef g():\n    return CONST\n\n" * 25, encoding="utf-8")
    (pasta / "f.py").write_text("try:\n    value = int('1')\nexcept ValueError:\n    value = 0\n\n" * 25, encoding="utf-8")

    saida = tmp_path / "processed"
    estatisticas = preparar_corpus_codigo(
        tmp_path / "raw", saida, tokenizer_dir=tokenizer_dir, proporcoes=(0.5, 0.25, 0.25)
    )
    assert estatisticas["train"]["tokens"] > 100

    for nome in ("train", "val", "test"):
        caminho = saida / f"{nome}.bin"
        if not caminho.exists():
            continue
        if caminho.stat().st_size == 0:
            continue
        tokens = carregar_tokens(caminho)
        assert len(tokens) > 0
        contagem = np.bincount(np.asarray(tokens, dtype=np.int64))
        bos = contagem[2] if len(contagem) > 2 else 0
        eos = contagem[3] if len(contagem) > 3 else 0
        # BOS/EOS não podem dominar o fluxo (no bug, ocupavam 66%).
        assert bos < len(tokens) * 0.1, f"{nome}: BOS demais ({bos}/{len(tokens)})"
        assert eos < len(tokens) * 0.1, f"{nome}: EOS demais ({eos}/{len(tokens)})"
        # Deve haver tokens de conteúdo (identificadores comuns).
        assert len(contagem) > 50, f"{nome}: vocabulário muito pequeno ({len(contagem)})"
