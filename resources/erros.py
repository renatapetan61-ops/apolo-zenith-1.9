"""Registro centralizado de erros do Apolo Zenith 1.9 (``logs/errors.log``).

Formato: uma linha JSON por erro, com arquivo, função, mensagem e traceback
resumido.  É a fonte de verdade para auditar problemas do pipeline.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Optional

CAMINHO_PADRAO = Path("logs/errors.log")


def registrar_erro(
    mensagem: str,
    excecao: Optional[BaseException] = None,
    contexto: Optional[dict] = None,
    caminho: str | Path = CAMINHO_PADRAO,
) -> Path:
    """Grava um erro no log (sem levantar exceção)."""
    registro: dict[str, Any] = {
        "timestamp": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mensagem": mensagem,
    }
    if contexto:
        registro["contexto"] = contexto
    if excecao is not None:
        registro["excecao"] = type(excecao).__name__
        registro["detalhe"] = str(excecao)[:500]
        registro["traceback"] = "".join(
            traceback.format_exception(type(excecao), excecao, excecao.__traceback__)
        )[-2000:]
    destino = Path(caminho)
    try:
        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(registro, ensure_ascii=False) + "\n")
    except OSError:
        pass
    return destino


def registrar_excecao(
    contexto: str,
    excecao: BaseException,
    caminho: str | Path = CAMINHO_PADRAO,
) -> Path:
    """Atalho para registrar uma exceção com contexto."""
    return registrar_erro(contexto, excecao, caminho=caminho)
