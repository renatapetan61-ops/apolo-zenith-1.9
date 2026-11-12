"""Configuração do pytest para o Apolo Zenith 1.9.

Garante que o diretório raiz do projeto esteja no ``sys.path`` para que os
pacotes (``core``, ``tokenizer``, ``training``, …) sejam importáveis.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
