"""Pipeline de dados de **código** do Apolo Zenith 1.9.

Objetivo: transformar os arquivos de código baixados (repositórios reais)
em um corpus tokenizado e empacotado, com marcação da linguagem.

Regras:

* leitura **streaming** (um arquivo por vez) — nada é carregado inteiro na
  RAM;
* os arquivos originais em ``datasets/code/raw`` **nunca** são modificados;
* documentos vazios ou triviais são descartados;
* o tokenizer BPE já treinado é reutilizado (mesmo vocabulário do texto).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from datasets.text_loader import (
    deduplicar,
    dividir_documentos,
    qualidade_texto,
    salvar_tokens,
)

LOGGER = logging.getLogger("apolo.datasets.code")

#: Mapeamento extensão -> linguagem (para a marcação dos documentos).
EXTENSOES_LINGUAGEM = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".kt": "kotlin",
    ".rs": "rust",
    ".go": "go",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".sh": "shell",
    ".bash": "shell",
    ".hs": "haskell",
}


@dataclass
class DocumentoCodigo:
    """Um documento de código com metadados."""

    texto: str
    linguagem: str
    repositorio: str
    caminho: str

    def para_dict(self) -> Dict[str, Any]:
        return {
            "texto": self.texto,
            "linguagem": self.linguagem,
            "repositorio": self.repositorio,
            "caminho": self.caminho,
        }


def coletar_codigo(
    pasta: str | Path,
    max_chars: int = 20000,
    min_chars: int = 120,
    max_arquivos: Optional[int] = None,
) -> List[DocumentoCodigo]:
    """Coleta os arquivos de código (streaming) de uma pasta.

    A estrutura esperada é ``<pasta>/<linguagem>/<repositorio>/<arquivo>``.
    """
    pasta = Path(pasta)
    documentos: List[DocumentoCodigo] = []
    if not pasta.exists():
        return documentos
    for arquivo in sorted(pasta.rglob("*")):
        if not arquivo.is_file():
            continue
        linguagem = EXTENSOES_LINGUAGEM.get(arquivo.suffix.lower())
        if linguagem is None:
            continue
        try:
            texto = arquivo.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        texto = texto.replace("\r\n", "\n").replace("\r", "\n")
        if len(texto) > max_chars:
            texto = texto[:max_chars]
        if len(texto) < min_chars:
            continue
        partes = arquivo.relative_to(pasta).parts
        repositorio = partes[1] if len(partes) > 1 else arquivo.parent.name
        documentos.append(
            DocumentoCodigo(
                texto=texto,
                linguagem=linguagem,
                repositorio=repositorio,
                caminho=str(arquivo),
            )
        )
        if max_arquivos and len(documentos) >= max_arquivos:
            break
    return documentos


def para_documento_texto(documento: DocumentoCodigo) -> str:
    """Formata o código para o LM (marcação da linguagem + corpo)."""
    return f"<|codigo:{documento.linguagem}|>\n{documento.texto}\n"


def preparar_corpus_codigo(
    pasta_raw: str | Path,
    pasta_saida: str | Path,
    tokenizer_dir: str | Path,
    proporcoes: Tuple[float, float, float] = (0.9, 0.05, 0.05),
    max_arquivos: Optional[int] = None,
    seed: int = 1337,
) -> Dict[str, Any]:
    """Tokeniza e empacota o corpus de código em ``train/val/test.bin``."""
    from tokenizer.tokenizer import BPETokenizer

    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    tokenizer = BPETokenizer.load(tokenizer_dir)

    documentos = coletar_codigo(pasta_raw, max_arquivos=max_arquivos)
    if not documentos:
        raise SystemExit(f"nenhum arquivo de código encontrado em {pasta_raw}")

    textos = [para_documento_texto(d) for d in documentos]
    # Deduplicação exata/aproximada sobre o texto completo.
    mantidos, removidos = deduplicar(textos)
    divisao = dividir_documentos(mantidos, proporcoes=proporcoes, semente=seed)

    estatisticas: Dict[str, Any] = {"arquivos": len(documentos), "removidos_dup": removidos}
    for nome, lista in divisao.items():
        ids: List[int] = []
        for texto in lista:
            # Atenção: ``encode_dataset`` recebe uma **lista** de textos.
            # Passar uma string faria o iterador percorrer os caracteres
            # (bug real que gerou um corpus de documentos de 1 token).
            ids.extend(tokenizer.encode(texto, add_bos=True, add_eos=True))
        caminho = salvar_tokens(ids, pasta_saida / f"{nome}.bin")
        estatisticas[nome] = {"tokens": len(ids), "caminho": str(caminho)}
        # Índice de metadados (uma linha por documento, para análise).
        if nome == "train":
            with (pasta_saida / "train.jsonl").open("w", encoding="utf-8") as fh:
                for doc, texto in zip(documentos, mantidos):
                    fh.write(
                        json.dumps(
                            {
                                "linguagem": doc.linguagem,
                                "repositorio": doc.repositorio,
                                "caminho": doc.caminho,
                                "chars": len(texto),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    (pasta_saida / "resumo.json").write_text(
        json.dumps(estatisticas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return estatisticas


def contar_linguagens(pasta_raw: str | Path) -> Dict[str, int]:
    """Conta os arquivos por linguagem (real, sem carregar o conteúdo)."""
    pasta = Path(pasta_raw)
    contagem: Dict[str, int] = {}
    if not pasta.exists():
        return contagem
    for arquivo in pasta.rglob("*"):
        if not arquivo.is_file():
            continue
        linguagem = EXTENSOES_LINGUAGEM.get(arquivo.suffix.lower())
        if linguagem:
            contagem[linguagem] = contagem.get(linguagem, 0) + 1
    return dict(sorted(contagem.items(), key=lambda kv: -kv[1]))
