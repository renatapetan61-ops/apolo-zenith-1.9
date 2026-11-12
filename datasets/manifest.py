"""Manifesto de datasets do Apolo Zenith 1.9 (``dataset_manifest.json``).

Cada entrada registra, **apenas com valores medidos**:

* nome, modalidade, origem, URL oficial e licença;
* tamanho em bytes (baixado e em disco);
* quantidade de amostras (contada de verdade);
* formato, status de download, integridade SHA-256;
* status de preparação e status de utilização no treinamento.

Nada é preenchido com valores estimados: campos desconhecidos ficam
``null``/``"desconhecido"``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

STATUS_DOWNLOAD = ("pendente", "em_andamento", "concluido", "parcial", "falhou")
STATUS_PREPARACAO = ("pendente", "em_andamento", "concluido", "parcial", "falhou")
STATUS_TREINO = ("nao_utilizado", "em_andamento", "utilizado")


@dataclass
class EntradaManifesto:
    """Uma entrada do manifesto de datasets."""

    nome: str
    modalidade: str
    origem: str
    url_oficial: str
    licenca: str
    formato: str = ""
    tamanho_bytes: Optional[int] = None
    tamanho_em_disco_bytes: Optional[int] = None
    amostras: Optional[int] = None
    status_download: str = "pendente"
    sha256: Optional[str] = None
    status_preparacao: str = "pendente"
    status_treino: str = "nao_utilizado"
    arquivos: List[Dict[str, Any]] = field(default_factory=list)
    subfontes: List[Dict[str, Any]] = field(default_factory=list)
    observacoes: str = ""
    atualizado_em: float = 0.0

    def para_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetManifest:
    """Leitura e escrita do ``dataset_manifest.json``."""

    def __init__(self, caminho: str | Path = "dataset_manifest.json") -> None:
        self.caminho = Path(caminho)
        self.entradas: Dict[str, EntradaManifesto] = {}
        self._carregar()

    # ------------------------------------------------------------------ #
    def _carregar(self) -> None:
        if not self.caminho.exists():
            return
        try:
            dados = json.loads(self.caminho.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        itens = dados.get("datasets", dados) if isinstance(dados, dict) else dados
        for item in itens or []:
            if not isinstance(item, dict):
                continue
            validos = set(EntradaManifesto.__dataclass_fields__)  # type: ignore[attr-defined]
            entrada = EntradaManifesto(**{k: v for k, v in item.items() if k in validos})
            self.entradas[entrada.nome] = entrada

    def salvar(self) -> Path:
        """Grava o manifesto de forma atômica."""
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        dados = {
            "projeto": "Apolo Zenith 1.9",
            "atualizado_em": time.time(),
            "total_datasets": len(self.entradas),
            "total_bytes_baixado": sum(
                e.tamanho_bytes or 0 for e in self.entradas.values()
            ),
            "total_bytes_em_disco": sum(
                e.tamanho_em_disco_bytes or 0 for e in self.entradas.values()
            ),
            "por_modalidade": self.totais_por_modalidade(),
            "datasets": [e.para_dict() for e in self.entradas.values()],
        }
        tmp = self.caminho.with_suffix(self.caminho.suffix + ".tmp")
        tmp.write_text(json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.caminho)
        return self.caminho

    # ------------------------------------------------------------------ #
    def obter(self, nome: str) -> EntradaManifesto:
        if nome not in self.entradas:
            raise KeyError(f"Dataset fora do manifesto: {nome}")
        return self.entradas[nome]

    def registrar(self, entrada: EntradaManifesto) -> EntradaManifesto:
        entrada.atualizado_em = time.time()
        self.entradas[entrada.nome] = entrada
        self.salvar()
        return entrada

    def atualizar(self, nome: str, **campos: Any) -> EntradaManifesto:
        entrada = self.obter(nome)
        for chave, valor in campos.items():
            if hasattr(entrada, chave):
                setattr(entrada, chave, valor)
        entrada.atualizado_em = time.time()
        self.salvar()
        return entrada

    def definir_status(
        self,
        nome: str,
        status_download: Optional[str] = None,
        status_preparacao: Optional[str] = None,
        status_treino: Optional[str] = None,
        **campos: Any,
    ) -> EntradaManifesto:
        entrada = self.obter(nome)
        if status_download is not None:
            entrada.status_download = status_download
        if status_preparacao is not None:
            entrada.status_preparacao = status_preparacao
        if status_treino is not None:
            entrada.status_treino = status_treino
        for chave, valor in campos.items():
            if hasattr(entrada, chave):
                setattr(entrada, chave, valor)
        entrada.atualizado_em = time.time()
        self.salvar()
        return entrada

    # ------------------------------------------------------------------ #
    def totais_por_modalidade(self) -> Dict[str, Dict[str, Any]]:
        """Totais reais por modalidade (bytes baixados, em disco, amostras)."""
        totais: Dict[str, Dict[str, Any]] = {}
        for entrada in self.entradas.values():
            item = totais.setdefault(
                entrada.modalidade,
                {"datasets": 0, "bytes_baixado": 0, "bytes_em_disco": 0, "amostras": 0},
            )
            item["datasets"] += 1
            item["bytes_baixado"] += entrada.tamanho_bytes or 0
            item["bytes_em_disco"] += entrada.tamanho_em_disco_bytes or 0
            if entrada.amostras is not None:
                item["amostras"] += entrada.amostras
        return totais

    def resumo(self) -> Dict[str, Any]:
        """Resumo serializável."""
        return {
            "total_datasets": len(self.entradas),
            "bytes_baixado": sum(e.tamanho_bytes or 0 for e in self.entradas.values()),
            "bytes_em_disco": sum(
                e.tamanho_em_disco_bytes or 0 for e in self.entradas.values()
            ),
            "amostras": {
                e.nome: e.amostras for e in self.entradas.values() if e.amostras is not None
            },
            "por_modalidade": self.totais_por_modalidade(),
        }
