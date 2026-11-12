"""Processamento incremental com progresso persistido.

Usado para preparar datasets (download, extração de frames, espectrogramas,
tokenização) em **blocos**, de forma que uma interrupção não obrigue a
recomeçar do zero.

O progresso é gravado em JSON após cada bloco:

.. code-block:: json

    {
      "ultimo_indice": 1200,
      "processados": 1200,
      "concluidos": 1198,
      "falhas": 2,
      "atualizado_em": 1730000000.0
    }
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

LOGGER = logging.getLogger("apolo.datasets.incremental")


@dataclass
class ProgressoIncremental:
    """Estado persistido do processamento incremental."""

    total: int = 0
    processados: int = 0
    concluidos: int = 0
    falhas: int = 0
    ultimo_indice: int = -1
    inicio: Optional[float] = None
    fim: Optional[float] = None
    estado: str = "nao_iniciado"   # nao_iniciado | em_andamento | concluido | falhou | pausado
    mensagem: str = ""
    ids_processados: List[str] = field(default_factory=list)

    def para_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def de_dict(dados: Dict[str, Any]) -> "ProgressoIncremental":
        return ProgressoIncremental(
            total=int(dados.get("total", 0) or 0),
            processados=int(dados.get("processados", 0) or 0),
            concluidos=int(dados.get("concluidos", 0) or 0),
            falhas=int(dados.get("falhas", 0) or 0),
            ultimo_indice=int(dados.get("ultimo_indice", -1)),
            inicio=dados.get("inicio"),
            fim=dados.get("fim"),
            estado=dados.get("estado", "nao_iniciado"),
            mensagem=dados.get("mensagem", ""),
            ids_processados=list(dados.get("ids_processados", []) or []),
        )


class IncrementalProcessor:
    """Processa uma lista de itens em blocos, persistindo o progresso."""

    def __init__(
        self,
        arquivo_progresso: str | Path,
        itens: Optional[Sequence[Any]] = None,
        tamanho_bloco: int = 100,
        log: Optional[logging.Logger] = None,
        guardar_ids: bool = True,
    ) -> None:
        self.arquivo_progresso = Path(arquivo_progresso)
        self.itens: List[Any] = list(itens or [])
        self.tamanho_bloco = max(1, int(tamanho_bloco))
        self.log = log or LOGGER
        self.guardar_ids = guardar_ids
        self.progresso = self.carregar()

    # ------------------------------------------------------------------ #
    # Persistência
    # ------------------------------------------------------------------ #
    def carregar(self) -> ProgressoIncremental:
        """Carrega o progresso salvo (ou um novo)."""
        if self.arquivo_progresso.exists():
            try:
                dados = json.loads(self.arquivo_progresso.read_text(encoding="utf-8"))
                return ProgressoIncremental.de_dict(dados)
            except (OSError, json.JSONDecodeError):
                self.log.warning("Progresso ilegível; reiniciando: %s", self.arquivo_progresso)
        progresso = ProgressoIncremental(total=len(self.itens))
        return progresso

    def salvar(self) -> None:
        """Grava o progresso de forma atômica."""
        self.arquivo_progresso.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.arquivo_progresso.with_suffix(self.arquivo_progresso.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.progresso.para_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, self.arquivo_progresso)

    # ------------------------------------------------------------------ #
    # Execução
    # ------------------------------------------------------------------ #
    def _ja_processado(self, indice: int, item: Any) -> bool:
        """Define se o item já foi processado (para retomada)."""
        if indice <= self.progresso.ultimo_indice:
            return True
        if self.guardar_ids and self.progresso.ids_processados:
            return str(item) in set(self.progresso.ids_processados)
        return False

    def executar(
        self,
        processar: Callable[[Any, int], Any],
        ao_concluir_bloco: Optional[Callable[[int, int, int], None]] = None,
        parar_quando: Optional[Callable[[Any, int], bool]] = None,
    ) -> ProgressoIncremental:
        """Processa os itens restantes.

        ``processar`` recebe ``(item, indice)`` e pode levantar exceções; erros
        são contabilizados e o item é registrado como falha (o treino não
        quebra por um arquivo isolado).
        """
        if not self.itens:
            self.progresso.total = 0
            self.progresso.estado = "concluido"
            self.progresso.fim = time.time()
            self.salvar()
            return self.progresso

        if self.progresso.inicio is None:
            self.progresso.inicio = time.time()
        self.progresso.total = len(self.itens)
        self.progresso.estado = "em_andamento"

        inicio = self.progresso.ultimo_indice + 1
        for indice in range(inicio, len(self.itens)):
            item = self.itens[indice]
            if self._ja_processado(indice, item):
                continue
            if parar_quando is not None and parar_quando(item, indice):
                self.progresso.estado = "pausado"
                self.progresso.mensagem = f"parado no índice {indice}"
                self.progresso.ultimo_indice = indice - 1
                self.salvar()
                return self.progresso
            try:
                processar(item, indice)
                self.progresso.concluidos += 1
                if self.guardar_ids:
                    self.progresso.ids_processados.append(str(item))
            except Exception as exc:  # item isolado não derruba o lote
                self.progresso.falhas += 1
                self.log.warning("Falha ao processar item %d (%s): %s", indice, item, exc)
            self.progresso.processados += 1
            self.progresso.ultimo_indice = indice

            if self.progresso.processados % self.tamanho_bloco == 0:
                self.salvar()
                if ao_concluir_bloco is not None:
                    ao_concluir_bloco(
                        self.progresso.processados,
                        self.progresso.concluidos,
                        self.progresso.falhas,
                    )

        self.progresso.estado = "concluido"
        self.progresso.fim = time.time()
        self.salvar()
        return self.progresso

    # ------------------------------------------------------------------ #
    @property
    def concluido(self) -> bool:
        return self.progresso.estado == "concluido"

    def reiniciar(self) -> None:
        """Zera o progresso (reprocessa tudo)."""
        self.progresso = ProgressoIncremental(total=len(self.itens))
        self.salvar()

    def resumo(self) -> Dict[str, Any]:
        p = self.progresso
        duracao = (p.fim or time.time()) - (p.inicio or time.time())
        return {
            "total": p.total,
            "processados": p.processados,
            "concluidos": p.concluidos,
            "falhas": p.falhas,
            "estado": p.estado,
            "duracao_s": round(max(duracao, 0.0), 2),
            "arquivo": str(self.arquivo_progresso),
        }
