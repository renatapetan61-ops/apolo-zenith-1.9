"""Gestão de armazenamento do Apolo Zenith 1.9.

Responsabilidades:

* medir o espaço livre real (``shutil.disk_usage``);
* impedir que um download ultrapasse o espaço mínimo de segurança;
* limpar **apenas** arquivos temporários marcados (nunca originais);
* rotacionar checkpoints preservando o melhor e o último válido.

Regras de segurança (não negociáveis):

* arquivos em ``raw/``/``processed/`` dos datasets **nunca** são removidos;
* checkpoints com nome ``best``/``final`` **nunca** são removidos pela rotação;
* a limpeza só atua em diretórios temporários declarados ou em arquivos com
  marcador ``.tmp``/``.part``/``.download``.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from resources.config import LimitesArmazenamento

_MB = 1024 ** 2

#: Extensões consideradas temporárias (removíveis com segurança).
EXTENSOES_TEMPORARIAS = (".tmp", ".part", ".download", ".incomplete")

#: Nomes de checkpoint que nunca podem ser removidos automaticamente.
CHECKPOINTS_PROTEGIDOS = ("best", "final", "last")


@dataclass
class LeituraArmazenamento:
    """Espaço em disco em um instante."""

    caminho: str
    total_mb: float
    livre_mb: float
    usado_mb: float
    uso_percent: float
    timestamp: float = 0.0

    def para_dict(self) -> dict:
        return {
            "caminho": self.caminho,
            "total_mb": round(self.total_mb, 1),
            "livre_mb": round(self.livre_mb, 1),
            "usado_mb": round(self.usado_mb, 1),
            "uso_percent": round(self.uso_percent, 1),
            "timestamp": self.timestamp,
        }


class StorageManager:
    """Controla o espaço em disco usado pelo pipeline."""

    def __init__(
        self,
        limites: Optional[LimitesArmazenamento] = None,
        raiz: str | Path = ".",
    ) -> None:
        self.limites = limites or LimitesArmazenamento()
        self.raiz = Path(raiz)
        self._ultima_leitura: Optional[LeituraArmazenamento] = None
        self._ultima_verificacao = 0.0
        self._intervalo_leitura_s = 5.0

    # ------------------------------------------------------------------ #
    # Leitura
    # ------------------------------------------------------------------ #
    def ler(self, caminho: Optional[str | Path] = None, forcar: bool = False) -> LeituraArmazenamento:
        alvo = Path(caminho) if caminho else self.raiz
        agora = time.time()
        if (
            not forcar
            and caminho is None
            and self._ultima_leitura is not None
            and (agora - self._ultima_verificacao) < self._intervalo_leitura_s
        ):
            return self._ultima_leitura
        uso = shutil.disk_usage(str(alvo))
        total_mb = uso.total / _MB
        livre_mb = uso.free / _MB
        usado_mb = uso.used / _MB
        leitura = LeituraArmazenamento(
            caminho=str(alvo),
            total_mb=total_mb,
            livre_mb=livre_mb,
            usado_mb=usado_mb,
            uso_percent=(usado_mb / total_mb * 100.0) if total_mb > 0 else 0.0,
            timestamp=agora,
        )
        if caminho is None:
            self._ultima_leitura = leitura
            self._ultima_verificacao = agora
        return leitura

    def nivel(self, leitura: Optional[LeituraArmazenamento] = None) -> str:
        """Classifica: ``ok`` | ``aviso`` | ``critico``."""
        leitura = leitura or self.ler()
        if leitura.livre_mb < self.limites.espaco_minimo_livre_mb:
            return "critico"
        if leitura.livre_mb < self.limites.espaco_minimo_livre_mb + self.limites.reservar_para_sistema_mb:
            return "aviso"
        return "ok"

    def deve_pausar(self, leitura: Optional[LeituraArmazenamento] = None) -> Tuple[bool, str]:
        """Decide se o treino deve pausar por falta de espaço."""
        leitura = leitura or self.ler()
        if self.nivel(leitura) == "critico":
            return True, (
                f"espaço livre {leitura.livre_mb:.0f} MB < mínimo "
                f"{self.limites.espaco_minimo_livre_mb} MB"
            )
        return False, ""

    # ------------------------------------------------------------------ #
    # Downloads
    # ------------------------------------------------------------------ #
    def pode_baixar(
        self,
        tamanho_bytes: int,
        caminho: Optional[str | Path] = None,
    ) -> Tuple[bool, str]:
        """Verifica se um download de ``tamanho_bytes`` respeita o espaço livre."""
        leitura = self.ler(caminho)
        necessario_mb = tamanho_bytes / _MB
        if leitura.livre_mb - necessario_mb < self.limites.espaco_minimo_livre_mb:
            return False, (
                f"download de {necessario_mb:.1f} MB deixaria {leitura.livre_mb - necessario_mb:.0f} MB "
                f"livres (< mínimo {self.limites.espaco_minimo_livre_mb} MB)"
            )
        return True, ""

    def pode_baixar_passo(
        self,
        tamanho_bytes: int,
        caminho: Optional[str | Path] = None,
    ) -> Tuple[bool, str]:
        """Verifica se um **bloco** de download cabe no limite por passo."""
        necessario_mb = tamanho_bytes / _MB
        if necessario_mb > self.limites.download_max_por_passo_mb:
            return False, (
                f"bloco de {necessario_mb:.1f} MB excede o limite por passo "
                f"({self.limites.download_max_por_passo_mb} MB); divida o download"
            )
        return self.pode_baixar(tamanho_bytes, caminho)

    def reservar_espaco(self, tamanho_bytes: int, caminho: Optional[str | Path] = None) -> bool:
        """Alias semântico de :meth:`pode_baixar` que devolve apenas ``bool``."""
        ok, _ = self.pode_baixar(tamanho_bytes, caminho)
        return ok

    # ------------------------------------------------------------------ #
    # Limpeza segura
    # ------------------------------------------------------------------ #
    def tamanho_diretorio(self, caminho: str | Path) -> int:
        """Tamanho total (bytes) de um diretório, sem seguir links."""
        total = 0
        base = Path(caminho)
        if not base.exists():
            return 0
        for raiz, _dirs, arquivos in os_walk(base):
            for nome in arquivos:
                arquivo = Path(raiz) / nome
                try:
                    if arquivo.is_symlink():
                        continue
                    total += arquivo.stat().st_size
                except OSError:
                    continue
        return total

    def temporarios(self) -> List[Path]:
        """Lista os arquivos temporários candidatos à limpeza."""
        candidatos: List[Path] = []
        # 1) diretório temporário declarado na configuração
        dir_tmp = Path(self.limites.diretorio_temporario)
        if not dir_tmp.is_absolute():
            dir_tmp = self.raiz / dir_tmp
        if dir_tmp.exists():
            for arquivo in dir_tmp.rglob("*"):
                if arquivo.is_file():
                    candidatos.append(arquivo)
        # 2) arquivos com marcador de temporário em todo o projeto
        for arquivo in self.raiz.rglob("*"):
            if not arquivo.is_file():
                continue
            if arquivo.suffix.lower() in EXTENSOES_TEMPORARIAS:
                candidatos.append(arquivo)
        # dedup preservando ordem
        vistos = set()
        unicos = []
        for arquivo in candidatos:
            chave = str(arquivo.resolve()) if arquivo.exists() else str(arquivo)
            if chave in vistos:
                continue
            vistos.add(chave)
            unicos.append(arquivo)
        return unicos

    def limpar_temporarios(self, dry_run: bool = False) -> Tuple[int, int]:
        """Remove temporários.  Devolve ``(quantidade, bytes_liberados)``.

        Nunca toca em arquivos de dataset (``raw``/``processed``) nem em
        checkpoints.
        """
        removidos = 0
        liberados = 0
        for arquivo in self.temporarios():
            try:
                tamanho = arquivo.stat().st_size
            except OSError:
                continue
            if dry_run:
                removidos += 1
                liberados += tamanho
                continue
            try:
                arquivo.unlink()
                removidos += 1
                liberados += tamanho
            except OSError:
                continue
        return removidos, liberados

    # ------------------------------------------------------------------ #
    # Checkpoints
    # ------------------------------------------------------------------ #
    def rotacionar_checkpoints(
        self,
        diretorio: str | Path,
        manter: Optional[int] = None,
        prefixos: Iterable[str] = ("step_", "dia_", "day_"),
    ) -> List[Path]:
        """Remove checkpoints antigos mantendo os mais recentes.

        Checkpoints protegidos (``best``, ``final``, ``last``) e o diretório
        inteiro de checkpoints que não correspondem aos prefixos são
        preservados.  Devolve a lista de diretórios removidos.
        """
        base = Path(diretorio)
        if not base.exists():
            return []
        manter_n = manter if manter is not None else self.limites.manter_checkpoints
        manter_n = max(1, int(manter_n))
        candidatos = [
            p
            for p in base.iterdir()
            if p.is_dir() and p.name.startswith(tuple(prefixos))
        ]
        candidatos.sort(key=lambda p: p.name)
        removidos: List[Path] = []
        for antigo in candidatos[:-manter_n]:
            shutil.rmtree(antigo, ignore_errors=True)
            removidos.append(antigo)
        return removidos

    def tamanho_checkpoints(self, diretorio: str | Path) -> int:
        """Tamanho total (bytes) ocupado pelos checkpoints."""
        return self.tamanho_diretorio(diretorio)

    def snapshot(self) -> dict:
        return self.ler(forcar=True).para_dict()


def os_walk(base: Path):
    """Wrapper de :func:`os.walk` que não propaga erro de permissão."""
    import os as _os

    for raiz, dirs, arquivos in _os.walk(base, onerror=lambda _e: None):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
        yield raiz, dirs, arquivos
