"""Exclusividade de treinamento e fila por prioridade.

O lock é baseado em ``fcntl.flock`` sobre um arquivo, o que é atômico entre
processos no mesmo sistema de arquivos (inclusive dentro do Proot).  Dentro do
arquivo são gravados o PID do dono, o *heartbeat* e a modalidade, de modo que
um processo morto possa ser detectado e o lock recuperado.

A :class:`FilaTreinos` mantém a fila persistida em JSON, respeitando a
prioridade ``linguagem > codigo > visao > audio > video > multimodal``.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from resources.config import PrioridadesTreino

DEFAULT_ORDEM = {
    "linguagem": 1,
    "codigo": 2,
    "visao": 3,
    "audio": 4,
    "video": 5,
    "multimodal": 6,
}


class TreinoConcorrenteError(RuntimeError):
    """Levantado quando já existe um treinamento ativo e o lock é ocupado."""

    def __init__(self, mensagem: str, dono: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(mensagem)
        self.dono = dono or {}


@dataclass
class TreinoSolicitacao:
    """Um item da fila de treinamentos."""

    id: str
    modalidade: str
    comando: List[str] = field(default_factory=list)
    prioridade: int = 99
    solicitado_em: float = 0.0
    estado: str = "aguardando"   # aguardando | rodando | concluido | falhou | cancelado
    inicio: Optional[float] = None
    fim: Optional[float] = None
    erro: Optional[str] = None
    metadados: Dict[str, Any] = field(default_factory=dict)

    def para_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def de_dict(dados: Dict[str, Any]) -> "TreinoSolicitacao":
        return TreinoSolicitacao(
            id=dados.get("id", ""),
            modalidade=dados.get("modalidade", "linguagem"),
            comando=list(dados.get("comando", [])),
            prioridade=int(dados.get("prioridade", 99)),
            solicitado_em=float(dados.get("solicitado_em", 0.0)),
            estado=dados.get("estado", "aguardando"),
            inicio=dados.get("inicio"),
            fim=dados.get("fim"),
            erro=dados.get("erro"),
            metadados=dados.get("metadados", {}) or {},
        )


def _pid_ativo(pid: int) -> bool:
    """Verifica se o PID existe (sem enviar sinal)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _pid_start_time(pid: int) -> Optional[int]:
    """Campo 22 de ``/proc/<pid>/stat`` (starttime em ticks), se acessível."""
    try:
        bruto = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    try:
        # O nome do processo pode conter espaços: tudo após o ')' é o resto.
        resto = bruto[bruto.rfind(")") + 1 :].split()
        return int(resto[19])  # campo 22 contando desde 1 -> índice 19
    except (ValueError, IndexError):
        return None


class TrainingLock:
    """Lock de exclusividade para treinos (um por vez).

    O mecanismo atômico é ``fcntl.flock`` sobre ``caminho``.  O arquivo de
    lock **nunca** é substituído (``os.replace`` trocaria o inode e liberaria
    o ``flock`` silenciosamente): o estado (PID, heartbeat, modalidade) é
    gravado em um arquivo lateral ``<lock>.estado.json``.
    """

    def __init__(
        self,
        caminho: str | Path = "logs/apolo_training.lock",
        ordem_prioridade: Optional[Dict[str, int]] = None,
    ) -> None:
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.caminho_estado = self.caminho.with_name(self.caminho.name + ".estado.json")
        self.ordem_prioridade = dict(ordem_prioridade or DEFAULT_ORDEM)
        self._fd: Optional[int] = None
        self._dono_pid: Optional[int] = None
        self._modalidade: Optional[str] = None
        self._heartbeat = 0.0
        self._adquirido_em = 0.0

    # ------------------------------------------------------------------ #
    # Leitura do estado do lock
    # ------------------------------------------------------------------ #
    def _ler_estado_arquivo(self) -> Dict[str, Any]:
        if not self.caminho_estado.exists():
            return {}
        try:
            return json.loads(self.caminho_estado.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _escrever_estado(self, dados: Dict[str, Any]) -> None:
        tmp = self.caminho_estado.with_suffix(self.caminho_estado.suffix + ".tmp")
        tmp.write_text(json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.caminho_estado)

    def dono(self) -> Optional[Dict[str, Any]]:
        """Dados do processo que detém o lock, se houver."""
        estado = self._ler_estado_arquivo()
        if not estado:
            return None
        pid = int(estado.get("pid", 0) or 0)
        if not _pid_ativo(pid):
            return None
        inicio = estado.get("start_time")
        if inicio is not None:
            atual = _pid_start_time(pid)
            if atual is not None and int(inicio) != atual:
                # PID reutilizado por outro processo: lock considered órfão.
                return None
        return estado

    def ocupado(self) -> bool:
        """``True`` se há um treino ativo (lock válido)."""
        return self.dono() is not None

    # ------------------------------------------------------------------ #
    # Aquisição / liberação
    # ------------------------------------------------------------------ #
    def adquirir(
        self,
        modalidade: str = "linguagem",
        comando: Optional[List[str]] = None,
        forcar: bool = False,
    ) -> "TrainingLock":
        """Tenta adquirir o lock.

        Levanta :class:`TreinoConcorrenteError` se outro treino válido já
        estiver rodando (a menos que ``forcar``).
        """
        if self._fd is not None:
            return self

        fd = os.open(str(self.caminho), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                dono = self.dono()
                if forcar:
                    self.registrar_evento("lock_forcado", dono=dono)
                else:
                    raise TreinoConcorrenteError(
                        "Já existe um treinamento em andamento; "
                        f"dono: {dono or 'desconhecido'}",
                        dono,
                    ) from exc
            else:  # pragma: no cover - erro inesperado
                os.close(fd)
                raise

        # Se o flock foi acquired mas o estado registra um dono **vivo** de
        # outro processo, aborta (situação anômala de lock substituído).
        estado = self._ler_estado_arquivo()
        pid_antigo = int(estado.get("pid", 0) or 0)
        if pid_antigo and pid_antigo != os.getpid() and _pid_ativo(pid_antigo):
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            raise TreinoConcorrenteError(
                f"Lock retido por outro processo ativo (PID {pid_antigo})", estado
            )

        self._fd = fd
        self._dono_pid = os.getpid()
        self._modalidade = modalidade
        self._adquirido_em = time.time()
        self._heartbeat = self._adquirido_em
        self._escrever_estado(
            {
                "pid": self._dono_pid,
                "modalidade": modalidade,
                "comando": list(comando or []),
                "inicio": self._adquirido_em,
                "heartbeat": self._heartbeat,
                "start_time": _pid_start_time(self._dono_pid),
                "host": os.uname().nodename,
            }
        )
        self.registrar_evento("lock_adquirido", modalidade=modalidade)
        return self

    def liberar(self) -> None:
        """Libera o lock (se for deste processo)."""
        if self._fd is None:
            return
        try:
            self._escrever_estado(
                {
                    "pid": self._dono_pid,
                    "modalidade": self._modalidade,
                    "liberado_em": time.time(),
                    "estado": "liberado",
                }
            )
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
            self._dono_pid = None
            self.registrar_evento("lock_liberado", modalidade=self._modalidade)

    def heartbeat(self) -> float:
        """Atualiza o *heartbeat* (sinal de vida) e devolve o timestamp."""
        if self._fd is None:
            return 0.0
        self._heartbeat = time.time()
        estado = self._ler_estado_arquivo()
        estado.update({"heartbeat": self._heartbeat, "pid": self._dono_pid})
        try:
            self._escrever_estado(estado)
        except OSError:
            pass
        return self._heartbeat

    def esta_morto(self, limite_s: float = 300.0) -> bool:
        """``True`` se o dono do lock não atualiza o heartbeat há muito tempo."""
        estado = self._ler_estado_arquivo()
        if not estado:
            return False
        if not _pid_ativo(int(estado.get("pid", 0) or 0)):
            return True
        ultimo = float(estado.get("heartbeat", 0.0) or 0.0)
        if ultimo <= 0:
            return False
        return (time.time() - ultimo) > limite_s

    def recuperar_se_morto(self) -> bool:
        """Se o dono morreu, marca o lock como órfão (permite novo treino)."""
        if not self.esta_morto():
            return False
        try:
            self._escrever_estado({"estado": "orfao", "liberado_em": time.time()})
        except OSError:
            pass
        self.registrar_evento("lock_orfao_recuperado")
        return True

    # ------------------------------------------------------------------ #
    # API auxiliar
    # ------------------------------------------------------------------ #
    def prioridade(self, modalidade: str) -> int:
        return int(self.ordem_prioridade.get(modalidade, 99))

    def registrar_evento(self, evento: str, **campos: Any) -> None:
        registro = {"timestamp": time.time(), "evento": evento, **campos}
        log = self.caminho.with_name(self.caminho.stem + ".eventos.jsonl")
        try:
            with log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(registro, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def status(self) -> Dict[str, Any]:
        """Estado atual do lock para relatórios."""
        return {
            "caminho": str(self.caminho),
            "ocupado": self.ocupado(),
            "dono": self.dono(),
            "processo_atual_dono": self._dono_pid == os.getpid() if self._dono_pid else False,
        }

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "TrainingLock":
        if self._fd is None:
            self.adquirir(self._modalidade or "linguagem")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.liberar()


class FilaTreinos:
    """Fila persistente de treinamentos, ordenada por prioridade."""

    def __init__(
        self,
        arquivo: str | Path = "logs/fila_treinos.json",
        prioridades: Optional[PrioridadesTreino] = None,
    ) -> None:
        self.arquivo = Path(arquivo)
        self.prioridades = prioridades or PrioridadesTreino()
        self._itens: List[TreinoSolicitacao] = []
        self._carregar()

    # ------------------------------------------------------------------ #
    def _carregar(self) -> None:
        if not self.arquivo.exists():
            return
        try:
            dados = json.loads(self.arquivo.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        itens = dados.get("itens", dados) if isinstance(dados, dict) else dados
        self._itens = [TreinoSolicitacao.de_dict(i) for i in itens if isinstance(i, dict)]

    def _salvar(self) -> None:
        self.arquivo.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.arquivo.with_suffix(self.arquivo.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {"atualizado_em": time.time(), "itens": [i.para_dict() for i in self._itens]},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, self.arquivo)

    # ------------------------------------------------------------------ #
    def enfileirar(
        self,
        modalidade: str,
        comando: Optional[List[str]] = None,
        metadados: Optional[Dict[str, Any]] = None,
    ) -> TreinoSolicitacao:
        """Adiciona um item à fila respeitando a prioridade."""
        prioridade = int(self.prioridades.ordem.get(modalidade, 99))
        item = TreinoSolicitacao(
            id=uuid.uuid4().hex[:12],
            modalidade=modalidade,
            comando=list(comando or []),
            prioridade=prioridade,
            solicitado_em=time.time(),
            metadados=dict(metadados or {}),
        )
        self._itens.append(item)
        self._ordenar()
        self._salvar()
        return item

    def _ordenar(self) -> None:
        self._itens.sort(key=lambda i: (i.prioridade, i.solicitado_em))

    def proxima(self) -> Optional[TreinoSolicitacao]:
        """Próximo item aguardando (maior prioridade primeiro)."""
        for item in self._itens:
            if item.estado == "aguardando":
                return item
        return None

    def marcar_inicio(self, item_id: str) -> None:
        for item in self._itens:
            if item.id == item_id:
                item.estado = "rodando"
                item.inicio = time.time()
                break
        self._salvar()

    def marcar_fim(self, item_id: str, estado: str = "concluido", erro: Optional[str] = None) -> None:
        for item in self._itens:
            if item.id == item_id:
                item.estado = estado
                item.fim = time.time()
                item.erro = erro
                break
        self._salvar()

    def cancelar(self, item_id: str) -> bool:
        for item in self._itens:
            if item.id == item_id and item.estado == "aguardando":
                item.estado = "cancelado"
                item.fim = time.time()
                self._salvar()
                return True
        return False

    def listar(self) -> List[Dict[str, Any]]:
        return [i.para_dict() for i in self._itens]

    @property
    def pendentes(self) -> List[TreinoSolicitacao]:
        return [i for i in self._itens if i.estado == "aguardando"]

    def remover_concluidos(self, manter: int = 20) -> int:
        """Mantém no máximo ``manter`` itens finalizados (histórico)."""
        finalizados = [i for i in self._itens if i.estado in ("concluido", "falhou", "cancelado")]
        if len(finalizados) <= manter:
            return 0
        remover = finalizados[: len(finalizados) - manter]
        ids = {i.id for i in remover}
        self._itens = [i for i in self._itens if i.id not in ids]
        self._salvar()
        return len(remover)
