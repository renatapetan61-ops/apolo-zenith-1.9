"""Gerenciador de jobs de treinamento do servidor Apolo.

Cada job é um processo do sistema operacional independente (normalmente
``python -m scripts.train ...``), com log próprio, status rastreado e
possibilidade de cancelamento.  O processo do servidor **nunca** bloqueia
esperando pelo treino: o mobile apenas consulta o status.

**Regra de segurança:** apenas **um** treinamento pode existir por vez.  O
gerenciador adquire o :class:`~resources.training_lock.TrainingLock` antes de
lançar o processo e o mantém retido até o job terminar (o ``flock`` é
liberado automaticamente pelo kernel se o servidor cair).
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from resources.training_lock import TreinoConcorrenteError, TrainingLock


@dataclass
class Job:
    """Estado de um job de treinamento."""

    id: str
    nome: str
    comando: List[str]
    log: str
    status: str = "pendente"          # pendente | rodando | concluido | falhou | cancelado
    inicio: Optional[float] = None
    fim: Optional[float] = None
    codigo_saida: Optional[int] = None
    pid: Optional[int] = None
    erro: Optional[str] = None
    processo: Optional[subprocess.Popen] = field(default=None, repr=False, compare=False)

    def para_dict(self) -> Dict:
        """Representação serializável do job (sem o handle do processo)."""
        return {
            "id": self.id,
            "nome": self.nome,
            "comando": list(self.comando),
            "log": self.log,
            "status": self.status,
            "inicio": self.inicio,
            "fim": self.fim,
            "codigo_saida": self.codigo_saida,
            "pid": self.pid,
            "erro": self.erro,
            "duracao_s": (
                round((self.fim or time.time()) - self.inicio, 2) if self.inicio else None
            ),
        }


class JobManager:
    """Gerencia jobs de treino do servidor (um por vez)."""

    def __init__(
        self,
        pasta: str | Path = "logs/jobs",
        caminho_lock: str | Path = "logs/apolo_training.lock",
        usar_lock: bool = True,
    ) -> None:
        self.pasta = Path(pasta)
        self.pasta.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, Job] = {}
        self._arquivo_estado = self.pasta / "jobs.json"
        self.usar_lock = usar_lock
        self.lock = TrainingLock(caminho_lock) if usar_lock else None
        self._carregar()

    # ------------------------------------------------------------------ #
    # Persistência simples do estado (o processo em si é gerenciado pelo SO)
    # ------------------------------------------------------------------ #
    def _carregar(self) -> None:
        if self._arquivo_estado.exists():
            try:
                dados = json.loads(self._arquivo_estado.read_text(encoding="utf-8"))
                campos = set(Job.__dataclass_fields__)  # type: ignore[attr-defined]
                for item in dados:
                    filtrado = {k: v for k, v in item.items() if k in campos}
                    job = Job(**filtrado)
                    self.jobs[job.id] = job
            except (json.JSONDecodeError, TypeError):
                pass

    def _salvar(self) -> None:
        self._arquivo_estado.write_text(
            json.dumps([j.para_dict() for j in self.jobs.values()], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    # Ciclo de vida
    # ------------------------------------------------------------------ #
    def iniciar(
        self,
        nome: str,
        comando: List[str] | str,
        cwd: Optional[str] = None,
        modalidade: str = "linguagem",
    ) -> Job:
        """Inicia um job de treino em background.

        Adquire o lock de treinamento: se já existir um treino ativo, levanta
        :class:`~resources.training_lock.TreinoConcorrenteError` e **nenhum**
        processo é criado.
        """
        if isinstance(comando, str):
            comando = shlex.split(comando)

        if self.lock is not None:
            try:
                self.lock.adquirir(modalidade, comando=list(comando))
            except TreinoConcorrenteError as exc:
                job = Job(
                    id=uuid.uuid4().hex[:12],
                    nome=nome,
                    comando=list(comando),
                    log="",
                    status="falhou",
                    inicio=time.time(),
                    fim=time.time(),
                    codigo_saida=1,
                    erro=str(exc),
                )
                self.jobs[job.id] = job
                self._salvar()
                raise

        job_id = uuid.uuid4().hex[:12]
        log_path = self.pasta / f"{job_id}.log"
        log = log_path.open("w", encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        processo = subprocess.Popen(
            comando,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        job = Job(
            id=job_id,
            nome=nome,
            comando=list(comando),
            log=str(log_path),
            status="rodando",
            inicio=time.time(),
            pid=processo.pid,
            processo=processo,
        )
        self.jobs[job_id] = job
        self._salvar()
        return job

    def atualizar(self, job_id: str) -> Optional[Job]:
        """Atualiza o status de um job consultando o processo do SO."""
        job = self.jobs.get(job_id)
        if job is None or job.processo is None:
            return job
        codigo = job.processo.poll()
        if codigo is None:
            return job
        job.codigo_saida = codigo
        job.fim = time.time()
        job.status = "concluido" if codigo == 0 else "falhou"
        job.processo = None
        # O job terminou: libera o lock de treinamento.
        if self.lock is not None:
            self.lock.liberar()
        self._salvar()
        return job

    def cancelar(self, job_id: str) -> bool:
        """Cancela um job em execução (SIGTERM no grupo de processos)."""
        job = self.jobs.get(job_id)
        if job is None or job.processo is None:
            return False
        try:
            os.killpg(os.getpgid(job.processo.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                job.processo.terminate()
            except OSError:
                return False
        job.status = "cancelado"
        job.fim = time.time()
        if self.lock is not None:
            self.lock.liberar()
        self._salvar()
        return True

    def listar(self) -> List[Dict]:
        """Lista todos os jobs com status atualizado."""
        for job_id in list(self.jobs):
            self.atualizar(job_id)
        return [job.para_dict() for job in self.jobs.values()]

    def obter(self, job_id: str) -> Optional[Job]:
        return self.atualizar(job_id)

    def logs(self, job_id: str, linhas: int = 200) -> str:
        """Retorna as últimas linhas do log de um job."""
        job = self.jobs.get(job_id)
        if job is None:
            return ""
        caminho = Path(job.log)
        if not caminho.exists():
            return ""
        try:
            with caminho.open("r", encoding="utf-8", errors="replace") as fh:
                return "".join(fh.readlines()[-linhas:])
        except OSError:
            return ""
