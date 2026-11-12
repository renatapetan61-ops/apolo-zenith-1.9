"""Scheduler do ciclo de 10 dias do Apolo Zenith 1.9 (240 horas).

O scheduler **não** é um loop infinito: ele executa um *plano* de fases
definido, mede o tempo realmente gasto, respeita os guardas de recursos e
persiste o estado a cada tarefa em ``training_state.json``.

Garantias:

* apenas **um** treinamento por vez (usa :class:`TrainingLock`);
* cada tarefa roda sob o :class:`ResourceMonitor` (térmico/RAM/disco);
* o estado é gravado **após cada tarefa** — uma queda de energia não perde
  o progresso;
* a duração-alvo de 240 h é uma **meta**: o scheduler para quando o plano
  termina, antes de continuar por conta própria.

Uso::

    python -m orchestrator.scheduler status
    python -m orchestrator.scheduler plan
    python -m orchestrator.scheduler run --fase dia2      # executa uma fase
    python -m orchestrator.scheduler run --todas          # executa o plano
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from resources.config import carregar_config_recursos  # noqa: E402
from resources.resource_monitor import ResourceMonitor  # noqa: E402
from resources.training_lock import TreinoConcorrenteError, TrainingLock  # noqa: E402


# --------------------------------------------------------------------------- #
# Plano de 10 dias
# --------------------------------------------------------------------------- #
@dataclass
class Tarefa:
    """Uma tarefa executável do plano."""

    nome: str
    comando: List[str]
    descricao: str = ""
    modalidade: str = "linguagem"
    duracao_alvo_s: float = 0.0

    def para_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Fase:
    """Uma fase do plano (um dia)."""

    dia: int
    nome: str
    objetivo: str
    tarefas: List[Tarefa] = field(default_factory=list)

    def para_dict(self) -> Dict[str, Any]:
        return {
            "dia": self.dia,
            "nome": self.nome,
            "objetivo": self.objetivo,
            "tarefas": [t.para_dict() for t in self.tarefas],
        }


def _tarefa(nome, comando, descricao="", modalidade="linguagem", duracao=0.0) -> Tarefa:
    return Tarefa(nome, list(comando), descricao, modalidade, duracao)


def plano_padrao() -> List[Fase]:
    """O plano de 10 dias descrito na especificação do projeto."""
    return [
        Fase(
            1,
            "Preparação e validação",
            "Analisar hardware, preparar datasets, validar integridade e tokenizer.",
            [
                _tarefa(
                    "relatorio_hardware",
                    [sys.executable, "-m", "scripts.hardware_report"],
                    "Mede CPU, RAM, disco, bibliotecas e teste de capacidade real.",
                ),
                _tarefa(
                    "download_datasets",
                    [sys.executable, "-m", "scripts.download_datasets", "--datasets",
                     "flickr8k_test", "librispeech_dev_clean", "codigo_github_permissivo",
                     "wikimedia_videos"],
                    "Baixa e verifica (SHA-256) os datasets multimodais.",
                ),
                _tarefa(
                    "preparacao_multimodal",
                    [sys.executable, "-m", "scripts.prepare_multimodal", "--todas"],
                    "Tokeniza o corpus de código e extrai frames/índices.",
                ),
                _tarefa(
                    "testes_de_memoria",
                    [sys.executable, "-m", "pytest", "tests/", "-q"],
                    "Valida a suíte de testes e os guardas de recursos.",
                ),
            ],
        ),
        Fase(
            2,
            "Texto",
            "Treinar o Language Core sobre o corpus textual.",
            [
                _tarefa(
                    "treino_texto",
                    [sys.executable, "-m", "training.text_trainer", "--steps", "700",
                     "--saida", "checkpoints/apolo-lm-continuacao",
                     "--retomar", "checkpoints/apolo-lm-tiny/checkpoints/best"],
                    "Continua o treino do Language Core (retomada do checkpoint existente).",
                    "linguagem",
                )
            ],
        ),
        Fase(
            3,
            "Código",
            "Preparar datasets de programação e treinar a especialização textual.",
            [
                _tarefa(
                    "treino_codigo",
                    [sys.executable, "-m", "training.code_trainer", "--steps", "400",
                     "--dados", "datasets/code/processed", "--saida", "checkpoints/apolo-code"],
                    "Treina o LM sobre o corpus de código tokenizado.",
                    "codigo",
                )
            ],
        ),
        Fase(
            4,
            "Texto e código",
            "Continuar o treino e avaliar a geração textual e de código.",
            [
                _tarefa(
                    "avaliacao_texto",
                    [sys.executable, "-m", "scripts.evaluate", "--checkpoint",
                     "checkpoints/apolo-lm-tiny/checkpoints/best"],
                    "Avalia perplexidade e benchmarks do Language Core.",
                    "linguagem",
                ),
                _tarefa(
                    "avaliacao_codigo",
                    [sys.executable, "-m", "training.code_trainer", "--steps", "200",
                     "--dados", "datasets/code/processed", "--saida", "checkpoints/apolo-code",
                     "--sobrescrever"],
                    "Continua o treino de código e reavalia.",
                    "codigo",
                ),
            ],
        ),
        Fase(
            5,
            "Imagens",
            "Processar imagens e treinar o encoder visual.",
            [
                _tarefa(
                    "treino_visao",
                    [sys.executable, "-m", "training.image_trainer", "--passos", "200",
                     "--saida", "models/vision"],
                    "Treina o ViT compacto + projetor com perda contrastiva.",
                    "visao",
                )
            ],
        ),
        Fase(
            6,
            "Texto e imagens",
            "Treinar a associação imagem-texto e avaliar a recuperação.",
            [
                _tarefa(
                    "avaliacao_visao",
                    [sys.executable, "-m", "scripts.evaluate_multimodal", "--modalidade", "images"],
                    "Mede a recuperação imagem→texto no conjunto de validação.",
                    "visao",
                )
            ],
        ),
        Fase(
            7,
            "Áudio",
            "Processar áudio e treinar o encoder de mel-spectrogram.",
            [
                _tarefa(
                    "treino_audio",
                    [sys.executable, "-m", "training.audio_trainer", "--passos", "200",
                     "--saida", "models/audio"],
                    "Treina o encoder de áudio + projetor (associação áudio-texto).",
                    "audio",
                )
            ],
        ),
        Fase(
            8,
            "Vídeo",
            "Extrair frames e treinar o encoder temporal.",
            [
                _tarefa(
                    "treino_video",
                    [sys.executable, "-m", "training.video_trainer", "--passos", "150",
                     "--saida", "models/video"],
                    "Treina o encoder temporal (ordem temporal + alinhamento).",
                    "video",
                )
            ],
        ),
        Fase(
            9,
            "Integração multimodal",
            "Integrar os módulos treinados e treinar o alinhamento conjunto.",
            [
                _tarefa(
                    "treino_multimodal",
                    [sys.executable, "-m", "training.multimodal_trainer", "--passos", "200",
                     "--saida", "models/multimodal"],
                    "Alinha visão e áudio ao espaço textual do Language Core.",
                    "multimodal",
                )
            ],
        ),
        Fase(
            10,
            "Avaliação final",
            "Avaliar todas as modalidades e gerar o relatório final.",
            [
                _tarefa(
                    "avaliacao_final",
                    [sys.executable, "-m", "scripts.evaluate_multimodal", "--todas"],
                    "Métricas finais por modalidade + relatório.",
                    "multimodal",
                )
            ],
        ),
    ]


# --------------------------------------------------------------------------- #
# Estado persistido
# --------------------------------------------------------------------------- #
@dataclass
class EstadoTreino:
    """Estado do ciclo — serializado em ``training_state.json``."""

    inicio: Optional[float] = None
    fim_previsto: Optional[float] = None
    fim: Optional[float] = None
    duracao_alvo_s: float = 240 * 3600.0
    tempo_executado_s: float = 0.0
    tempo_pausado_s: float = 0.0
    fase_atual: int = 0
    tarefas_concluidas: List[str] = field(default_factory=list)
    tarefas_com_erro: List[Dict[str, Any]] = field(default_factory=list)
    epochs: int = 0
    passos: int = 0
    loss_atual: Optional[float] = None
    lr_atual: Optional[float] = None
    memoria_mb: Optional[float] = None
    temperatura_c: Optional[float] = None
    checkpoints: List[str] = field(default_factory=list)
    metricas: Dict[str, Any] = field(default_factory=dict)
    historico: List[Dict[str, Any]] = field(default_factory=list)
    estado: str = "nao_iniciado"   # nao_iniciado | executando | pausado | concluido | falhou

    def para_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def de_dict(dados: Dict[str, Any]) -> "EstadoTreino":
        validos = set(EstadoTreino.__dataclass_fields__)  # type: ignore[attr-defined]
        return EstadoTreino(**{k: v for k, v in dados.items() if k in validos})


class Scheduler:
    """Executor do plano de 10 dias."""

    def __init__(
        self,
        arquivo_estado: str | Path = "training_state.json",
        config_recursos: str | Path = "configs/resource_manager.yaml",
        raiz: str | Path = ".",
    ) -> None:
        self.raiz = Path(raiz)
        self.arquivo_estado = Path(arquivo_estado)
        if not self.arquivo_estado.is_absolute():
            self.arquivo_estado = self.raiz / self.arquivo_estado
        self.config = carregar_config_recursos(config_recursos)
        self.monitor = ResourceMonitor(self.config, raiz=self.raiz)
        self.plano = plano_padrao()
        self.estado = self._carregar()

    # ------------------------------------------------------------------ #
    def _carregar(self) -> EstadoTreino:
        if self.arquivo_estado.exists():
            try:
                dados = json.loads(self.arquivo_estado.read_text(encoding="utf-8"))
                return EstadoTreino.de_dict(dados)
            except (OSError, json.JSONDecodeError):
                pass
        return EstadoTreino()

    def salvar(self) -> Path:
        """Grava o estado de forma atômica."""
        self.arquivo_estado.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.arquivo_estado.with_suffix(self.arquivo_estado.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.estado.para_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp, self.arquivo_estado)
        return self.arquivo_estado

    # ------------------------------------------------------------------ #
    def status(self) -> Dict[str, Any]:
        """Resumo do estado atual."""
        executado_h = self.estado.tempo_executado_s / 3600.0
        return {
            "estado": self.estado.estado,
            "fase_atual": self.estado.fase_atual,
            "fase_nome": (
                self.plano[self.estado.fase_atual].nome
                if 0 <= self.estado.fase_atual < len(self.plano)
                else "concluído"
            ),
            "inicio": self.estado.inicio,
            "fim_previsto": self.estado.fim_previsto,
            "tempo_executado_h": round(executado_h, 3),
            "tempo_pausado_h": round(self.estado.tempo_pausado_s / 3600.0, 3),
            "meta_h": self.estado.duracao_alvo_s / 3600.0,
            "tarefas_concluidas": len(self.estado.tarefas_concluidas),
            "passos": self.estado.passos,
            "loss_atual": self.estado.loss_atual,
            "lr_atual": self.estado.lr_atual,
            "memoria_mb": self.estado.memoria_mb,
            "temperatura_c": self.estado.temperatura_c,
            "sensor_termico": self.monitor.termico.ler(forcar=True).fonte,
            "tarefas_com_erro": self.estado.tarefas_com_erro,
            "historico": self.estado.historico[-10:],
        }

    # ------------------------------------------------------------------ #
    def executar_tarefa(self, tarefa: Tarefa, log: Path) -> Dict[str, Any]:
        """Executa uma tarefa sob os guardas de recursos."""
        print(f"\n>>> {tarefa.nome}: {tarefa.descricao}")
        print(f"    $ {' '.join(tarefa.comando)}")
        inicio = time.time()
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n\n===== {tarefa.nome} ({time.strftime('%Y-%m-%d %H:%M:%S')}) =====\n")
            fh.flush()
            processo = subprocess.Popen(
                tarefa.comando,
                cwd=str(self.raiz),
                stdout=fh,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            while True:
                codigo = processo.poll()
                if codigo is not None:
                    break
                decisao = self.monitor.verificar()
                self.estado.memoria_mb = (
                    decisao.leitura_memoria.rss_mb if decisao.leitura_memoria else None
                )
                self.estado.temperatura_c = (
                    decisao.leitura_termica.temperatura_c if decisao.leitura_termica else None
                )
                if decisao.deve_pausar:
                    print(f"    [pausa] {decisao.motivo}")
                    self.estado.tempo_pausado_s += 30.0
                    self.estado.estado = "pausado"
                    self.salvar()
                    time.sleep(30.0)
                    self.estado.estado = "executando"
                time.sleep(5.0)
        duracao = time.time() - inicio
        self.estado.tempo_executado_s += duracao
        print(f"    concluído em {duracao / 60:.1f} min (código {codigo})")
        return {"nome": tarefa.nome, "codigo": codigo, "duracao_s": round(duracao, 1)}

    # ------------------------------------------------------------------ #
    def executar(self, fase: Optional[int] = None, ate_fim: bool = False) -> Dict[str, Any]:
        """Executa uma fase (ou todas) do plano."""
        if self.estado.inicio is None:
            self.estado.inicio = time.time()
            self.estado.fim_previsto = self.estado.inicio + self.estado.duracao_alvo_s
        self.estado.estado = "executando"
        self.salvar()

        lock = TrainingLock(self.raiz / self.config.diretorio_lock)
        try:
            lock.adquirir("multimodal")
        except TreinoConcorrenteError as exc:
            print(f"[erro] já existe um treinamento em andamento: {exc}")
            self.estado.estado = "pausado"
            self.salvar()
            return self.status()

        log = self.raiz / "logs" / "scheduler.log"
        try:
            # ``--fase`` é 1-based (1..10), como no plano.
            indice = None if fase is None else fase - 1
            fases = (
                self.plano
                if ate_fim
                else [self.plano[indice]]
                if indice is not None and 0 <= indice < len(self.plano)
                else []
            )
            for f in fases:
                self.estado.fase_atual = f.dia - 1
                self.salvar()
                print(f"\n===== DIA {f.dia}: {f.nome} =====")
                print(f"Objetivo: {f.objetivo}")
                for tarefa in f.tarefas:
                    if tarefa.nome in self.estado.tarefas_concluidas:
                        print(f">>> {tarefa.nome}: já concluída (retomada)")
                        continue
                    resultado = self.executar_tarefa(tarefa, log)
                    if resultado["codigo"] == 0:
                        self.estado.tarefas_concluidas.append(tarefa.nome)
                    else:
                        self.estado.tarefas_com_erro.append(resultado)
                    self.estado.historico.append(
                        {"tarefa": tarefa.nome, "dia": f.dia, **resultado}
                    )
                    self.salvar()
            self.estado.fim = time.time()
            self.estado.estado = "concluido" if not self.estado.tarefas_com_erro else "falhou"
        finally:
            lock.liberar()
            self.salvar()
        return self.status()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduler do ciclo de 10 dias do Apolo")
    sub = parser.add_subparsers(dest="comando", required=True)

    sub.add_parser("status", help="mostra o estado atual do ciclo")
    sub.add_parser("plan", help="mostra o plano de 10 dias")

    p_run = sub.add_parser("run", help="executa uma fase (ou todas)")
    p_run.add_argument("--fase", type=int, default=None, help="número da fase (1..10)")
    p_run.add_argument("--todas", action="store_true", help="executa todas as fases restantes")
    p_run.add_argument("--estado", default="training_state.json")
    args = parser.parse_args()

    scheduler = Scheduler(
        arquivo_estado=getattr(args, "estado", "training_state.json")
    )

    if args.comando == "status":
        if not scheduler.arquivo_estado.exists():
            # Garante a existência do arquivo de estado desde o primeiro uso.
            scheduler.salvar()
        print(json.dumps(scheduler.status(), indent=2, ensure_ascii=False))
        return
    if args.comando == "plan":
        for fase in scheduler.plano:
            print(f"\nDIA {fase.dia}: {fase.nome}")
            print(f"  Objetivo: {fase.objetivo}")
            for tarefa in fase.tarefas:
                print(f"    - {tarefa.nome}: {tarefa.descricao}")
        return
    if args.comando == "run":
        resultado = scheduler.executar(fase=args.fase, ate_fim=args.todas)
        print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
