"""Loop de treinamento do Apolo Zenith 1.9.

Recursos implementados:

* otimizador **AdamW** com decaimento desacoplado;
* **warmup + cosine decay** da taxa de aprendizado;
* **gradient clipping** e **gradient accumulation**;
* **mixed precision** (bf16/fp16) quando suportado pelo hardware;
* **checkpointing** automático com rotação e retenção do melhor modelo;
* **early stopping** com paciência;
* validação periódica, **métricas em JSONL** e retomada de treino;
* semente fixa para reprodutibilidade;
* **gerenciador de recursos**: verificação térmica/RAM/disco a cada passo,
  pausa segura (com checkpoint antes de pausar), **batch adaptativo** com
  reação a OOM e checkpoint com estado completo (RNG, dataset, batch, fila).

As proteções vêm do pacote :mod:`resources` e são configuradas em
``configs/resource_manager.yaml``.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.config import ModelConfig
from core.language_model import ApoloLanguageModel
from resources.adaptive_batch import AdaptiveBatchController
from resources.config import carregar_config_recursos
from resources.erros import registrar_erro
from resources.resource_monitor import ResourceMonitor
from training.adamw import build_optimizer
from training.checkpoint import CheckpointManager
from training.metrics import MetricsTracker, perplexity
from training.scheduler import WarmupCosineScheduler


@dataclass
class TrainingConfig:
    """Hiperparâmetros do loop de treino."""

    saida: str = "checkpoints/apolo-lm-tiny"
    batch_size: int = 16
    block_size: int = 256
    max_steps: int = 1000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    eval_interval: int = 100
    log_interval: int = 10
    save_interval: int = 200
    early_stopping_patience: int = 0     # 0 = desativado
    seed: int = 1337
    device: str = "auto"                 # "auto" | "cpu" | "cuda"
    amp: str = "auto"                    # "auto" | "off" | "bf16" | "fp16"
    num_workers: int = 0
    usar_otimizador_torch: bool = False
    checkpoints_manter: int = 3
    log_arquivo: str = "logs/train.jsonl"
    # --- gerenciamento de recursos ----------------------------------- #
    usar_gerenciador_recursos: bool = True
    aplicar_limites_batch: bool = True
    config_recursos: str = "configs/resource_manager.yaml"
    pausa_maxima_s: float = 3600.0       # tempo máximo de pausa antes de abortar
    intervalo_verificacao_recursos_s: float = 15.0
    max_lotes_validacao: int = 0    # 0 = sem limite (celular: use 40)

    def para_dict(self) -> Dict[str, Any]:
        return asdict(self)


def definir_semente(seed: int) -> None:
    """Fixa as sementes de Python, NumPy e PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolver_dispositivo(escolha: str) -> torch.device:
    if escolha == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(escolha)


def _e_oom(exc: BaseException) -> bool:
    """Distingue um OOM real de outros erros de runtime."""
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    mensagem = str(exc).lower()
    return "out of memory" in mensagem or "cannot allocate" in mensagem


def resolver_amp(escolha: str, device: torch.device) -> tuple[Optional[torch.dtype], bool]:
    """Resolve dtype de autocast e se deve usar GradScaler.

    Em CPU, bf16 é aceito pelo PyTorch, mas em ARM pode ser mais lento que
    fp32; por isso ``auto`` mantém fp32 em CPU.
    """
    if escolha == "off":
        return None, False
    if escolha == "auto":
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16, False
            return torch.float16, True
        return None, False
    if escolha == "bf16":
        return torch.bfloat16, False
    if escolha == "fp16":
        return torch.float16, device.type == "cuda"
    raise ValueError(f"Modo AMP desconhecido: {escolha}")


class Trainer:
    """Treinador do Apolo Language Core."""

    def __init__(
        self,
        model: ApoloLanguageModel,
        config: TrainingConfig,
        dataset_treino: Dataset,
        dataset_validacao: Optional[Dataset] = None,
        tokenizer_dir: Optional[str | Path] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.dataset_treino = dataset_treino
        self.dataset_validacao = dataset_validacao
        self.tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else None

        definir_semente(config.seed)
        self.device = resolver_dispositivo(config.device)
        self.model.to(self.device)

        self.amp_dtype, self.usar_scaler = resolver_amp(config.amp, self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.usar_scaler)

        self.otimizador = build_optimizer(
            self.model,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            use_torch=config.usar_otimizador_torch,
        )
        self.agendador = WarmupCosineScheduler(
            self.otimizador,
            max_steps=config.max_steps,
            warmup_steps=config.warmup_steps,
            min_lr=config.min_lr,
        )

        saida = Path(config.saida)
        saida.mkdir(parents=True, exist_ok=True)
        self.metricas = MetricsTracker(saida / "metricas.jsonl")
        self.checkpoints = CheckpointManager(saida / "checkpoints", manter=config.checkpoints_manter)

        self.passo = 0
        self.melhor_perda_validacao = float("inf")
        self.evaluacoes_sem_melhoria = 0
        self.historico: list[Dict[str, Any]] = []

        # --------------------------------------------------- recursos
        self.recursos: Optional[ResourceMonitor] = None
        self.batch_adaptativo: Optional[AdaptiveBatchController] = None
        self.pausas: list[Dict[str, Any]] = []
        self._ultima_verificacao_recursos = 0.0
        self._pausa_em_andamento = False
        if config.usar_gerenciador_recursos:
            self._configurar_recursos()

    # ------------------------------------------------------------------ #
    # Gerenciamento de recursos
    # ------------------------------------------------------------------ #
    def _configurar_recursos(self) -> None:
        """Ativa o monitor de recursos e aplica os limites de segurança."""
        cfg = carregar_config_recursos(self.config.config_recursos)
        self.recursos = ResourceMonitor(cfg, raiz=Path(self.config.saida).parent)
        self.batch_adaptativo = AdaptiveBatchController(
            cfg.batch, batch_inicial=self.config.batch_size
        )

        if self.config.aplicar_limites_batch:
            original_batch = self.config.batch_size
            original_seq = self.config.block_size
            self.config.batch_size = max(
                cfg.batch.batch_minimo,
                min(self.config.batch_size, cfg.batch.batch_maximo),
            )
            self.config.block_size = max(
                cfg.batch.seq_len_minima,
                min(self.config.block_size, cfg.batch.seq_len_maxima),
            )
            if (
                original_batch != self.config.batch_size
                or original_seq != self.config.block_size
            ):
                print(
                    f"  [recursos] limites aplicados: batch {original_batch} -> "
                    f"{self.config.batch_size}, sequência {original_seq} -> "
                    f"{self.config.block_size} (teto do dispositivo)"
                )

    def _estado_dataset(self) -> Dict[str, Any]:
        """Estado do dataset para o checkpoint (progresso real)."""
        estado: Dict[str, Any] = {}
        for nome, dataset in (
            ("treino", self.dataset_treino),
            ("validacao", self.dataset_validacao),
        ):
            if dataset is None:
                continue
            info: Dict[str, Any] = {"tipo": type(dataset).__name__}
            for atributo in ("num_tokens", "num_sequences", "path", "shards"):
                if hasattr(dataset, atributo):
                    valor = getattr(dataset, atributo)
                    info[atributo] = (
                        str(valor) if not isinstance(valor, (int, float, str, list)) else valor
                    )
            estado[nome] = info
        return estado

    def _estado_treino(self) -> Dict[str, Any]:
        """Contadores do treino para o checkpoint."""
        return {
            "passo": self.passo,
            "melhor_perda_validacao": self.melhor_perda_validacao,
            "evaluacoes_sem_melhoria": self.evaluacoes_sem_melhoria,
            "dispositivo": str(self.device),
            "amp": str(self.amp_dtype),
            "batch_atual": self.batch_adaptativo.batch_atual if self.batch_adaptativo else None,
            "pausas": len(self.pausas),
            "tempo_pausado_s": (
                round(self.recursos.tempo_total_pausado, 1) if self.recursos else 0.0
            ),
        }

    def _salvar_checkpoint(
        self,
        nome: str,
        extra: Optional[Dict[str, Any]] = None,
        metrica: Optional[float] = None,
        metrica_modo: str = "min",
    ) -> None:
        """Salva um checkpoint com o estado completo de retomada."""
        self.checkpoints.salvar(
            nome,
            self.model,
            self.otimizador,
            self.agendador,
            step=self.passo,
            config=self.model.config,
            tokenizer_dir=self.tokenizer_dir,
            extra=extra,
            metrica=metrica,
            metrica_modo=metrica_modo,
            dataset_state=self._estado_dataset(),
            batch_state=self.batch_adaptativo.estado() if self.batch_adaptativo else None,
            estado_treino=self._estado_treino(),
        )

    def _pausar_por_recursos(self, motivo: str) -> bool:
        """Pausa o treino com segurança.

        Gera um checkpoint **antes** de pausar, aguarda o resfriamento e
        devolve ``True`` se o treino pode continuar (ou ``False`` se o tempo
        máximo de pausa foi excedido).
        """
        if self.recursos is None:
            return True
        print(f"  [pausa] {motivo}")
        inicio = time.time()
        try:
            self._salvar_checkpoint(
                f"pausa_{self.passo:06d}",
                extra={"pausa": {"motivo": motivo, "inicio": inicio}},
            )
        except Exception as exc:  # pragma: no cover - defensivo
            print(f"  [pausa] aviso: checkpoint de pausa falhou ({exc})")

        self._pausa_em_andamento = True
        try:
            while True:
                decorrido = time.time() - inicio
                if decorrido > self.config.pausa_maxima_s:
                    print(
                        f"  [pausa] tempo máximo de pausa atingido "
                        f"({decorrido / 60:.1f} min); encerrando com segurança."
                    )
                    self.pausas.append(
                        {"motivo": motivo, "inicio": inicio, "fim": time.time(), "resultado": "abortada"}
                    )
                    return False
                time.sleep(min(self.config.intervalo_verificacao_recursos_s, 30.0))
                decisao = self.recursos.verificar(forcar=True)
                if not decisao.deve_pausar:
                    duracao = time.time() - inicio
                    print(f"  [pausa] retomada após {duracao / 60:.1f} min")
                    self.pausas.append(
                        {"motivo": motivo, "inicio": inicio, "fim": time.time(), "resultado": "retomada"}
                    )
                    return True
        finally:
            self._pausa_em_andamento = False

    # ------------------------------------------------------------------ #
    # Dados
    # ------------------------------------------------------------------ #
    def _carregador(self, dataset: Dataset, shuffle: bool, batch_size: Optional[int] = None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size or self.config.batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=self.config.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

    @property
    def batch_atual(self) -> int:
        """Batch efetivo (adaptativo quando o gerenciador está ativo)."""
        if self.batch_adaptativo is not None:
            return self.batch_adaptativo.batch_atual
        return self.config.batch_size

    # ------------------------------------------------------------------ #
    # Avaliação
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def avaliar(self, dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """Calcula perda e perplexidade em um conjunto de validação."""
        dataset = dataset or self.dataset_validacao
        if dataset is None:
            return {}
        loader = self._carregador(dataset, shuffle=False)
        self.model.eval()
        perda_total = 0.0
        tokens_total = 0
        # A validação é limitada: no celular, percorrer todo o conjunto
        # (dezenas de milhares de sequências) custaria mais que o treino.
        limite = self.config.max_lotes_validacao
        for indice, (x, y) in enumerate(loader):
            if limite and indice >= limite:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                _, loss = self.model(x, y)
            if loss is None:  # pragma: no cover - defesa
                continue
            n = y.numel()
            perda_total += float(loss.item()) * n
            tokens_total += n
        if tokens_total == 0:
            return {}
        perda = perda_total / tokens_total
        return {"val_loss": perda, "val_perplexity": perplexity(perda), "val_tokens": tokens_total}

    # ------------------------------------------------------------------ #
    # Treino
    # ------------------------------------------------------------------ #
    def treinar(self) -> Dict[str, Any]:
        """Executa o loop de treino completo e devolve o resumo."""
        if self.recursos is not None:
            self.recursos.iniciar()

        batch = self.batch_atual
        loader = self._carregador(self.dataset_treino, shuffle=True, batch_size=batch)
        iterador = iter(loader)
        self.model.train()

        inicio = time.time()
        tokens_processados = 0
        historico: list[Dict[str, Any]] = []
        lr_atual = self.agendador.lr_at(self.passo)
        perda_media = float("nan")
        encerrado_por_recursos = False

        while self.passo < self.config.max_steps:
            # ------------------------------------------- guarda de recursos
            if self.recursos is not None and not self._pausa_em_andamento:
                agora = time.time()
                if (agora - self._ultima_verificacao_recursos) >= self.config.intervalo_verificacao_recursos_s:
                    self._ultima_verificacao_recursos = agora
                    decisao = self.recursos.verificar()
                    if decisao.deve_pausar:
                        if not self._pausar_por_recursos(decisao.motivo):
                            encerrado_por_recursos = True
                            break
                        # Retomou: recarrega o loader (o batch pode ter mudado).
                        batch = self.batch_atual
                        loader = self._carregador(self.dataset_treino, shuffle=True, batch_size=batch)
                        iterador = iter(loader)
                        continue

            self.otimizador.zero_grad(set_to_none=True)
            perda_media = 0.0
            norma_media = 0.0

            passo_com_oom = False
            for _ in range(self.config.grad_accum_steps):
                try:
                    x, y = next(iterador)
                except StopIteration:
                    iterador = iter(loader)
                    x, y = next(iterador)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                try:
                    with torch.autocast(
                        device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None
                    ):
                        _, loss = self.model(x, y)
                    if loss is None:  # pragma: no cover - defesa
                        continue
                    perda_media += float(loss.item()) / self.config.grad_accum_steps

                    if self.usar_scaler:
                        self.scaler.scale(loss / self.config.grad_accum_steps).backward()
                    else:
                        (loss / self.config.grad_accum_steps).backward()
                    tokens_processados += int(y.numel())
                except (MemoryError, torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    if not _e_oom(exc):
                        raise
                    # --------------------------------------- recuperação de OOM
                    self.otimizador.zero_grad(set_to_none=True)
                    tokens_processados -= int(y.numel())
                    if self.recursos is not None:
                        self.recursos.memoria.liberar()
                    if self.batch_adaptativo is None:
                        raise
                    try:
                        evento = self.batch_adaptativo.registrar_oom(self.passo + 1, exc)
                    except MemoryError:
                        if self.recursos is not None:
                            self.recursos.falhar(f"OOM repetido: {exc}")
                        registrar_erro("OOM repetido no treino", exc, {"passo": self.passo})
                        raise
                    print(
                        f"  [OOM] {evento.motivo} (batch {evento.batch_anterior} -> {evento.batch_novo})"
                    )
                    batch = self.batch_adaptativo.batch_atual
                    loader = self._carregador(self.dataset_treino, shuffle=True, batch_size=batch)
                    iterador = iter(loader)
                    passo_com_oom = True
                    break

            if passo_com_oom:
                # O passo não consumou: repete com o batch menor.
                continue

            if self.config.grad_clip and self.config.grad_clip > 0:
                if self.usar_scaler:
                    self.scaler.unscale_(self.otimizador)
                norma = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                norma_media = float(norma.item())

            if self.usar_scaler:
                self.scaler.step(self.otimizador)
                self.scaler.update()
            else:
                self.otimizador.step()

            self.passo += 1
            lr_atual = self.agendador.step_update(self.passo)

            # ------------------------------------------- batch adaptativo
            if self.batch_adaptativo is not None:
                ram_disponivel = (
                    self.recursos.memoria.ler().ram_disponivel_mb if self.recursos else None
                )
                evento = self.batch_adaptativo.registrar_sucesso(self.passo, ram_disponivel)
                if evento is not None and evento.batch_novo != self.config.batch_size:
                    batch = self.batch_adaptativo.batch_atual
                    loader = self._carregador(self.dataset_treino, shuffle=True, batch_size=batch)
                    iterador = iter(loader)
                    print(f"  [batch] {evento.motivo}: {evento.batch_anterior} -> {evento.batch_novo}")

            # -------------------------------------------------- logging
            if self.passo % self.config.log_interval == 0 or self.passo == 1:
                decorrido = time.time() - inicio
                tps = tokens_processados / max(decorrido, 1e-6)
                registro = self.metricas.registrar(
                    fase="treino",
                    step=self.passo,
                    perda=perda_media,
                    lr=lr_atual,
                    grad_norm=norma_media,
                    tokens_s=round(tps, 1),
                    memoria_mb=round(
                        torch.cuda.memory_allocated() / 1024**2 if self.device.type == "cuda" else 0.0, 1
                    ),
                )
                historico.append(registro)
                # Log consolidado exigido pelo projeto (logs/training.log).
                try:
                    log_path = Path(self.config.log_arquivo).parent / "training.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a", encoding="utf-8") as fh:
                        fh.write(
                            json.dumps(
                                {
                                    "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "saida": self.config.saida,
                                    **registro,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                except OSError:
                    pass
                print(
                    f"  passo {self.passo:>5}/{self.config.max_steps} | perda {perda_media:.4f} | "
                    f"lr {lr_atual:.2e} | {tps:.0f} tok/s | elapsed {decorrido:.0f}s"
                )

            # -------------------------------------------------- validação
            if self.passo % self.config.eval_interval == 0 or self.passo == self.config.max_steps:
                resultado = self.avaliar()
                if resultado:
                    print(
                        f"  [validação] passo {self.passo} | perda {resultado['val_loss']:.4f} | "
                        f"ppl {resultado['val_perplexity']:.2f}"
                    )
                    self.metricas.registrar(fase="validacao", step=self.passo, **resultado)
                    self.historico.append(resultado)
                    if resultado["val_loss"] < self.melhor_perda_validacao - 1e-4:
                        self.melhor_perda_validacao = resultado["val_loss"]
                        self.evaluacoes_sem_melhoria = 0
                        self._salvar_checkpoint(
                            "best",
                            extra={"validacao": resultado},
                            metrica=resultado["val_loss"],
                            metrica_modo="min",
                        )
                    else:
                        self.evaluacoes_sem_melhoria += 1
                        if (
                            self.config.early_stopping_patience > 0
                            and self.evaluacoes_sem_melhoria >= self.config.early_stopping_patience
                        ):
                            print(
                                f"  [early stopping] sem melhora por {self.evaluacoes_sem_melhoria} "
                                "avaliações; encerrando."
                            )
                            break

            # -------------------------------------------------- checkpoint
            if self.config.save_interval and self.passo % self.config.save_interval == 0:
                self._salvar_checkpoint(
                    f"step_{self.passo:06d}",
                    extra={"treino": {"perda": perda_media, "lr": lr_atual}},
                )

        # Checkpoint final
        self._salvar_checkpoint(
            "final",
            extra={"treino": {"perda": perda_media}},
        )
        if self.recursos is not None:
            if encerrado_por_recursos:
                self.recursos.falhar("pausa máxima de recursos excedida")
            else:
                self.recursos.concluir()
        # Configuração do experimento (reprodutibilidade)
        Path(self.config.saida).mkdir(parents=True, exist_ok=True)
        (Path(self.config.saida) / "config_treino.json").write_text(
            json.dumps(self.config.para_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

        resumo = {
            "passos": self.passo,
            "melhor_perda_validacao": self.melhor_perda_validacao,
            "tempo_s": round(time.time() - inicio, 2),
            "tokens_processados": tokens_processados,
            "tokens_por_segundo": round(tokens_processados / max(time.time() - inicio, 1e-6), 1),
            "device": str(self.device),
            "amp": str(self.amp_dtype),
            "parametros": self.model.num_parameters(),
            "batch_final": self.batch_atual,
            "pausas": self.pausas,
            "tempo_pausado_s": (
                round(self.recursos.tempo_total_pausado, 1) if self.recursos else 0.0
            ),
            "encerrado_por_recursos": encerrado_por_recursos,
            "historico": historico[-20:],
        }
        print(json.dumps({k: v for k, v in resumo.items() if k != "historico"}, indent=2, ensure_ascii=False))
        return resumo

    # ------------------------------------------------------------------ #
    # Retomada
    # ------------------------------------------------------------------ #
    def retomar(self, checkpoint_dir: str | Path, restaurar_rng: bool = True) -> Dict[str, Any]:
        """Retoma o treino a partir de um checkpoint existente.

        Restaura pesos, otimizador, agendador, passo, melhor perda de
        validação, **estado do batch adaptativo** e (opcionalmente) o estado
        do RNG, tornando a retomada reprodutível.
        """
        from training.checkpoint import load_checkpoint

        meta = load_checkpoint(
            checkpoint_dir,
            map_location=self.device,
            model=self.model,
            optimizer=self.otimizador,
            scheduler=self.agendador,
            restaurar_rng=restaurar_rng,
        )
        self.passo = int(meta.get("step", 0))
        val = meta.get("validacao") or {}
        if "val_loss" in val:
            self.melhor_perda_validacao = float(val["val_loss"])
        estado = meta.get("estado_treino") or {}
        if "evaluacoes_sem_melhoria" in estado:
            self.evaluacoes_sem_melhoria = int(estado["evaluacoes_sem_melhoria"])

        # Estado do batch adaptativo (para retomar com o batch correto).
        estado_batch = meta.get("batch_state") or {}
        if self.batch_adaptativo is not None and estado_batch:
            self.batch_adaptativo.reset()
            atual = int(estado_batch.get("batch_atual", self.batch_atual) or self.batch_atual)
            self.batch_adaptativo._batch = max(
                self.batch_adaptativo.limites.batch_minimo,
                min(atual, self.batch_adaptativo.limites.batch_maximo),
            )

        print(
            f"  [retomada] checkpoint carregado: passo {self.passo} | "
            f"batch {self.batch_atual} | rng restaurado: {meta.get('rng_restaurado', False)}"
        )
        return meta
