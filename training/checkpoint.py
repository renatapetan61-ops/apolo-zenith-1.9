"""Checkpointing do Apolo: gravação atômica, retomada e rotação de versões.

Cada checkpoint é um diretório contendo:

* ``model.pt``      – pesos do modelo (state_dict);
* ``optimizer.pt``  – estado do AdamW (momentos, contadores);
* ``scheduler.pt``  – estado do agendador de LR;
* ``meta.json``     – metadados (passo, época, configuração, métricas);
* ``tokenizer/``    – cópia do tokenizer usado no treino (opcional).

Além dos arquivos, o ``meta.json`` guarda o **estado completo** necessário
para retomar o treino exatamente de onde parou:

* ``rng_state``     – estado do RNG (Python, NumPy, PyTorch CPU/CUDA);
* ``dataset_state`` – progresso do dataset (offsets, shards, índices);
* ``batch_state``   – batch atual, cooldown e histórico de OOM;
* ``fila_state``    – estado da fila de treinamentos;
* ``estado_treino`` – contadores (época, tokens, tempo), erros e configuração.

A gravação é atômica **e validada**: os arquivos são escritos em um diretório
temporário, verificados e só então movidos para o destino final.  O
checkpoint anterior é preservado até a troca ser concluída, de modo que uma
interrupção nunca deixe o treino sem checkpoint válido.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from core.config import ModelConfig

#: Nomes de checkpoints que a rotação automática nunca remove.
CHECKPOINTS_PROTEGIDOS = ("best", "final", "last")


# --------------------------------------------------------------------------- #
# Estado do RNG
# --------------------------------------------------------------------------- #
def capturar_estado_rng() -> Dict[str, Any]:
    """Captura o estado dos geradores aleatórios (reprodutibilidade)."""
    estado: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():  # pragma: no cover - sem GPU no alvo
        estado["torch_cuda"] = torch.cuda.get_rng_state_all()
    return estado


def _serializar_estado_rng(estado: Dict[str, Any]) -> Dict[str, Any]:
    """Converte o estado do RNG para uma estrutura serializável em JSON."""
    import base64

    saida: Dict[str, Any] = {}

    # Python: (versão, tupla interna, gauss_next)
    estado_python = estado.get("python")
    if estado_python is not None:
        versao, interno, gauss = estado_python
        saida["python"] = {
            "versao": versao,
            "interno": list(interno),
            "gauss_next": gauss,
        }

    # NumPy: ('MT19937', uint32[624], pos, has_gauss, cached_gaussian)
    estado_numpy = estado.get("numpy")
    if estado_numpy is not None:
        algoritmo, palavras, pos, tem_gauss, gauss_cache = estado_numpy
        saida["numpy"] = {
            "algoritmo": algoritmo,
            "palavras_b64": base64.b64encode(palavras.tobytes()).decode("ascii"),
            "pos": int(pos),
            "tem_gauss": int(tem_gauss),
            "cached_gaussian": float(gauss_cache),
        }

    # PyTorch (CPU)
    estado_torch = estado.get("torch_cpu")
    if estado_torch is not None:
        tensor = torch.as_tensor(estado_torch, dtype=torch.uint8)
        saida["torch_cpu_b64"] = base64.b64encode(tensor.numpy().tobytes()).decode("ascii")

    estado_cuda = estado.get("torch_cuda")
    if estado_cuda is not None:  # pragma: no cover - sem GPU no alvo
        partes = []
        for tensor in estado_cuda:
            t = torch.as_tensor(tensor, dtype=torch.uint8)
            partes.append(base64.b64encode(t.numpy().tobytes()).decode("ascii"))
        saida["torch_cuda_b64"] = partes
    return saida


def _deserializar_estado_rng(dados: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstrói o estado do RNG a partir da forma serializada."""
    import base64

    estado: Dict[str, Any] = {}

    if "python" in dados:
        bruto = dados["python"]
        try:
            estado["python"] = (
                int(bruto["versao"]),
                tuple(int(v) for v in bruto["interno"]),
                bruto.get("gauss_next"),
            )
        except (KeyError, TypeError, ValueError):
            pass

    if "numpy" in dados:
        bruto = dados["numpy"]
        try:
            palavras = np.frombuffer(
                base64.b64decode(bruto["palavras_b64"]), dtype=np.uint32
            ).copy()
            estado["numpy"] = (
                bruto["algoritmo"],
                palavras,
                int(bruto["pos"]),
                int(bruto["tem_gauss"]),
                float(bruto["cached_gaussian"]),
            )
        except (KeyError, TypeError, ValueError):
            pass

    if "torch_cpu_b64" in dados:
        try:
            dados_brutos = base64.b64decode(dados["torch_cpu_b64"])
            estado["torch_cpu"] = torch.frombuffer(
                bytearray(dados_brutos), dtype=torch.uint8
            ).clone()
        except (TypeError, ValueError):
            pass

    if "torch_cuda_b64" in dados and torch.cuda.is_available():  # pragma: no cover
        partes = []
        for parte in dados["torch_cuda_b64"]:
            buffer = bytearray(base64.b64decode(parte))
            partes.append(torch.frombuffer(buffer, dtype=torch.uint8).clone())
        estado["torch_cuda"] = partes
    return estado


def restaurar_estado_rng(estado: Optional[Dict[str, Any]]) -> bool:
    """Restaura o estado dos geradores aleatórios.  Devolve ``True`` em caso
    de sucesso."""
    if not estado:
        return False
    # A forma serializada em JSON usa "torch_cpu_b64" e "python" como dict.
    serializado = (
        "torch_cpu_b64" in estado
        or isinstance(estado.get("python"), dict)
        or isinstance(estado.get("numpy"), dict)
    )
    if serializado:
        estado = _deserializar_estado_rng(estado)
    try:
        if "python" in estado:
            random.setstate(estado["python"])
        if "numpy" in estado:
            np.random.set_state(estado["numpy"])
        if "torch_cpu" in estado:
            torch.set_rng_state(torch.as_tensor(estado["torch_cpu"], dtype=torch.uint8))
        if "torch_cuda" in estado and torch.cuda.is_available():  # pragma: no cover
            torch.cuda.set_rng_state_all(estado["torch_cuda"])
        return True
    except (ValueError, TypeError, KeyError, RuntimeError):
        return False


# --------------------------------------------------------------------------- #
# Gravar / carregar
# --------------------------------------------------------------------------- #
def save_checkpoint(
    diretorio: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    step: int = 0,
    epoch: int = 0,
    config: Optional[ModelConfig] = None,
    tokenizer_dir: Optional[str | Path] = None,
    extra: Optional[Dict[str, Any]] = None,
    dataset_state: Optional[Dict[str, Any]] = None,
    batch_state: Optional[Dict[str, Any]] = None,
    fila_state: Optional[Dict[str, Any]] = None,
    estado_treino: Optional[Dict[str, Any]] = None,
    rng_state: Optional[Dict[str, Any]] = None,
    capturar_rng: bool = True,
) -> Path:
    """Salva um checkpoint completo de forma atômica e validada.

    O checkpoint existente só é substituído **depois** que o novo foi
    gravado e verificado, mantendo sempre uma cópia válida.
    """
    destino = Path(diretorio)
    temporario = destino.with_name(destino.name + ".tmp")
    backup = destino.with_name(destino.name + ".old")
    if temporario.exists():
        shutil.rmtree(temporario)
    temporario.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), temporario / "model.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), temporario / "optimizer.pt")
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        torch.save(scheduler.state_dict(), temporario / "scheduler.pt")

    meta: Dict[str, Any] = {
        "step": step,
        "epoch": epoch,
        "timestamp": time.time(),
        "torch_version": torch.__version__,
    }
    if config is not None:
        meta["model_config"] = config.to_dict()
        # Grava também config.json para que o checkpoint seja carregável
        # diretamente por ApoloLanguageModel.load().
        (temporario / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if extra:
        meta.update(extra)
    if dataset_state is not None:
        meta["dataset_state"] = dataset_state
    if batch_state is not None:
        meta["batch_state"] = batch_state
    if fila_state is not None:
        meta["fila_state"] = fila_state
    if estado_treino is not None:
        meta["estado_treino"] = estado_treino
    if capturar_rng:
        bruto = rng_state if rng_state is not None else capturar_estado_rng()
        meta["rng_state"] = _serializar_estado_rng(bruto)

    (temporario / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if tokenizer_dir is not None and Path(tokenizer_dir).exists():
        shutil.copytree(tokenizer_dir, temporario / "tokenizer", dirs_exist_ok=True)

    # ---- validação da gravação antes de substituir o checkpoint válido ----
    _validar_checkpoint(temporario)

    if destino.exists():
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        os.replace(destino, backup)
    try:
        os.replace(temporario, destino)
    except OSError:
        # Restaura o checkpoint anterior se a troca falhar.
        if backup.exists() and not destino.exists():
            os.replace(backup, destino)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    return destino


def _validar_checkpoint(diretorio: Path) -> None:
    """Verifica que o checkpoint recém-gravado está completo e legível."""
    modelo = diretorio / "model.pt"
    if not modelo.exists() or modelo.stat().st_size == 0:
        raise IOError(f"Checkpoint inválido: {modelo} ausente ou vazio")
    meta_path = diretorio / "meta.json"
    if not meta_path.exists():
        raise IOError(f"Checkpoint inválido: {meta_path} ausente")
    json.loads(meta_path.read_text(encoding="utf-8"))
    # Carrega os pesos para garantir que não estão truncados.
    torch.load(modelo, map_location="cpu", weights_only=True)


def load_checkpoint(
    diretorio: str | Path,
    map_location: str | torch.device = "cpu",
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    restaurar_rng: bool = False,
) -> Dict[str, Any]:
    """Carrega um checkpoint.  Se ``model``/``optimizer`` forem fornecidos,
    o estado é aplicado neles e a retomada de treino fica pronta.

    Com ``restaurar_rng=True`` o estado do RNG é reaplicado, tornando a
    retomada **bit a bit** reprodutível.
    """
    diretorio = Path(diretorio)
    if not diretorio.exists():
        raise FileNotFoundError(f"Checkpoint inexistente: {diretorio}")

    meta_path = diretorio / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    if model is not None:
        estado = torch.load(diretorio / "model.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(estado)
    if optimizer is not None and (diretorio / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(diretorio / "optimizer.pt", map_location=map_location, weights_only=False))
    if scheduler is not None and hasattr(scheduler, "load_state_dict") and (diretorio / "scheduler.pt").exists():
        scheduler.load_state_dict(torch.load(diretorio / "scheduler.pt", map_location=map_location, weights_only=False))
    if restaurar_rng:
        meta["rng_restaurado"] = restaurar_estado_rng(meta.get("rng_state"))
    return meta


class CheckpointManager:
    """Mantém os últimos N checkpoints e o melhor checkpoint por métrica.

    Garantias:

    * ``best``/``final``/``last`` **nunca** são removidos pela rotação;
    * a cópia do melhor checkpoint só é substituída **depois** que o
      checkpoint de origem foi gravado e validado;
    * um checkpoint válido existente nunca é apagado sem substituto pronto.
    """

    def __init__(self, diretorio: str | Path, manter: int = 3, nome_melhor: str = "best") -> None:
        self.diretorio = Path(diretorio)
        self.diretorio.mkdir(parents=True, exist_ok=True)
        self.manter = max(1, manter)
        self.nome_melhor = nome_melhor
        self.melhor_valor: Optional[float] = None

    def salvar(
        self,
        nome: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        step: int = 0,
        epoch: int = 0,
        config: Optional[ModelConfig] = None,
        tokenizer_dir: Optional[str | Path] = None,
        extra: Optional[Dict[str, Any]] = None,
        metrica: Optional[float] = None,
        metrica_modo: str = "min",
        dataset_state: Optional[Dict[str, Any]] = None,
        batch_state: Optional[Dict[str, Any]] = None,
        fila_state: Optional[Dict[str, Any]] = None,
        estado_treino: Optional[Dict[str, Any]] = None,
        rng_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        caminho = self.diretorio / nome
        save_checkpoint(
            caminho,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            epoch=epoch,
            config=config,
            tokenizer_dir=tokenizer_dir,
            extra=extra,
            dataset_state=dataset_state,
            batch_state=batch_state,
            fila_state=fila_state,
            estado_treino=estado_treino,
            rng_state=rng_state,
        )
        if metrica is not None:
            melhor = (
                self.melhor_valor is None
                or (metrica_modo == "min" and metrica < self.melhor_valor)
                or (metrica_modo == "max" and metrica > self.melhor_valor)
            )
            if melhor:
                self.melhor_valor = metrica
                destino = self.diretorio / self.nome_melhor
                if destino.resolve() != caminho.resolve():
                    # Copia para um temporário e troca: o "best" anterior só
                    # desaparece quando a nova cópia está completa.
                    temporario = destino.with_name(destino.name + ".copytmp")
                    if temporario.exists():
                        shutil.rmtree(temporario)
                    shutil.copytree(caminho, temporario)
                    backup = destino.with_name(destino.name + ".old")
                    if destino.exists():
                        shutil.rmtree(backup, ignore_errors=True)
                        os.replace(destino, backup)
                    os.replace(temporario, destino)
                    shutil.rmtree(backup, ignore_errors=True)
        self._rotacionar()
        return caminho

    def _rotacionar(self) -> None:
        """Remove checkpoints antigos mantendo os ``manter`` mais recentes.

        Nomes protegidos (``best``, ``final``, ``last``) nunca são removidos.
        """
        candidatos = sorted(
            [
                p
                for p in self.diretorio.glob("step_*")
                if p.is_dir() and p.name not in CHECKPOINTS_PROTEGIDOS
            ],
            key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 0,
        )
        for antigo in candidatos[:-self.manter]:
            shutil.rmtree(antigo, ignore_errors=True)

    def listar(self) -> list[Dict[str, Any]]:
        """Lista os checkpoints disponíveis com seus metadados."""
        saida = []
        for p in sorted(self.diretorio.glob("*")):
            if not p.is_dir():
                continue
            meta_path = p / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            saida.append({"nome": p.name, "caminho": str(p), "step": meta.get("step"), "epoch": meta.get("epoch")})
        return saida
