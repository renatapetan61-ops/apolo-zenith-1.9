"""Detecção de hardware do servidor de treinamento do Apolo.

Este módulo é a base para *não* assumir capacidades inexistentes: o servidor
externo promete entre 100 TB e 400 TB de armazenamento, mas armazenamento não
implica RAM, VRAM ou GPUs.  Tudo aqui é medido em tempo real.

Uso::

    python -m server.hardware_detection --json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class InfoCPU:
    modelo: str = "desconhecido"
    nucleos_logicos: int = 0
    nucleos_fisicos: Optional[int] = None
    frequencia_mhz: Optional[float] = None
    arquitetura: str = ""


@dataclass
class InfoMemoria:
    total_gb: float = 0.0
    disponivel_gb: float = 0.0
    usada_gb: float = 0.0
    swap_total_gb: float = 0.0


@dataclass
class InfoDisco:
    caminho: str = "/"
    total_gb: float = 0.0
    usado_gb: float = 0.0
    livre_gb: float = 0.0


@dataclass
class InfoGPU:
    nome: str = ""
    vram_gb: float = 0.0
    total_vram_gb: float = 0.0
    driver: str = ""
    cuda: str = ""
    suporta_bf16: bool = False


@dataclass
class InfoRede:
    host: str = ""
    testado: bool = False
    mbps_download: Optional[float] = None
    latencia_ms: Optional[float] = None
    observacao: str = ""


@dataclass
class RelatorioHardware:
    """Relatório completo de capacidade do host."""

    cpu: InfoCPU = field(default_factory=InfoCPU)
    memoria: InfoMemoria = field(default_factory=InfoMemoria)
    discos: List[InfoDisco] = field(default_factory=list)
    gpus: List[InfoGPU] = field(default_factory=list)
    rede: InfoRede = field(default_factory=InfoRede)
    torch: str = ""
    cuda_disponivel: bool = False
    plataforma: str = ""
    python: str = ""

    def para_dict(self) -> Dict:
        dados = asdict(self)
        dados["total_vram_gb"] = sum(g.vram_gb for g in self.gpus)
        return dados


# --------------------------------------------------------------------------- #
# Coletas
# --------------------------------------------------------------------------- #
def detectar_cpu() -> InfoCPU:
    info = InfoCPU()
    info.nucleos_logicos = os.cpu_count() or 1
    info.arquitetura = platform.machine()
    try:
        texto = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        implementadores: set[str] = set()
        ids_fisicos: set[str] = set()
        for linha in texto.splitlines():
            chave, _, valor = linha.partition(":")
            chave = chave.strip().lower()
            valor = valor.strip()
            if chave in {"model name", "cpu model", "hardware", "processor", "cpu"} and valor:
                if chave == "processor" and valor.isdigit():
                    continue
                if info.modelo == "desconhecido":
                    info.modelo = valor
            if chave == "cpu implementer" and valor:
                implementadores.add(valor)
            if chave in {"cpu mhz", "clock"} and info.frequencia_mhz is None:
                try:
                    info.frequencia_mhz = float(valor)
                except ValueError:
                    pass
            if chave in {"physical id", "core id"} and valor:
                ids_fisicos.add(valor)
        if info.modelo == "desconhecido" and implementadores:
            info.modelo = f"CPU ARM (implementer {', '.join(sorted(implementadores))})"
        if ids_fisicos:
            info.nucleos_fisicos = len(ids_fisicos)
    except OSError:
        pass
    return info


def detectar_memoria() -> InfoMemoria:
    info = InfoMemoria()
    try:
        texto = Path("/proc/meminfo").read_text(encoding="utf-8")
        valores = {}
        for linha in texto.splitlines():
            chave, _, resto = linha.partition(":")
            partes = resto.strip().split()
            if partes and partes[0].replace("kB", "").isdigit():
                valores[chave] = int(partes[0].replace("kB", "")) / (1024**2)
        info.total_gb = valores.get("MemTotal", 0.0)
        info.disponivel_gb = valores.get("MemAvailable", valores.get("MemFree", 0.0))
        info.usada_gb = info.total_gb - info.disponivel_gb
        info.swap_total_gb = valores.get("SwapTotal", 0.0)
    except OSError:
        pass
    return info


def detectar_discos(caminhos: Optional[List[str]] = None) -> List[InfoDisco]:
    caminhos = caminhos or ["/", "/root", "/data"]
    discos: List[InfoDisco] = []
    vistos = set()
    for caminho in caminhos:
        try:
            uso = shutil.disk_usage(caminho)
        except OSError:
            continue
        chave = (uso.total, uso.free)
        if chave in vistos:
            continue
        vistos.add(chave)
        discos.append(
            InfoDisco(
                caminho=caminho,
                total_gb=uso.total / (1024**3),
                usado_gb=uso.used / (1024**3),
                livre_gb=uso.free / (1024**3),
            )
        )
    return discos


def detectar_gpus() -> List[InfoGPU]:
    """Detecta GPUs via PyTorch e via ``nvidia-smi`` (fallback)."""
    gpus: List[InfoGPU] = []
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append(
                    InfoGPU(
                        nome=props.name,
                        vram_gb=props.total_memory / (1024**3),
                        total_vram_gb=props.total_memory / (1024**3),
                        suporta_bf16=bool(getattr(props, "major", 0) >= 8),
                    )
                )
            gpus[0].cuda = torch.version.cuda or ""
    except Exception:  # pragma: no cover - torch ausente
        pass

    if not gpus:
        try:
            saida = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if saida.returncode == 0:
                for linha in saida.stdout.strip().splitlines():
                    partes = [p.strip() for p in linha.split(",")]
                    if len(partes) >= 3:
                        gpus.append(
                            InfoGPU(
                                nome=partes[0],
                                vram_gb=float(partes[1]) / 1024,
                                total_vram_gb=float(partes[1]) / 1024,
                                driver=partes[2],
                            )
                        )
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return gpus


def detectar_rede(testar: bool = False, url: str = "https://www.gutenberg.org/cache/epub/43627/pg43627.txt") -> InfoRede:
    """Coleta informações de rede (e mede vazão se ``testar=True``)."""
    import socket
    import urllib.request

    info = InfoRede()
    try:
        info.host = socket.gethostname()
    except OSError:
        info.host = "desconhecido"

    if not testar:
        info.observacao = "medição de vazão desativada (use --testar-rede)"
        return info

    inicio = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ApoloZenith/1.9"})
        with urllib.request.urlopen(req, timeout=20) as resposta:
            dados = resposta.read(1 << 20)
        decorrido = max(time.time() - inicio, 1e-6)
        info.tested = True
        info.mbps_download = round(len(dados) / decorrido / 1024 / 1024, 3)
        info.latencia_ms = round(decorrido * 1000, 1)
        info.observacao = f"download de {len(dados)/1024:.0f} KB em {decorrido:.2f}s"
    except Exception as exc:  # pragma: no cover - rede pode falhar
        info.observacao = f"falha na medição: {exc}"
    return info


def coletar_relatorio(testar_rede: bool = False) -> RelatorioHardware:
    """Executa a coleta completa de hardware."""
    import torch

    relatorio = RelatorioHardware()
    relatorio.cpu = detectar_cpu()
    relatorio.memoria = detectar_memoria()
    relatorio.discos = detectar_discos()
    relatorio.gpus = detectar_gpus()
    relatorio.rede = detectar_rede(testar=testar_rede)
    relatorio.torch = torch.__version__
    relatorio.cuda_disponivel = torch.cuda.is_available()
    relatorio.plataforma = platform.platform()
    relatorio.python = platform.python_version()
    return relatorio


# --------------------------------------------------------------------------- #
# Estimativas
# --------------------------------------------------------------------------- #
def estimar_memoria_treino(
    n_parametros: int,
    bytes_param: int = 4,
    bytes_grad: int = 4,
    bytes_estado_adamw: int = 8,
    ativa_bytes_por_token: int = 0,
    tokens_por_passo: int = 0,
) -> Dict[str, float]:
    """Estima a memória necessária para treinar um modelo.

    Returns dicionário com GB de pesos, gradientes, otimizador, ativações e total.
    """
    pesos = n_parametros * bytes_param
    gradientes = n_parametros * bytes_grad
    otimizador = n_parametros * bytes_estado_adamw
    ativacoes = ativa_bytes_por_token * tokens_por_passo
    total = pesos + gradientes + otimizador + ativacoes
    gb = 1024**3
    return {
        "pesos_gb": round(pesos / gb, 6),
        "gradientes_gb": round(gradientes / gb, 6),
        "adamw_gb": round(otimizador / gb, 6),
        "ativacoes_gb": round(ativacoes / gb, 6),
        "total_gb": round(total / gb, 6),
        "checkpoint_gb": round(pesos / gb, 6),
    }


def recomendar_estrategia(relatorio: RelatorioHardware, n_parametros: int) -> Dict:
    """Recomenda a estratégia de treino com base no hardware medido."""
    from training.distributed import detectar_estrategia

    vram_total = sum(g.vram_gb for g in relatorio.gpus)
    estrategia = detectar_estrategia(len(relatorio.gpus), vram_total if relatorio.gpus else None)
    memoria = estimar_memoria_treino(n_parametros)
    pode_gpu = vram_total >= memoria["total_gb"] if relatorio.gpus else False
    recomendacoes = []
    if not relatorio.gpus:
        recomendacoes.append(
            "Sem GPU: treinar em CPU com modelo pequeno (1-10M params), threads limitadas, "
            "ou migrar o treino para o servidor externo com GPUs."
        )
    elif not pode_gpu:
        recomendacoes.append(
            f"VRAM ({vram_total:.1f} GB) insuficiente para o modelo estimado "
            f"({memoria['total_gb']:.2f} GB): usar gradient accumulation, ativação checkpointing "
            "ou FSDP/CPU offload."
        )
    else:
        recomendacoes.append("VRAM suficiente: batch size pode ser aumentado até o limite de memória.")
    if relatorio.memoria.disponivel_gb < 8:
        recomendacoes.append(
            f"RAM disponível baixa ({relatorio.memoria.disponivel_gb:.1f} GB): usar datasets em memmap "
            "e limitar workers de DataLoader."
        )
    return {
        "estrategia": estrategia.para_dict(),
        "memoria_estimada": memoria,
        "recomendacoes": recomendacoes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecta o hardware disponível para o Apolo")
    parser.add_argument("--json", action="store_true", help="saída em JSON")
    parser.add_argument("--testar-rede", action="store_true", help="mede vazão de download")
    parser.add_argument("--parametros", type=int, default=2_000_000, help="parâmetros a estimar")
    args = parser.parse_args()

    relatorio = coletar_relatorio(testar_rede=args.testar_rede)
    dados = relatorio.para_dict()
    dados["recomendacao"] = recomendar_estrategia(relatorio, args.parametros)
    if args.json:
        print(json.dumps(dados, indent=2, ensure_ascii=False))
    else:
        print(f"CPU:      {relatorio.cpu.modelo} ({relatorio.cpu.nucleos_logicos} threads, {relatorio.cpu.arquitetura})")
        print(f"Memória:  {relatorio.memoria.total_gb:.1f} GB total, {relatorio.memoria.disponivel_gb:.1f} GB livres")
        for gpu in relatorio.gpus:
            print(f"GPU:      {gpu.nome} ({gpu.vram_gb:.1f} GB VRAM, cuda {gpu.cuda})")
        if not relatorio.gpus:
            print("GPU:      nenhuma detectada")
        for disco in relatorio.discos:
            print(f"Disco:    {disco.caminho} {disco.livre_gb:.0f} GB livres de {disco.total_gb:.0f} GB")
        print(f"Torch:    {relatorio.torch} | CUDA disponível: {relatorio.cuda_disponivel}")
        print(json.dumps(dados["recomendacao"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
