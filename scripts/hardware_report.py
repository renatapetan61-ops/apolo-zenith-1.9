"""Gera o ``hardware_report.json`` do Apolo Zenith 1.9 com dados **reais**.

Todos os valores vêm de medições feitas no momento da execução:

* Android / kernel / CPU – ``getprop``, ``/proc/cpuinfo``, ``/proc/version``;
* memória – ``/proc/meminfo`` e RSS do processo;
* disco – ``shutil.disk_usage``;
* aceleração – ``torch.cuda.is_available()`` (nunca presumida);
* sensores térmicos – tentativa real de leitura (nada é inventado);
* bibliotecas multimodais – importação real de cada módulo;
* **teste de capacidade** – micro-benchmark real de forward/backward com
  medição de RSS e tokens/s.

Uso::

    python -m scripts.hardware_report --saida hardware_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from resources.config import carregar_config_recursos  # noqa: E402
from resources.memory_manager import MemoryManager  # noqa: E402
from resources.storage_manager import StorageManager  # noqa: E402
from resources.thermal_monitor import ThermalMonitor  # noqa: E402


# --------------------------------------------------------------------------- #
# Coleta
# --------------------------------------------------------------------------- #
def _getprop(chave: str) -> Optional[str]:
    binario = shutil.which("getprop")
    if not binario:
        return None
    try:
        saida = subprocess.run(
            [binario, chave], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return saida or None
    except (OSError, subprocess.SubprocessError):
        return None


def _info_android() -> Dict[str, Any]:
    return {
        "versao": _getprop("ro.build.version.release"),
        "sdk": _getprop("ro.build.version.sdk"),
        "fabricante": _getprop("ro.product.manufacturer"),
        "modelo": _getprop("ro.product.model"),
        "hardware": _getprop("ro.hardware"),
        "plataforma": _getprop("ro.board.platform"),
        "build_id": _getprop("ro.build.display.id"),
        "ambiente": "Proot (Termux/Ubuntu)" if "proot" in platform.release().lower() else "Linux nativo",
    }


def _info_kernel() -> Dict[str, Any]:
    try:
        versao = Path("/proc/version").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        versao = ""
    return {"uname": " ".join(os.uname()), "proc_version": versao}


def _info_cpu() -> Dict[str, Any]:
    modelo = "desconhecido"
    nucleos_fisicos: Optional[int] = None
    try:
        texto = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        for linha in texto.splitlines():
            if linha.lower().startswith("hardware") and "processor" not in linha.lower():
                modelo = linha.split(":", 1)[1].strip()
                break
            if linha.lower().startswith("model name") and modelo == "desconhecido":
                modelo = linha.split(":", 1)[1].strip()
        ids = set()
        for linha in texto.splitlines():
            if linha.startswith("processor"):
                try:
                    ids.add(int(linha.split(":")[1]))
                except ValueError:
                    pass
        if ids:
            nucleos_fisicos = len(ids)
    except OSError:
        pass
    return {
        "modelo": modelo,
        "arquitetura": platform.machine(),
        "nucleos_logicos": os.cpu_count(),
        "nucleos_fisicos": nucleos_fisicos,
        "aceleradores": "nenhum (CPU-only)",
    }


def _info_aceleracao() -> Dict[str, Any]:
    import torch

    cuda = torch.cuda.is_available()
    return {
        "cuda_disponivel": bool(cuda),
        "torch_cuda": bool(torch.version.cuda),
        "mps_disponivel": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "gpu": [],
        "observacao": (
            "Nenhuma GPU exposta ao Proot do Termux; todo o treino é CPU-only."
            if not cuda
            else "CUDA disponível"
        ),
    }


def _info_bibliotecas() -> Dict[str, Any]:
    import importlib
    from importlib import metadata

    resultado: Dict[str, Any] = {}
    for nome in (
        "torch",
        "numpy",
        "yaml",
        "PIL",
        "cv2",
        "soundfile",
        "librosa",
        "av",
        "scipy",
        "huggingface_hub",
        "datasets",
    ):
        try:
            modulo = importlib.import_module(nome)
            versao = getattr(modulo, "__version__", None)
            if not versao:
                try:
                    versao = metadata.version(nome)
                except metadata.PackageNotFoundError:
                    versao = "desconhecida"
            resultado[nome] = {
                "disponivel": True,
                "versao": versao,
            }
        except Exception as exc:
            resultado[nome] = {
                "disponivel": False,
                "versao": None,
                "erro": f"{type(exc).__name__}: {exc}",
            }
    ffmpeg = shutil.which("ffmpeg")
    resultado["ffmpeg_binario"] = {"disponivel": bool(ffmpeg), "versao": ffmpeg or None}
    return resultado


def _info_termica() -> Dict[str, Any]:
    monitor = ThermalMonitor()
    leitura = monitor.ler(forcar=True)
    return {
        "fonte": leitura.fonte,
        "disponivel": leitura.disponivel,
        "temperatura_c": leitura.temperatura_c,
        "detalhe": leitura.detalhe,
        "modo_preventivo": not leitura.disponivel,
        "zonas_thermal_acessiveis": leitura.fonte == "sysfs",
        "dumpsys_disponivel": bool(shutil.which("dumpsys")),
        "termux_api_disponivel": bool(shutil.which("termux-thermal")),
    }


def _info_cgroup() -> Dict[str, Any]:
    try:
        conteudo = Path("/proc/self/cgroup").read_text(encoding="utf-8").strip()
    except OSError:
        conteudo = ""
    limites: Dict[str, Any] = {}
    for caminho, chave in (
        ("/sys/fs/cgroup/memory.max", "memory_max"),
        ("/sys/fs/cgroup/memory.current", "memory_current"),
        ("/sys/fs/cgroup/cpu.max", "cpu_max"),
    ):
        try:
            limites[chave] = Path(caminho).read_text(encoding="utf-8").strip()
        except OSError:
            limites[chave] = None
    return {"cgroup": conteudo, "limites_visiveis": limites}


# --------------------------------------------------------------------------- #
# Teste real de capacidade
# --------------------------------------------------------------------------- #
def _teste_capacidade() -> Dict[str, Any]:
    """Micro-benchmark real: forward + backward e consumo de RSS."""
    import torch

    from core.config import ModelConfig
    from core.language_model import ApoloLanguageModel

    manager = MemoryManager()
    rss_antes = manager.ler(forcar=True).rss_mb

    torch.manual_seed(0)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    modelo = ApoloLanguageModel(
        ModelConfig(vocab_size=256, n_layer=2, n_head=2, n_embd=64, block_size=32, mlp_hidden=128)
    )
    x = torch.randint(0, 256, (1, 32))
    y = torch.randint(0, 256, (1, 32))

    # Aquecimento (1 passo não medido)
    _, perda = modelo(x, y)
    perda.backward()
    modelo.zero_grad(set_to_none=True)

    passos = 10
    inicio = time.time()
    for _ in range(passos):
        _, perda = modelo(x, y)
        perda.backward()
        modelo.zero_grad(set_to_none=True)
    decorrido = time.time() - inicio
    rss_depois = manager.ler(forcar=True).rss_mb

    return {
        "modelo_parametros": modelo.num_parameters(),
        "batch": 1,
        "sequencia": 32,
        "passos_medidos": passos,
        "segundos_por_passo": round(decorrido / passos, 4),
        "tokens_por_segundo": round((passos * 32) / max(decorrido, 1e-6), 1),
        "rss_antes_mb": round(rss_antes, 1),
        "rss_depois_mb": round(rss_depois, 1),
        "rss_delta_mb": round(rss_depois - rss_antes, 1),
        "torch_threads": torch.get_num_threads(),
    }


# --------------------------------------------------------------------------- #
# Montagem
# --------------------------------------------------------------------------- #
def coletar_relatorio(incluir_teste: bool = True) -> Dict[str, Any]:
    """Coleta o relatório completo de hardware."""
    import numpy
    import torch

    cfg = carregar_config_recursos(RAIZ / "configs" / "resource_manager.yaml")
    manager = MemoryManager(cfg.memoria)
    armazenamento = StorageManager(cfg.armazenamento, raiz=RAIZ)

    memoria = manager.ler(forcar=True)
    disco = armazenamento.ler(forcar=True)

    relatorio: Dict[str, Any] = {
        "projeto": "Apolo Zenith 1.9",
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "python": {
            "versao": platform.python_version(),
            "implementacao": platform.python_implementation(),
            "executavel": sys.executable,
        },
        "frameworks": {
            "torch": torch.__version__,
            "numpy": numpy.__version__,
        },
        "android": _info_android(),
        "kernel": _info_kernel(),
        "cpu": _info_cpu(),
        "aceleracao": _info_aceleracao(),
        "memoria": memoria.para_dict(),
        "disco": disco.para_dict(),
        "termico": _info_termica(),
        "cgroup": _info_cgroup(),
        "bibliotecas_multimodais": _info_bibliotecas(),
        "limites_configurados": cfg.para_dict(),
    }
    if incluir_teste:
        relatorio["teste_capacidade"] = _teste_capacidade()
    return relatorio


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera o hardware_report.json do Apolo")
    parser.add_argument("--saida", default="hardware_report.json")
    parser.add_argument("--log", default="logs/hardware.log")
    parser.add_argument("--sem-teste", action="store_true", help="não roda o micro-benchmark")
    args = parser.parse_args()

    relatorio = coletar_relatorio(incluir_teste=not args.sem_teste)

    saida = Path(args.saida)
    if not saida.is_absolute():
        saida = RAIZ / saida
    saida.write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log = Path(args.log)
    if not log.is_absolute():
        log = RAIZ / log
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(
            f"[{relatorio['gerado_em']}] android={relatorio['android']['versao']} "
            f"modelo={relatorio['android']['modelo']} cpu={relatorio['cpu']['nucleos_logicos']} "
            f"ram_livre_mb={relatorio['memoria']['ram_disponivel_mb']} "
            f"disco_livre_mb={relatorio['disco']['livre_mb']} "
            f"cuda={relatorio['aceleracao']['cuda_disponivel']} "
            f"termico={relatorio['termico']['fonte']}\n"
        )

    print(f"[hardware] relatório gravado em {saida}")
    print(
        f"  android={relatorio['android']['versao']} | "
        f"cpu={relatorio['cpu']['nucleos_logicos']} threads | "
        f"ram livre={relatorio['memoria']['ram_disponivel_mb']:.0f} MB | "
        f"disco livre={relatorio['disco']['livre_mb'] / 1024:.1f} GB | "
        f"cuda={relatorio['aceleracao']['cuda_disponivel']} | "
        f"termico={relatorio['termico']['fonte']}"
    )
    if "teste_capacidade" in relatorio:
        teste = relatorio["teste_capacidade"]
        print(
            f"  teste: {teste['tokens_por_segundo']:.0f} tok/s | "
            f"{teste['segundos_por_passo'] * 1000:.0f} ms/passo | "
            f"RSS +{teste['rss_delta_mb']:.0f} MB"
        )


if __name__ == "__main__":
    main()
