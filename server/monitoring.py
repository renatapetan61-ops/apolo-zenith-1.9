"""Monitoramento do servidor Apolo: métricas de sistema, logs e checkpoints."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional


def metricas_sistema() -> Dict:
    """Métricas instantâneas de CPU/RAM/disco (sem dependências externas)."""
    dados: Dict = {"timestamp": time.time()}

    # CPU: uso agregado via /proc/stat
    try:
        with open("/proc/stat", "r", encoding="utf-8") as fh:
            campos = fh.readline().split()[1:]
        valores = [float(c) for c in campos]
        total = sum(valores)
        ocioso = valores[3] if len(valores) > 3 else 0.0
        dados["cpu_percent"] = round((1.0 - ocioso / total) * 100, 2) if total > 0 else None
    except (OSError, ValueError, IndexError):
        dados["cpu_percent"] = None

    # RAM
    try:
        info = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for linha in fh:
                chave, _, resto = linha.partition(":")
                partes = resto.strip().split()
                if partes:
                    info[chave] = float(partes[0].replace("kB", "")) / (1024**2)
        total_ram = info.get("MemTotal", 0.0)
        livre = info.get("MemAvailable", info.get("MemFree", 0.0))
        dados["memoria_total_gb"] = round(total_ram, 2)
        dados["memoria_livre_gb"] = round(livre, 2)
        dados["memoria_percent"] = round((1 - livre / total_ram) * 100, 2) if total_ram else None
    except (OSError, ValueError):
        pass

    # Disco
    try:
        uso = os.statvfs("/")
        total = (uso.f_blocks * uso.f_frsize) / (1024**3)
        livre = (uso.f_bavail * uso.f_frsize) / (1024**3)
        dados["disco_total_gb"] = round(total, 2)
        dados["disco_livre_gb"] = round(livre, 2)
    except OSError:
        pass

    return dados


def ler_metricas(arquivo: str | Path, ultimas: int = 50) -> List[Dict]:
    """Lê as últimas linhas de um arquivo JSONL de métricas."""
    caminho = Path(arquivo)
    if not caminho.exists():
        return []
    linhas: List[str] = []
    try:
        with caminho.open("r", encoding="utf-8", errors="replace") as fh:
            for linha in fh:
                if linha.strip():
                    linhas.append(linha)
                if len(linhas) > ultimas * 4:
                    linhas = linhas[-ultimas * 4 :]
    except OSError:
        return []
    saida = []
    for linha in linhas[-ultimas:]:
        try:
            saida.append(json.loads(linha))
        except json.JSONDecodeError:
            continue
    return saida


def listar_checkpoints(diretorio: str | Path = "checkpoints") -> List[Dict]:
    """Lista os checkpoints disponíveis com tamanho e metadados."""
    raiz = Path(diretorio)
    saida: List[Dict] = []
    if not raiz.exists():
        return saida
    for caminho in sorted(raiz.rglob("meta.json")):
        diretorio_ckpt = caminho.parent
        try:
            meta = json.loads(caminho.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
        tamanho = sum(p.stat().st_size for p in diretorio_ckpt.glob("*.pt"))
        saida.append(
            {
                "nome": diretorio_ckpt.name,
                "caminho": str(diretorio_ckpt),
                "step": meta.get("step"),
                "epoch": meta.get("epoch"),
                "tamanho_mb": round(tamanho / (1024**2), 2),
                "model_config": meta.get("model_config"),
                "validacao": meta.get("validacao"),
            }
        )
    return saida


def ultimas_linhas(caminho: str | Path, n: int = 100) -> str:
    """Retorna as últimas linhas de um arquivo de log."""
    p = Path(caminho)
    if not p.exists():
        return ""
    try:
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return ""
