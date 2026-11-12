"""Servidor de treinamento e controle do Apolo Zenith 1.9."""

from server.hardware_detection import coletar_relatorio, recomendar_estrategia
from server.job_manager import Job, JobManager
from server.monitoring import ler_metricas, listar_checkpoints, metricas_sistema

__all__ = [
    "coletar_relatorio",
    "recomendar_estrategia",
    "Job",
    "JobManager",
    "metricas_sistema",
    "ler_metricas",
    "listar_checkpoints",
]
