"""Metadados de status dos módulos do ecossistema Apolo.

O registro de status é o mecanismo que impede o projeto de **declarar
capacidades que não existem**: cada módulo informa explicitamente se está
implementado e em qual fase.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StatusModulo:
    """Metadados honestos sobre o estado de um módulo do ecossistema."""

    nome: str
    fase: int
    implementado: bool
    descricao: str

    def resumo(self) -> str:
        estado = "implementado" if self.implementado else "planejado (sem código funcional)"
        return f"{self.nome} — fase {self.fase} — {estado}: {self.descricao}"

    def para_dict(self) -> dict:
        return {
            "nome": self.nome,
            "fase": self.fase,
            "implementado": self.implementado,
            "descricao": self.descricao,
        }


__all__ = ["StatusModulo"]
