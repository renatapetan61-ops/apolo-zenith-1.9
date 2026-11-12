"""Apolo Coding Intelligence — programação (Fase 3).

STATUS REAL: **não implementado**.

Estrutura reservada para o tokenizer de código, carregamento de datasets de
programação, geração/completude de código, understanding de repositório,
tool calling e avaliação pass@k com execução de testes.
"""

from __future__ import annotations

from core.status import StatusModulo

STATUS = StatusModulo(
    nome="Apolo Coding Intelligence",
    fase=3,
    implementado=False,
    descricao=(
        "Modelo especializado em código com dataset loader, geração, "
        "tool calling e avaliação pass@1/pass@k."
    ),
)


def modelo_codigo(*args, **kwargs):  # pragma: no cover - stub explícito
    """Levanta erro até que a Fase 3 seja implementada."""
    raise NotImplementedError(
        "Apolo Coding Intelligence ainda não está implementado (Fase 3). "
        "Nenhuma capacidade de programação é simulada neste repositório."
    )


__all__ = ["STATUS", "modelo_codigo"]
