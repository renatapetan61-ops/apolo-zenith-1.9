"""Apolo Image/Video Generator — geração de conteúdo (Fases 7 e 8).

STATUS REAL: **não implementado**.

Estrutura reservada para Latent Diffusion (imagem) e Video Diffusion
(vídeo), com VAE, condicionamento textual e métricas FID/CLIP.
"""

from __future__ import annotations

from core.status import StatusModulo

STATUS_IMAGEM = StatusModulo(
    nome="Apolo Image Generator",
    fase=7,
    implementado=False,
    descricao="Latent Diffusion com U-Net/DiT, VAE e condicionamento textual.",
)

STATUS_VIDEO = StatusModulo(
    nome="Apolo Video Generator",
    fase=8,
    implementado=False,
    descricao="Video Diffusion temporal com consistência entre frames.",
)


def gerar_imagem(*args, **kwargs):  # pragma: no cover - stub explícito
    """Levanta erro até que a Fase 7 seja implementada."""
    raise NotImplementedError(
        "Apolo Image Generator ainda não está implementado (Fase 7). "
        "Nenhuma imagem é sintetizada neste repositório."
    )


def gerar_video(*args, **kwargs):  # pragma: no cover - stub explícito
    """Levanta erro até que a Fase 8 seja implementada."""
    raise NotImplementedError(
        "Apolo Video Generator ainda não está implementado (Fase 8). "
        "Nenhum vídeo é sintetizado neste repositório."
    )


__all__ = ["STATUS_IMAGEM", "STATUS_VIDEO", "gerar_imagem", "gerar_video"]
