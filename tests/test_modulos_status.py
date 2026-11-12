"""Testes do registro de status dos módulos do ecossistema.

Estes testes garantem que o projeto **não declare capacidades inexistentes**:

* módulos **implementados** (Language Core, visão, áudio, vídeo) devem
  funcionar de verdade;
* módulos **ainda não implementados** (geração de imagem/vídeo, geração de
  voz, decoder multimodal) precisam levantar erro explícito em vez de simular
  inferência.
"""

from __future__ import annotations

import pytest

from core.status import StatusModulo


def test_status_do_language_core_e_implementado() -> None:
    from core.language_model import ApoloLanguageModel
    from core.config import ModelConfig

    modelo = ApoloLanguageModel(ModelConfig(vocab_size=300, n_layer=1, n_head=2, n_embd=32, block_size=16))
    assert modelo.num_parameters() > 0


# --------------------------------------------------------------------------- #
# Módulos implementados: devem funcionar
# --------------------------------------------------------------------------- #
def test_vision_encoder_visual_funciona() -> None:
    """O encoder de visão existe e produz embeddings."""
    import torch

    from vision import encoder_visual

    modelo = encoder_visual()
    saida = modelo(torch.zeros(1, 3, 64, 64))
    assert saida.shape[0] == 1
    assert torch.isfinite(saida).all()


def test_audio_encoder_audio_funciona() -> None:
    """O encoder de áudio existe e produz embeddings."""
    import torch

    from audio import encoder_audio

    modelo = encoder_audio()
    saida = modelo(torch.zeros(1, 80, 100))
    assert saida.shape[0] == 1
    assert torch.isfinite(saida).all()


def test_video_encoder_video_funciona() -> None:
    """O encoder de vídeo existe e produz embeddings."""
    import torch

    from video import encoder_video

    modelo = encoder_video()
    saida = modelo(torch.zeros(1, 4, 3, 64, 64))
    assert saida.shape[0] == 1
    assert torch.isfinite(saida).all()


# --------------------------------------------------------------------------- #
# Módulos NÃO implementados: precisam falhar explicitamente
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "modulo, funcao",
    [
        ("generation", "gerar_imagem"),
        ("generation", "gerar_video"),
        ("coding", "modelo_codigo"),
    ],
)
def test_modulos_nao_implementados_nao_simulam(modulo: str, funcao: str) -> None:
    """Módulos das fases futuras devem falhar explicitamente."""
    import importlib

    mod = importlib.import_module(modulo)
    with pytest.raises(NotImplementedError) as exc:
        getattr(mod, funcao)()
    assert "não" in str(exc.value).lower() or "nao" in str(exc.value).lower()


def test_todos_os_modulos_reportam_status() -> None:
    import audio
    import coding
    import generation
    import video
    import vision

    modulos = [
        vision.STATUS,
        audio.STATUS,
        video.STATUS,
        generation.STATUS_IMAGEM,
        generation.STATUS_VIDEO,
        coding.STATUS,
    ]
    for status in modulos:
        assert isinstance(status, StatusModulo)
        assert status.fase >= 1
        assert status.descricao
        assert "resumo" in dir(status)

    # Os módulos multimodais **implementados** declaram isso explicitamente.
    for status in (vision.STATUS, audio.STATUS, video.STATUS):
        assert status.implementado is True, (
            f"{status.nome} tem encoder implementado e treinado; o status deve "
            "reflectir isso."
        )

    # Os módulos de geração continuam **não** implementados.
    for status in (generation.STATUS_IMAGEM, generation.STATUS_VIDEO, coding.STATUS):
        assert status.implementado is False, (
            f"{status.nome} ainda não foi implementado; não pode ser declarado "
            "funcional."
        )


def test_modulos_multimodais_nao_geram_texto_sem_decoder() -> None:
    """Não existe decoder multimodal: os módulos não prometem gerar texto."""
    import audio
    import video
    import vision

    for status in (vision.STATUS, audio.STATUS, video.STATUS):
        assert "não gera" in status.descricao.lower() or "sem decoder" in status.descricao.lower()
