"""Testes dos módulos multimodais do Apolo Zenith 1.9.

Cobre:

* encoders (visão, áudio, vídeo) — formas, parâmetros e finitude;
* loss contrastiva e métrica de recuperação (casos sintéticos);
* loaders lazy (imagem, áudio, vídeo, código) com dados temporários;
* catálogo de datasets (licenças, URLs) e manifesto;
* scheduler do ciclo de 10 dias (plano, estado, exclusividade).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from audio.encoder import ApoloAudioModel, AudioConfig, mel_frontend
from datasets.catalog import CATALOGO, listar
from datasets.code_loader import coletar_codigo, contar_linguagens
from datasets.manifest import DatasetManifest, EntradaManifesto
from training.contraste import avaliar_recuperacao, perda_contrastiva
from video.encoder import ApoloVideoModel, VideoConfig
from vision.encoder import ApoloVisionModel, VisionConfig


# =========================================================================== #
# Encoders
# =========================================================================== #
def test_encoder_visual_saida_e_parametros() -> None:
    modelo = ApoloVisionModel(VisionConfig(dim=64, n_layer=2, n_head=4, patch_size=16, dim_projetor=32))
    imagens = torch.zeros(2, 3, 64, 64)
    saida = modelo(imagens)
    assert saida.shape == (2, 32)
    assert torch.isfinite(saida).all()
    assert modelo.num_parameters() > 0


def test_encoder_visual_usa_patch_configurado() -> None:
    modelo = ApoloVisionModel(VisionConfig(dim=32, n_layer=1, n_head=2, patch_size=16, dim_projetor=16))
    assert modelo.config.num_patches == 16  # 64/16 -> 4x4
    modelo2 = ApoloVisionModel(VisionConfig(dim=32, n_layer=1, n_head=2, patch_size=8, dim_projetor=16))
    assert modelo2.config.num_patches == 64  # 64/8 -> 8x8


def test_encoder_audio_saida_e_parametros() -> None:
    config = AudioConfig(n_mels=80, dim=64, n_layer=1, n_head=4, dim_projetor=32)
    modelo = ApoloAudioModel(config)
    mel = torch.randn(2, 80, 120)
    saida = modelo(mel)
    assert saida.shape == (2, 32)
    assert torch.isfinite(saida).all()
    assert modelo.num_parameters() > 0


def test_mel_frontend_real() -> None:
    """O mel-spectrogram é calculado de verdade (sem mock)."""
    config = AudioConfig(n_mels=40, sample_rate=16000)
    onda = np.sin(np.linspace(0, 2 * math.pi * 440, 16000)).astype(np.float32)
    mel = mel_frontend(onda, config)
    assert mel.shape[0] == 40
    assert mel.shape[1] > 10
    assert np.isfinite(mel).all()


def test_encoder_video_saida_e_parametros() -> None:
    config = VideoConfig(max_frames=4, frame_size=32, n_camadas_temporais=1, dim_projetor=16)
    modelo = ApoloVideoModel(config)
    frames = torch.zeros(2, 4, 3, 32, 32)
    saida = modelo(frames)
    assert saida.shape == (2, 16)
    assert torch.isfinite(saida).all()


# =========================================================================== #
# Contraste
# =========================================================================== #
def test_perda_contrastiva_baixa_quando_alinhado() -> None:
    """Embeddings alinhados → perda baixa; aleatórios → perda alta."""
    torch.manual_seed(0)
    a = torch.eye(8)
    b = torch.eye(8)
    perda_baixa, _ = perda_contrastiva(a, b, temperatura=0.07)
    c = torch.randn(8, 8)
    perda_alta, _ = perda_contrastiva(a, c, temperatura=0.07)
    assert perda_baixa.item() < perda_alta.item()
    assert perda_baixa.item() < 0.2


def test_centralizacao_remove_anisotropia() -> None:
    """A centralização remove a direção comum dos embeddings."""
    from training.contraste import centralizar, normalizar

    base = torch.randn(4, 8)
    deslocado = base + 5.0
    antes = normalizar(deslocado) @ normalizar(deslocado).t()
    depois = normalizar(centralizar(normalizar(deslocado))) @ normalizar(
        centralizar(normalizar(deslocado))
    ).t()
    diagonal_antes = antes[~torch.eye(4, dtype=bool)].mean().item()
    diagonal_depois = depois[~torch.eye(4, dtype=bool)].mean().item()
    assert diagonal_depois < diagonal_antes


def test_recuperacao_perfeita() -> None:
    a = torch.eye(10)
    b = torch.eye(10)
    metricas = avaliar_recuperacao(a, b)
    assert metricas["r1"] == 1.0
    assert metricas["chance"] == pytest.approx(0.1)


# =========================================================================== #
# Loaders
# =========================================================================== #
def test_loader_imagem_lazy(tmp_path: Path) -> None:
    """O Parquet de imagens é lido por row group (sem carregar tudo)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    from datasets.image_loader import ParquetImagemLegendaDataset, collate_imagens

    imagens = []
    legendas = []
    for i in range(4):
        buffer = __import__("io").BytesIO()
        Image.new("RGB", (32, 32), (i * 40, 80, 120)).save(buffer, format="JPEG")
        imagens.append({"bytes": buffer.getvalue(), "path": f"img_{i}.jpg"})
        legendas.append({"caption_0": f"foto {i}", "caption_1": f"outra {i}"})
    estrutura = pa.struct(
        [pa.field("bytes", pa.binary()), pa.field("path", pa.string())]
    )
    tabela = pa.table(
        {
            "image": pa.array(imagens, type=estrutura),
            "caption_0": [f"foto {i}" for i in range(4)],
            "caption_1": [f"outra {i}" for i in range(4)],
        }
    )
    caminho = tmp_path / "dados.parquet"
    pq.write_table(tabela, caminho)

    dataset = ParquetImagemLegendaDataset(caminho, tamanho=16)
    assert len(dataset) == 4
    item = dataset[0]
    assert item.imagem.shape == (3, 16, 16)
    assert len(item.legendas) == 2
    lote = collate_imagens([dataset[0], dataset[1]])
    assert lote["imagens"].shape == (2, 3, 16, 16)


def test_loader_audio_indexa_transcricoes(tmp_path: Path) -> None:
    """O par .flac/.trans.txt é indexado corretamente."""
    import numpy as np
    import soundfile as sf

    from datasets.audio_loader import LibriSpeechDataset

    pasta = tmp_path / "falas" / "1" / "2"
    pasta.mkdir(parents=True)
    for i in range(2):
        onda = np.zeros(16000, dtype=np.float32)
        sf.write(str(pasta / f"1-2-{i:04d}.flac"), onda, 16000)
    (pasta / "1-2.trans.txt").write_text(
        "1-2-0000 primeira fala\n1-2-0001 segunda fala\n", encoding="utf-8"
    )
    dataset = LibriSpeechDataset(tmp_path / "falas")
    assert len(dataset) == 2
    item = dataset[0]
    assert item.transcricao == "primeira fala"
    assert item.mel.shape[0] == 80
    assert item.duracao_s == pytest.approx(1.0, abs=0.01)


def test_loader_video_extrai_frames(tmp_path: Path) -> None:
    """Os frames são extraídos com PyAV (sem processar o vídeo inteiro)."""
    import av
    import numpy as np

    from datasets.video_loader import VideoDataset, extrair_frames

    caminho = tmp_path / "video.mkv"
    container = av.open(str(caminho), mode="w")
    stream = container.add_stream("libx264", rate=5)
    stream.width = 64
    stream.height = 64
    stream.pix_fmt = "yuv420p"
    for i in range(10):
        quadro = av.VideoFrame.from_ndarray(
            np.full((64, 64, 3), i * 20, dtype=np.uint8), format="rgb24"
        )
        for pacote in stream.encode(quadro):
            container.mux(pacote)
    for pacote in stream.encode():
        container.mux(pacote)
    container.close()

    frames, fps, duracao = extrair_frames(caminho, max_frames=4, tamanho=32)
    assert frames.shape[0] == 4
    assert frames.shape[1:] == (3, 32, 32)
    assert fps > 0

    dataset = VideoDataset(tmp_path, max_frames=3, tamanho=32)
    assert len(dataset) == 1
    assert dataset[0].frames.shape[0] == 3


def test_loader_codigo_por_linguagem(tmp_path: Path) -> None:
    """Os arquivos de código são coletados com a linguagem correta."""
    pasta = tmp_path / "python" / "repo"
    pasta.mkdir(parents=True)
    (pasta / "a.py").write_text("def f():\n    return 1\n" * 20, encoding="utf-8")
    (pasta / "b.py").write_text("x = 1\n" * 40, encoding="utf-8")
    (pasta / "c.txt").write_text("ignorado", encoding="utf-8")

    documentos = coletar_codigo(tmp_path)
    assert len(documentos) == 2
    assert all(d.linguagem == "python" for d in documentos)
    assert documentos[0].repositorio == "repo"
    assert contar_linguagens(tmp_path) == {"python": 2}


# =========================================================================== #
# Catálogo e manifesto
# =========================================================================== #
def test_catalogo_tem_licenca_e_url() -> None:
    """Todo dataset do catálogo declara licença, origem e URL."""
    assert CATALOGO
    for spec in listar():
        assert spec.licenca and len(spec.licenca) > 5, spec.nome
        assert spec.url_oficial.startswith("http"), spec.nome
        assert spec.origem, spec.nome
        assert spec.modalidade in {"text", "code", "images", "audio", "video"}


def test_manifesto_persiste(tmp_path: Path) -> None:
    caminho = tmp_path / "dataset_manifest.json"
    manifesto = DatasetManifest(caminho)
    manifesto.registrar(
        EntradaManifesto(
            nome="teste",
            modalidade="images",
            origem="local",
            url_oficial="http://exemplo",
            licenca="CC0",
            tamanho_bytes=123,
            amostras=7,
        )
    )
    outro = DatasetManifest(caminho)
    entrada = outro.obter("teste")
    assert entrada.amostras == 7
    assert entrada.tamanho_bytes == 123
    outro.definir_status("teste", status_treino="utilizado")
    assert DatasetManifest(caminho).obter("teste").status_treino == "utilizado"
    resumo = outro.resumo()
    assert resumo["total_datasets"] == 1
    assert resumo["bytes_baixado"] == 123


# =========================================================================== #
# Scheduler
# =========================================================================== #
def test_plano_de_10_dias() -> None:
    """O plano tem 10 fases com as tarefas descritas na especificação."""
    from orchestrator.scheduler import plano_padrao

    plano = plano_padrao()
    assert len(plano) == 10
    assert [f.dia for f in plano] == list(range(1, 11))
    nomes = " ".join(f.nome.lower() for f in plano)
    for esperado in ("preparação", "texto", "código", "imagens", "áudio", "vídeo", "multimodal"):
        assert esperado in nomes
    # Todas as fases têm ao menos uma tarefa executável.
    for fase in plano:
        assert fase.tarefas
        for tarefa in fase.tarefas:
            assert tarefa.comando
            assert tarefa.descricao


def test_scheduler_persiste_estado(tmp_path: Path) -> None:
    """O estado do ciclo é gravado e relido."""
    from orchestrator.scheduler import Scheduler

    scheduler = Scheduler(arquivo_estado=tmp_path / "training_state.json", raiz=tmp_path)
    scheduler.estado.estado = "executando"
    scheduler.estado.fase_atual = 2
    scheduler.estado.passos = 123
    scheduler.salvar()
    outro = Scheduler(arquivo_estado=tmp_path / "training_state.json", raiz=tmp_path)
    assert outro.estado.estado == "executando"
    assert outro.estado.fase_atual == 2
    assert outro.estado.passos == 123


def test_scheduler_bloqueia_execucao_concorrente(tmp_path: Path) -> None:
    """O scheduler usa o lock: duas execuções simultâneas não acontecem."""
    from orchestrator.scheduler import Scheduler
    from resources.training_lock import TreinoConcorrenteError, TrainingLock

    caminho_lock = tmp_path / "apolo.lock"
    lock = TrainingLock(caminho_lock)
    lock.adquirir("multimodal")
    try:
        scheduler = Scheduler(
            arquivo_estado=tmp_path / "estado.json",
            config_recursos=tmp_path / "sem-config.yaml",
            raiz=tmp_path,
        )
        scheduler.config.diretorio_lock = str(caminho_lock)
        resultado = scheduler.executar(fase=0)
        assert resultado["estado"] == "pausado"
    finally:
        lock.liberar()


def test_scheduler_nao_executa_loop_infinito(tmp_path: Path) -> None:
    """O scheduler executa um plano finito (nunca um loop sem fim)."""
    from orchestrator.scheduler import Scheduler

    scheduler = Scheduler(arquivo_estado=tmp_path / "estado.json", raiz=tmp_path)
    total = sum(len(f.tarefas) for f in scheduler.plano)
    assert total == len({t.nome for f in scheduler.plano for t in f.tarefas})
    assert scheduler.estado.duracao_alvo_s == 240 * 3600.0
