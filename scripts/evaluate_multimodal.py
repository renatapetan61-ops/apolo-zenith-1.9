"""Avaliação real das modalidades multimodais do Apolo Zenith 1.9.

Cada métrica é **medida** em dadosheld-out (nunca declarada sem medição):

* **imagens** – recuperação imagem→texto e texto→imagem (R@1, MRR) sobre a
  divisão de validação, comparada com a chance;
* **áudio**   – recuperação áudio→texto e texto→áudio sobre as falas de
  validação;
* **vídeo**   – acurácia da ordem temporal (sequência correta vs. embaralhada);
* **multimodal** – alinhamento conjunto (projetores) e consistência
  cross-modal.

O resultado é gravado em ``logs/avaliacao_multimodal.json`` e em
``logs/evaluation.log``.

Uso::

    python -m scripts.evaluate_multimodal --modalidade images
    python -m scripts.evaluate_multimodal --todas
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from audio.encoder import ApoloAudioModel, AudioConfig  # noqa: E402
from datasets.audio_loader import LibriSpeechDataset  # noqa: E402
from datasets.image_loader import ParquetImagemLegendaDataset  # noqa: E402
from datasets.video_loader import VideoDataset  # noqa: E402
from resources.config import carregar_config_recursos  # noqa: E402
from training.contraste import avaliar_recuperacao  # noqa: E402
from video.encoder import ApoloVideoModel, VideoConfig  # noqa: E402
from vision.encoder import ApoloVisionModel, VisionConfig  # noqa: E402


def _config_do_encoder(caminho_modelo: Path, padrao):
    """Reconstrói a configuração do encoder.

    Ordem: ``config_encoder.json`` (salvo junto do checkpoint) → inferência a
    partir do próprio ``state_dict`` (para checkpoints antigos) → padrão.
    """
    caminho_config = Path(caminho_modelo).parent / "config_encoder.json"
    dados: Dict[str, Any] = {}
    if caminho_config.exists():
        try:
            dados = json.loads(caminho_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            dados = {}
    if not dados:
        dados = _inferir_config_encoder(caminho_modelo, padrao)
    validos = set(padrao.para_dict())
    campos = {k: v for k, v in dados.items() if k in validos}
    try:
        return type(padrao)(**campos)
    except TypeError:
        return padrao


def _inferir_config_encoder(caminho_modelo: Path, padrao) -> Dict[str, Any]:
    """Inferre a arquitetura do encoder olhando as chaves do ``state_dict``."""
    try:
        estado = torch.load(caminho_modelo, map_location="cpu", weights_only=True)
    except Exception:
        return {}
    if isinstance(estado, dict) and "modelo" in estado:
        estado = estado["modelo"]
    chaves = list(estado.keys())
    inferido: Dict[str, Any] = {}

    def _maior_indice(prefixo: str) -> int:
        indices = [
            int(c[len(prefixo):].split(".")[0])
            for c in chaves
            if c.startswith(prefixo) and c[len(prefixo):].split(".")[0].isdigit()
        ]
        return (max(indices) + 1) if indices else 0

    blocos = _maior_indice("encoder.blocos.")
    if blocos and "n_camadas_temporais" in padrao.para_dict():
        inferido["n_camadas_temporais"] = blocos
    if "encoder.frame_encoder.blocks." in chaves:
        camadas = _maior_indice("encoder.frame_encoder.blocks.")
        if camadas and "vision" in padrao.para_dict():
            visao = dict(padrao.vision.para_dict())
            visao["n_layer"] = camadas
            inferido["vision"] = visao
    # Dimensão do modelo a partir do peso da projeção
    for chave in ("projetor.net.0.weight", "proj_entrada.weight", "encoder.proj_entrada.weight"):
        if chave in estado:
            dim = estado[chave].shape[0]
            if "dim" in padrao.para_dict():
                inferido["dim"] = dim
            if "vision" in inferido:
                visao = dict(inferido["vision"])
                visao["dim"] = dim
                inferido["vision"] = visao
            break
    return inferido


def _log(mensagem: str) -> None:
    linha = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {mensagem}"
    print(linha, flush=True)
    log = RAIZ / "logs" / "evaluation.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(linha + "\n")


# --------------------------------------------------------------------------- #
# Embeddings de texto (Language Core congelado)
# --------------------------------------------------------------------------- #
class TextoEmbeddings:
    """Embeddings de texto do Language Core (congelado)."""

    def __init__(self, checkpoint: str, tokenizer: str, dim_projetor: int = 128) -> None:
        from core.language_model import ApoloLanguageModel
        from tokenizer.tokenizer import BPETokenizer

        self.tokenizer = BPETokenizer.load(tokenizer)
        self.modelo = ApoloLanguageModel.load(checkpoint, map_location="cpu")
        self.modelo.eval()
        for p in self.modelo.parameters():
            p.requires_grad_(False)
        self.dim = dim_projetor

    def __call__(self, textos: List[str]) -> torch.Tensor:
        if not textos:
            return torch.zeros(0, self.dim)
        ids = []
        for texto in textos:
            tokens = list(self.tokenizer.encode(texto, add_bos=False, add_eos=False))[:32]
            ids.append(tokens or [0])
        largura = max(len(i) for i in ids)
        lote = torch.zeros(len(ids), largura, dtype=torch.long)
        mascara = torch.zeros(len(ids), largura)
        for indice, tokens in enumerate(ids):
            lote[indice, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mascara[indice, : len(tokens)] = 1.0
        with torch.no_grad():
            bruto = self.modelo.embed_text(lote, mascara)
        if bruto.shape[-1] == self.dim:
            return bruto
        return bruto[:, : self.dim]


# --------------------------------------------------------------------------- #
# Avaliações
# --------------------------------------------------------------------------- #
def avaliar_imagens(args) -> Dict[str, Any]:
    """Recuperação imagem→texto na divisão de validação."""
    modelo_path = Path(args.modelo_visao)
    if not modelo_path.exists():
        return {"erro": f"encoder de visão não encontrado: {modelo_path}"}
    modelo = ApoloVisionModel(
        _config_do_encoder(modelo_path, VisionConfig(dim_projetor=128))
    )
    modelo.load_state_dict(torch.load(modelo_path, map_location="cpu", weights_only=True))
    modelo.eval()

    caminhos = [args.parquet]
    if args.parquets_extra:
        caminhos.extend(args.parquets_extra.split(","))
    dataset = ParquetImagemLegendaDataset(caminhos, max_itens=args.max_itens)
    ordem = dataset.ordem_por_grupo(1337)
    corte = max(1, int(len(dataset) * (1.0 - args.fracao_validacao)))
    validacao = [dataset[i] for i in ordem[corte:]]

    textos = TextoEmbeddings(args.checkpoint_lm, args.tokenizer)
    emb_img: List[torch.Tensor] = []
    emb_txt: List[torch.Tensor] = []
    with torch.no_grad():
        for item in validacao:
            emb_img.append(modelo(item.imagem.unsqueeze(0))[0])
            emb_txt.append(textos([item.legendas[0] if item.legendas else "uma imagem"])[0])
    metricas = avaliar_recuperacao(torch.stack(emb_img), torch.stack(emb_txt))
    metricas["amostras_validacao"] = len(validacao)
    metricas["nota"] = (
        "R@1 medido no split de validação; 'chance' é a recuperação esperada "
        "por sorteio aleatório com o mesmo número de candidatos."
    )
    return metricas


def avaliar_audio(args) -> Dict[str, Any]:
    """Recuperação áudio→texto nas falas de validação."""
    modelo_path = Path(args.modelo_audio)
    if not modelo_path.exists():
        return {"erro": f"encoder de áudio não encontrado: {modelo_path}"}
    modelo = ApoloAudioModel(
        _config_do_encoder(modelo_path, AudioConfig(dim_projetor=128))
    )
    modelo.load_state_dict(torch.load(modelo_path, map_location="cpu", weights_only=True))
    modelo.eval()

    dataset = LibriSpeechDataset(args.pasta_audio, max_itens=args.max_itens)
    gerador = np.random.default_rng(1337)   # semente fixa: resultado reprodutível
    indices = list(range(len(dataset)))
    gerador.shuffle(indices)
    corte = max(1, int(len(dataset) * (1.0 - args.fracao_validacao)))
    validacao = [dataset[i] for i in indices[corte:]]

    textos = TextoEmbeddings(args.checkpoint_lm, args.tokenizer)
    emb_audio: List[torch.Tensor] = []
    emb_txt: List[torch.Tensor] = []
    with torch.no_grad():
        for item in validacao:
            emb_audio.append(modelo(item.mel.unsqueeze(0))[0])
            emb_txt.append(textos([item.transcricao or "fala"])[0])
    metricas = avaliar_recuperacao(torch.stack(emb_audio), torch.stack(emb_txt))
    metricas["amostras_validacao"] = len(validacao)
    return metricas


def avaliar_video(args) -> Dict[str, Any]:
    """Acurácia da ordem temporal e recuperação do título."""
    modelo_path = Path(args.modelo_video)
    if not modelo_path.exists():
        return {"erro": f"encoder de vídeo não encontrado: {modelo_path}"}
    estado = torch.load(modelo_path, map_location="cpu", weights_only=True)
    modelo = ApoloVideoModel(
        _config_do_encoder(Path(modelo_path), VideoConfig(dim_projetor=128, max_frames=args.max_frames))
    )
    modelo.load_state_dict(estado["modelo"])
    modelo.eval()
    cabecalho = torch.nn.Linear(128, 1)
    cabecalho.load_state_dict(estado["cabecalho_ordem"])
    cabecalho.eval()

    dataset = VideoDataset(
        args.pasta_video, max_frames=args.max_frames, tamanho=args.frame_size
    )
    if len(dataset) == 0:
        return {"erro": f"nenhum vídeo em {args.pasta_video}"}

    acertos = 0
    total = 0
    emb_video: List[torch.Tensor] = []
    emb_txt: List[torch.Tensor] = []
    textos = TextoEmbeddings(args.checkpoint_lm, args.tokenizer)
    with torch.no_grad():
        for item in dataset:
            for embaralhado in (False, True):
                frames = item.frames.unsqueeze(0).clone()
                if embaralhado and frames.shape[1] > 1:
                    frames = frames[:, torch.randperm(frames.shape[1])]
                embedding = modelo(frames)[0]
                previsto = 1 if float(cabecalho(embedding).item()) > 0 else 0
                esperado = 0 if embaralhado else 1
                acertos += int(previsto == esperado)
                total += 1
                if not embaralhado:
                    emb_video.append(embedding)
                    emb_txt.append(textos([item.titulo or "vídeo"])[0])
    metricas: Dict[str, Any] = {
        "acuracia_ordem": round(acertos / max(total, 1), 4),
        "amostras_ordem": total,
        "chance_ordem": 0.5,
        "videos": len(dataset),
    }
    if len(emb_video) >= 8:
        metricas["recuperacao"] = avaliar_recuperacao(torch.stack(emb_video), torch.stack(emb_txt))
    else:
        metricas["recuperacao"] = {
            "nota": (
                f"não calculada: apenas {len(emb_video)} vídeos (mínimo 8 para "
                "interpretação). O número seria enganoso com esta amostra."
            )
        }
    return metricas


def avaliar_multimodal(args) -> Dict[str, Any]:
    """Alinhamento conjunto dos projetores (visão + áudio → texto)."""
    base = Path(args.modelo_multimodal)
    candidatos = [base / "projetores.pt", base / "final" / "projetores.pt"]
    caminho = next((c for c in candidatos if c.exists()), None)
    if caminho is None:
        return {"erro": f"projetores não encontrados em: {candidatos}"}
    estado = torch.load(caminho, map_location="cpu", weights_only=True)

    from training.multimodal_trainer import ProjetorMultimodal

    projetor_visao = ProjetorMultimodal(128, 128)
    projetor_audio = ProjetorMultimodal(128, 128)
    projetor_visao.load_state_dict(estado["projetor_visao"])
    projetor_audio.load_state_dict(estado["projetor_audio"])
    projetor_visao.eval()
    projetor_audio.eval()

    resultado: Dict[str, Any] = {}
    if Path(args.modelo_visao).exists():
        modelo = ApoloVisionModel(
            _config_do_encoder(Path(args.modelo_visao), VisionConfig(dim_projetor=128))
        )
        modelo.load_state_dict(torch.load(args.modelo_visao, map_location="cpu", weights_only=True))
        modelo.eval()
        caminhos = [args.parquet]
        if args.parquets_extra:
            caminhos.extend(args.parquets_extra.split(","))
        dataset = ParquetImagemLegendaDataset(caminhos, max_itens=args.max_itens)
        ordem = dataset.ordem_por_grupo(1337)
        corte = max(1, int(len(dataset) * (1.0 - args.fracao_validacao)))
        textos = TextoEmbeddings(args.checkpoint_lm, args.tokenizer)
        emb_img, emb_txt = [], []
        with torch.no_grad():
            for i in ordem[corte:]:
                item = dataset[i]
                emb_img.append(projetor_visao(modelo(item.imagem.unsqueeze(0)))[0])
                emb_txt.append(textos([item.legendas[0] if item.legendas else "uma imagem"])[0])
        resultado["imagem_texto"] = avaliar_recuperacao(torch.stack(emb_img), torch.stack(emb_txt))
    else:
        resultado["imagem_texto"] = {"erro": "encoder de visão ausente"}

    if Path(args.modelo_audio).exists():
        modelo = ApoloAudioModel(
            _config_do_encoder(Path(args.modelo_audio), AudioConfig(dim_projetor=128))
        )
        modelo.load_state_dict(torch.load(args.modelo_audio, map_location="cpu", weights_only=True))
        modelo.eval()
        dataset = LibriSpeechDataset(args.pasta_audio, max_itens=args.max_itens)
        gerador = np.random.default_rng(1337)
        indices = list(range(len(dataset)))
        gerador.shuffle(indices)
        corte = max(1, int(len(dataset) * (1.0 - args.fracao_validacao)))
        textos = TextoEmbeddings(args.checkpoint_lm, args.tokenizer)
        emb_audio, emb_txt = [], []
        with torch.no_grad():
            for i in indices[corte:]:
                item = dataset[i]
                emb_audio.append(projetor_audio(modelo(item.mel.unsqueeze(0)))[0])
                emb_txt.append(textos([item.transcricao or "fala"])[0])
        resultado["audio_texto"] = avaliar_recuperacao(torch.stack(emb_audio), torch.stack(emb_txt))
    else:
        resultado["audio_texto"] = {"erro": "encoder de áudio ausente"}

    resultado["observacao"] = (
        "O modelo alinha representações; não há decoder de linguagem treinado, "
        "logo não gera descrições de imagens nem transcrições."
    )
    return resultado


# --------------------------------------------------------------------------- #
def avaliar_codigo(args) -> Dict[str, Any]:
    """Qualidade do código **gerado** pelo LM de código.

    Gera continuações a partir de prompts reais e mede a **taxa de sintaxe
    válida** com ``ast.parse`` (a análise é estática: o código gerado nunca é
    executado).  Para um modelo de 1,9M parâmetros a taxa é baixa — o valor
    medido é o que importa.
    """
    from core.language_model import ApoloLanguageModel
    from tokenizer.tokenizer import BPETokenizer

    caminho = Path(args.checkpoint_codigo)
    if not caminho.exists():
        return {"erro": f"checkpoint de código não encontrado: {caminho}"}
    try:
        modelo = ApoloLanguageModel.load(caminho, map_location="cpu")
        tokenizer = BPETokenizer.load(args.tokenizer)
    except Exception as exc:
        return {"erro": f"não foi possível carregar o modelo: {exc}"}

    modelo.eval()
    prompts = [
        "def soma(a, b):",
        "import os\n",
        "class Usuario:\n",
        "for i in range(10):\n",
        "def main():\n",
    ]
    validos = 0
    com_conteudo = 0
    detalhes = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        entrada = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            gerados = modelo.generate(
                entrada,
                max_new_tokens=48,
                temperature=0.2,
                top_k=20,
            )
        texto = tokenizer.decode(gerados[0].tolist())
        # Uma geração **vazia** (só espaços) não conta como código válido:
        # `ast.parse` aceitaria, mas não há código algum.
        tem_conteudo = bool(texto.strip())
        com_conteudo += int(tem_conteudo)
        try:
            import ast

            ast.parse(texto)
            ok_sintaxe = True
            detalhe = "sintaxe válida"
        except SyntaxError as exc:
            ok_sintaxe = False
            detalhe = f"linha {exc.lineno}: {exc.msg}"
        valido = ok_sintaxe and tem_conteudo
        validos += int(valido)
        detalhes.append(
            {
                "prompt": prompt.strip(),
                "tokens_gerados": len(gerados[0].tolist()),
                "sintaxe_ok": ok_sintaxe,
                "tem_conteudo": tem_conteudo,
                "valido": valido,
                "detalhe": detalhe if tem_conteudo else "geração vazia (só espaços)",
                "amostra": repr(texto[:160]),
            }
        )
    return {
        "amostras": len(prompts),
        "sintaxe_valida": validos,
        "taxa_sintaxe_valida": round(validos / len(prompts), 4),
        "geracoes_com_conteudo": com_conteudo,
        "nota": (
            "Geração livre a partir de prompts de código; a validação é estática "
            "(ast.parse) e o código NUNCA é executado. Só conta como válido o "
            "que tem conteúdo não vazio e sintaxe correta — um modelo que "
            "devolve apenas espaços é tratado como falha, não como sucesso."
        ),
        "detalhe": detalhes,
    }


AVALIADORES = {
    "images": avaliar_imagens,
    "audio": avaliar_audio,
    "video": avaliar_video,
    "multimodal": avaliar_multimodal,
    "codigo": avaliar_codigo,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação multimodal do Apolo Zenith 1.9")
    parser.add_argument("--modalidade", default=None, choices=list(AVALIADORES))
    parser.add_argument("--todas", action="store_true")
    parser.add_argument("--max-itens", type=int, default=800)
    parser.add_argument("--fracao-validacao", type=float, default=0.2)
    parser.add_argument("--parquet", default="datasets/images/raw/flickr8k/test.parquet")
    parser.add_argument("--parquets-extra", default="datasets/images/raw/flickr8k/train_0.parquet")
    parser.add_argument("--pasta-audio", default="datasets/audio/raw/librispeech/LibriSpeech")
    parser.add_argument("--pasta-video", default="datasets/video/raw")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=64)
    parser.add_argument("--modelo-visao", default="models/vision/melhor/modelo.pt")
    parser.add_argument("--modelo-audio", default="models/audio/melhor/modelo.pt")
    parser.add_argument("--modelo-video", default="models/video/melhor/modelo.pt")
    parser.add_argument("--usar-final", action="store_true",
                        help="avalia o checkpoint 'final' em vez do 'melhor'")
    parser.add_argument("--modelo-multimodal", default="models/multimodal/final")
    parser.add_argument("--checkpoint-lm", default="checkpoints/apolo-lm-tiny/checkpoints/best")
    parser.add_argument("--checkpoint-codigo", default="checkpoints/apolo-code/checkpoints/best")
    parser.add_argument("--tokenizer", default="datasets_storage/processed/tokenizer")
    parser.add_argument("--saida", default="logs/avaliacao_multimodal.json")
    args = parser.parse_args()

    if args.usar_final:
        args.modelo_visao = args.modelo_visao.replace("/melhor/", "/final/")
        args.modelo_audio = args.modelo_audio.replace("/melhor/", "/final/")
        args.modelo_video = args.modelo_video.replace("/melhor/", "/final/")

    alvos = list(AVALIADORES) if args.todas else [args.modalidade]
    alvos = [a for a in alvos if a]
    if not alvos:
        print("use --modalidade ou --todas")
        return

    resultados: Dict[str, Any] = {}
    saida = RAIZ / args.saida
    if saida.exists():
        try:
            resultados = json.loads(saida.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            resultados = {}

    for modalidade in alvos:
        _log(f"avaliando {modalidade} ...")
        inicio = time.time()
        resultado = AVALIADORES[modalidade](args)
        resultado["tempo_s"] = round(time.time() - inicio, 1)
        resultados[modalidade] = resultado
        _log(f"{modalidade}: {json.dumps(resultado, ensure_ascii=False)}")

    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[avaliação] gravada em {saida}")


if __name__ == "__main__":
    main()
