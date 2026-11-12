"""Treinador **MULTIMODAL** (alinhamento conjunto).

Este módulo **integra** os módulos realmente treinados:

* o Language Core (texto) — congelado, fornece os embeddings de texto;
* o ``VisionEncoder`` + projetor (visão) — carregados de ``models/vision``;
* o ``MelEncoder`` + projetor (áudio) — carregados de ``models/audio``;
* (opcional) o ``VideoEncoder`` + projetor (vídeo) — de ``models/video``.

O que é treinado aqui: um **projetor compartilhado** que leva os
embeddings visuais e de áudio ao mesmo espaço textual, com perda InfoNCE
multimodal (texto ↔ imagem e texto ↔ áudio) e uma perda de **consistência
cross-modal** (imagem ↔ áudio do mesmo conceito).

Métricas: recuperação imagem→texto, áudio→texto e texto→imagem/áudio.

Limitação honesta: o modelo **não** gera descrições em linguagem natural —
ele alinha representações.  Qualquer afirmação de "descrição de imagens" ou
"sumário de vídeos" exigiria um decoder de texto treinado, que não existe
neste repositório.

Uso::

    python -m training.multimodal_trainer --steps 200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from audio.encoder import ApoloAudioModel, AudioConfig  # noqa: E402
from core.language_model import ApoloLanguageModel  # noqa: E402
from datasets.audio_loader import LibriSpeechDataset  # noqa: E402
from datasets.image_loader import ParquetImagemLegendaDataset, collate_imagens  # noqa: E402
from training.audio_trainer import collate_audio  # noqa: E402
from training.contraste import avaliar_recuperacao, perda_contrastiva  # noqa: E402
from vision.encoder import ApoloVisionModel, VisionConfig  # noqa: E402


@dataclass
class ConfigTreinoMultimodal:
    """Hiperparâmetros do treino multimodal."""

    parquet_imagens: str = "datasets/images/raw/flickr8k/test.parquet"
    pasta_audio: str = "datasets/audio/raw/librispeech/LibriSpeech"
    modelo_visao: str = "models/vision/final/modelo.pt"
    modelo_audio: str = "models/audio/final/modelo.pt"
    checkpoint_lm: str = "checkpoints/apolo-lm-tiny/checkpoints/best"
    tokenizer: str = "datasets_storage/processed/tokenizer"
    saida: str = "models/multimodal"
    batch_size: int = 2
    passos: int = 200
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup: int = 20
    temperatura: float = 0.07
    peso_consistencia: float = 0.0   # ver nota: pares multimodal NÃO são do mesmo conteúdo
    max_itens: Optional[int] = 200
    seed: int = 1337
    dim_projetor: int = 128
    max_tokens: int = 32
    intervalo_avaliacao: int = 50
    log: str = "logs/train_multimodal.jsonl"

    def para_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProjetorMultimodal(torch.nn.Module):
    """Projetor compartilhado que leva qualquer modalidade ao espaço do texto."""

    def __init__(self, dim_entrada: int, dim_saida: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_entrada, dim_entrada),
            torch.nn.GELU(),
            torch.nn.Linear(dim_entrada, dim_saida),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _config_do_encoder(caminho_modelo: Path, padrao):
    """Reconstrói a configuração do encoder a partir do ``config_encoder.json``.

    Sem isso, um encoder treinado com outra dimensão/patch falharia ao
    carregar (e o erro apareceria como "encoder ausente").
    """
    caminho_config = Path(caminho_modelo).parent / "config_encoder.json"
    if not caminho_config.exists():
        return padrao
    try:
        dados = json.loads(caminho_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return padrao
    validos = {f for f in padrao.para_dict()}
    campos = {k: v for k, v in dados.items() if k in validos}
    try:
        return type(padrao)(**campos)
    except TypeError:
        return padrao


class MultimodalTrainer:
    """Alinha visão e áudio ao espaço textual do Language Core."""

    def __init__(self, config: ConfigTreinoMultimodal) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.device = torch.device("cpu")

        self.visao = self._carregar_visao()
        self.audio = self._carregar_audio()
        self.lm = self._carregar_lm()
        self.tokenizer = self._carregar_tokenizer()

        dim = self.config.dim_projetor
        self.projetor_visao = ProjetorMultimodal(dim, dim)
        self.projetor_audio = ProjetorMultimodal(dim, dim)

        parametros = list(self.projetor_visao.parameters()) + list(self.projetor_audio.parameters())
        # Os encoders carregados ficam congelados: só o alinhamento é treinado.
        for modelo in (self.visao, self.audio):
            if modelo is not None:
                for p in modelo.parameters():
                    p.requires_grad_(False)
        self.parametros = parametros

        Path(config.saida).mkdir(parents=True, exist_ok=True)
        self.log_path = RAIZ / config.log
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def _carregar_visao(self) -> Optional[ApoloVisionModel]:
        caminho = Path(self.config.modelo_visao)
        if not caminho.exists():
            print(f"[aviso] encoder de visão não encontrado: {caminho}")
            return None
        try:
            config = _config_do_encoder(caminho, VisionConfig(dim_projetor=self.config.dim_projetor))
            modelo = ApoloVisionModel(config)
            modelo.load_state_dict(torch.load(caminho, map_location="cpu", weights_only=True))
            modelo.eval()
            return modelo
        except Exception as exc:
            print(f"[aviso] falha ao carregar o encoder de visão: {exc}")
            return None

    def _carregar_audio(self) -> Optional[ApoloAudioModel]:
        caminho = Path(self.config.modelo_audio)
        if not caminho.exists():
            print(f"[aviso] encoder de áudio não encontrado: {caminho}")
            return None
        try:
            config = _config_do_encoder(caminho, AudioConfig(dim_projetor=self.config.dim_projetor))
            modelo = ApoloAudioModel(config)
            modelo.load_state_dict(torch.load(caminho, map_location="cpu", weights_only=True))
            modelo.eval()
            return modelo
        except Exception as exc:
            print(f"[aviso] falha ao carregar o encoder de áudio: {exc}")
            return None

    def _carregar_lm(self) -> Optional[ApoloLanguageModel]:
        caminho = Path(self.config.checkpoint_lm)
        if not caminho.exists():
            return None
        try:
            modelo = ApoloLanguageModel.load(caminho, map_location=self.device)
            modelo.eval()
            for p in modelo.parameters():
                p.requires_grad_(False)
            return modelo
        except Exception:
            return None

    def _carregar_tokenizer(self):
        caminho = Path(self.config.tokenizer)
        if not caminho.exists():
            return None
        try:
            from tokenizer.tokenizer import BPETokenizer

            return BPETokenizer.load(caminho)
        except Exception:
            return None

    def embeddings_texto(self, textos: List[str]) -> torch.Tensor:
        if self.tokenizer is None or self.lm is None:
            return torch.zeros(len(textos), self.config.dim_projetor)
        ids = []
        for texto in textos:
            tokens = list(self.tokenizer.encode(texto, add_bos=False, add_eos=False))
            ids.append((tokens or [0])[: self.config.max_tokens])
        largura = max(len(i) for i in ids)
        lote = torch.zeros(len(ids), largura, dtype=torch.long)
        mascara = torch.zeros(len(ids), largura)
        for indice, tokens in enumerate(ids):
            lote[indice, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mascara[indice, : len(tokens)] = 1.0
        with torch.no_grad():
            bruto = self.lm.embed_text(lote, mascara)
        if bruto.shape[-1] == self.config.dim_projetor:
            return bruto
        return bruto[:, : self.config.dim_projetor]

    # ------------------------------------------------------------------ #
    def treinar(self) -> Dict[str, Any]:
        if self.visao is None or self.audio is None or self.lm is None:
            raise SystemExit(
                "[erro] o treino multimodal exige os encoders de visão/áudio e o "
                "Language Core treinados. Rode os trainers anteriores primeiro."
            )

        dataset_img = ParquetImagemLegendaDataset(
            self.config.parquet_imagens, max_itens=self.config.max_itens
        )
        dataset_audio = LibriSpeechDataset(
            self.config.pasta_audio, max_itens=self.config.max_itens
        )
        gerador = torch.Generator()
        gerador.manual_seed(self.config.seed)
        loader_img = DataLoader(
            dataset_img,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            generator=gerador,
            collate_fn=collate_imagens,
        )
        loader_audio = DataLoader(
            dataset_audio,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            generator=gerador,
            collate_fn=lambda lote: collate_audio(lote),
        )

        otimizador = torch.optim.AdamW(
            self.parametros,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        total_passos = max(1, self.config.passos)
        escalonador = torch.optim.lr_scheduler.LambdaLR(
            otimizador,
            lr_lambda=lambda passo: min(1.0, (passo + 1) / max(self.config.warmup, 1))
            * max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(passo / total_passos, 1.0)))),
        )

        inicio = time.time()
        passo = 0
        historico: List[Dict[str, Any]] = []
        iter_img = iter(loader_img)
        iter_audio = iter(loader_audio)

        while passo < total_passos:
            try:
                lote_img = next(iter_img)
            except StopIteration:
                iter_img = iter(loader_img)
                lote_img = next(iter_img)
            try:
                lote_audio = next(iter_audio)
            except StopIteration:
                iter_audio = iter(loader_audio)
                lote_audio = next(iter_audio)

            with torch.no_grad():
                base_img = self.visao(lote_img["imagens"])
                base_audio = self.audio(lote_audio["mels"])
            emb_img = self.projetor_visao(base_img)
            emb_audio = self.projetor_audio(base_audio)
            emb_texto_img = self.embeddings_texto(
                [t or "uma imagem" for t in lote_img["legendas"]]
            )
            emb_texto_audio = self.embeddings_texto(
                [t or "uma fala" for t in lote_audio["transcricoes"]]
            )

            perda_img, _ = perda_contrastiva(
                emb_img, emb_texto_img, temperatura=self.config.temperatura
            )
            perda_audio, _ = perda_contrastiva(
                emb_audio, emb_texto_audio, temperatura=self.config.temperatura
            )
            # Consistência cross-modal: **desligada por padrão**.
            # Neste pipeline o par imagem↔áudio NÃO descreve o mesmo
            # conteúdo (a imagem é do Flickr8k, o áudio é do LibriSpeech),
            # então forçar proximidade entre eles alinharia conceitos
            # distintos e degradaria as duas modalidades.
            if self.config.peso_consistencia > 0:
                perda_consist = F.mse_loss(
                    F.normalize(emb_img, dim=-1), F.normalize(emb_audio, dim=-1)
                )
            else:
                perda_consist = torch.zeros((), device=emb_img.device)
            perda = perda_img + perda_audio + self.config.peso_consistencia * perda_consist

            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(self.parametros, 1.0)
            otimizador.step()
            escalonador.step()
            passo += 1

            if passo % 10 == 0 or passo == 1:
                registro = {
                    "passo": passo,
                    "perda": round(float(perda.item()), 4),
                    "perda_imagem": round(float(perda_img.item()), 4),
                    "perda_audio": round(float(perda_audio.item()), 4),
                    "perda_consistencia": round(float(perda_consist.item()), 4),
                    "elapsed_s": round(time.time() - inicio, 1),
                }
                historico.append(registro)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(registro, ensure_ascii=False) + "\n")
                print(
                    f"  [multimodal] passo {passo:>4}/{total_passos} | perda {perda.item():.4f} | "
                    f"{registro['elapsed_s']:.0f}s"
                )

            if passo % self.config.intervalo_avaliacao == 0 or passo == total_passos:
                metricas = self.avaliar(dataset_img, dataset_audio)
                print(f"  [multimodal][validação] {metricas}")

        self.salvar()
        resumo = {
            "modalidade": "multimodal",
            "passos": passo,
            "tempo_s": round(time.time() - inicio, 1),
            "parametros_projetores": sum(p.numel() for p in self.parametros),
            "historico": historico[-10:],
        }
        print(json.dumps(resumo, indent=2, ensure_ascii=False))
        return resumo

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def avaliar(self, dataset_img, dataset_audio) -> Dict[str, float]:
        self.projetor_visao.eval()
        self.projetor_audio.eval()
        emb_img: List[torch.Tensor] = []
        emb_txt_img: List[torch.Tensor] = []
        emb_audio: List[torch.Tensor] = []
        emb_txt_audio: List[torch.Tensor] = []
        for i, item in enumerate(dataset_img):
            if i >= 64:
                break
            emb_img.append(self.projetor_visao(self.visao(item.imagem.unsqueeze(0)))[0])
            emb_txt_img.append(self.embeddings_texto([item.legenda or "uma imagem"])[0])
        for i, item in enumerate(dataset_audio):
            if i >= 64:
                break
            emb_audio.append(self.projetor_audio(self.audio(item.mel.unsqueeze(0)))[0])
            emb_txt_audio.append(self.embeddings_texto([item.transcricao or "uma fala"])[0])

        metricas: Dict[str, Any] = {}
        if len(emb_img) >= 8:
            metricas["imagem_texto"] = avaliar_recuperacao(
                torch.stack(emb_img), torch.stack(emb_txt_img),
                temperatura=self.config.temperatura,
            )
        if len(emb_audio) >= 8:
            metricas["audio_texto"] = avaliar_recuperacao(
                torch.stack(emb_audio), torch.stack(emb_txt_audio),
                temperatura=self.config.temperatura,
            )
        metricas["observacao"] = (
            "alinhamento de representações; o modelo não gera texto a partir "
            "de imagem/áudio (não há decoder de linguagem)."
        )
        self.projetor_visao.train()
        self.projetor_audio.train()
        return metricas

    # ------------------------------------------------------------------ #
    def salvar(self, nome: str = "final") -> Path:
        saida = Path(self.config.saida) / nome
        saida.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "projetor_visao": self.projetor_visao.state_dict(),
                "projetor_audio": self.projetor_audio.state_dict(),
            },
            saida / "projetores.pt",
        )
        (saida / "config.json").write_text(
            json.dumps(self.config.para_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (saida / "metadados.json").write_text(
            json.dumps(
                {
                    "parametros": sum(p.numel() for p in self.parametros),
                    "treinado_em": time.time(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return saida


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento multimodal (alinhamento)")
    parser.add_argument("--passos", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-itens", type=int, default=200)
    parser.add_argument("--saida", default="models/multimodal")
    parser.add_argument("--modelo-visao", default="models/vision/final/modelo.pt")
    parser.add_argument("--modelo-audio", default="models/audio/final/modelo.pt")
    args = parser.parse_args()

    config = ConfigTreinoMultimodal(
        passos=args.passos,
        batch_size=args.batch_size,
        lr=args.lr,
        max_itens=args.max_itens,
        saida=args.saida,
        modelo_visao=args.modelo_visao,
        modelo_audio=args.modelo_audio,
    )
    MultimodalTrainer(config).treinar()


if __name__ == "__main__":
    main()
