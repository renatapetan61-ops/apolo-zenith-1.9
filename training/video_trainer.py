"""Treinador da modalidade **VÍDEO** (compreensão temporal).

O que é treinado de verdade:

* o ``VideoEncoder`` (encoder visual compartilhado + Transformer temporal);
* o ``VideoProjector`` (alinhamento com o Language Core);
* uma cabeça de **classificação temporal** que decide se a ordem dos frames
  foi preservada.

Objetivos (ambos reais e mensuráveis):

1. **Ordem temporal** — classificar sequências corretas vs. embaralhadas
   (métrica: acurácia de ordenação);
2. **Associação vídeo-texto** — InfoNCE contra o embedding do título
   (métrica: acurácia de recuperação, quando há amostras suficientes).

Limite honesto: com poucas dezenas de vídeos, a recuperação só é
interpretável acima de ~20 amostras; abaixo disso o relatório deixa isso
explícito em vez de apresentar um número enganoso.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from core.language_model import ApoloLanguageModel  # noqa: E402
from datasets.video_loader import ItemVideo, VideoDataset  # noqa: E402
from training.contraste import avaliar_recuperacao, perda_contrastiva  # noqa: E402
from video.encoder import ApoloVideoModel, VideoConfig  # noqa: E402


@dataclass
class ConfigTreinoVideo:
    """Hiperparâmetros do treino de vídeo."""

    pasta: str = "datasets/video/raw"
    saida: str = "models/video"
    tokenizer: str = "datasets_storage/processed/tokenizer"
    checkpoint_lm: str = "checkpoints/apolo-lm-tiny/checkpoints/best"
    max_frames: int = 8
    frame_size: int = 64
    amostra_fps: float = 1.0
    batch_size: int = 2
    passos: int = 150
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 20
    temperatura: float = 0.07
    peso_ordem: float = 0.5
    fracao_validacao: float = 0.25
    seed: int = 1337
    dim_projetor: int = 128
    dim_encoder: int = 192
    n_layer: int = 3
    n_head: int = 4
    max_tokens: int = 32
    intervalo_avaliacao: int = 25
    log: str = "logs/train_video.jsonl"

    def para_dict(self) -> Dict[str, Any]:
        return asdict(self)


def collate_video(lote: List[ItemVideo], max_frames: int = 8) -> Dict[str, Any]:
    """Agrupa frames (com padding) e títulos."""
    frames = []
    for item in lote:
        seq = item.frames
        if seq.shape[0] == 0:
            seq = torch.zeros(max_frames, 3, item.frames.shape[-2], item.frames.shape[-1])
        if seq.shape[0] > max_frames:
            seq = seq[:max_frames]
        frames.append(seq)
    alvo = max_frames
    lote_frames = torch.zeros(len(lote), alvo, *frames[0].shape[1:])
    mascara = torch.zeros(len(lote), alvo)
    for indice, seq in enumerate(frames):
        lote_frames[indice, : seq.shape[0]] = seq
        mascara[indice, : seq.shape[0]] = 1.0
    return {
        "frames": lote_frames,
        "mascara": mascara,
        "titulos": [item.titulo for item in lote],
        "identificadores": [item.identificador for item in lote],
    }


class VideoTrainer:
    """Treina o encoder temporal com ordem + associação texto."""

    def __init__(self, config: ConfigTreinoVideo) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.device = torch.device("cpu")

        cfg_video = VideoConfig(
            max_frames=config.max_frames,
            frame_size=config.frame_size,
            amostra_fps=config.amostra_fps,
            n_camadas_temporais=config.n_layer,
            n_head=config.n_head,
            dim_projetor=config.dim_projetor,
        )
        self.modelo = ApoloVideoModel(cfg_video).to(self.device)
        self.cabecalho_ordem = torch.nn.Linear(config.dim_projetor, 1).to(self.device)
        self.lm = self._carregar_lm()
        self.tokenizer = self._carregar_tokenizer()

        Path(config.saida).mkdir(parents=True, exist_ok=True)
        self.log_path = RAIZ / config.log
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def _carregar_lm(self) -> Optional[ApoloLanguageModel]:
        caminho = Path(self.config.checkpoint_lm)
        if not caminho.exists():
            return None
        try:
            modelo = ApoloLanguageModel.load(caminho, map_location=self.device)
            modelo.eval()
            for parametro in modelo.parameters():
                parametro.requires_grad_(False)
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
    def _dataloader(self):
        dataset = VideoDataset(
            self.config.pasta,
            max_frames=self.config.max_frames,
            amostra_fps=self.config.amostra_fps,
            tamanho=self.config.frame_size,
        )
        total = len(dataset)
        if total == 0:
            raise SystemExit(f"nenhum vídeo encontrado em {self.config.pasta}")
        corte = max(1, int(total * (1.0 - self.config.fracao_validacao)))
        indices = list(range(total))
        np.random.shuffle(indices)
        gerador = torch.Generator()
        gerador.manual_seed(self.config.seed)
        loader = DataLoader(
            Subset(dataset, indices[:corte]),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            generator=gerador,
            collate_fn=lambda lote: collate_video(lote, self.config.max_frames),
        )
        return loader, Subset(dataset, indices[corte:])

    # ------------------------------------------------------------------ #
    def _embaralhar(self, frames: torch.Tensor, mascara: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embaralha a ordem temporal dos frames e devolve o rótulo."""
        labels = torch.ones(frames.shape[0], dtype=torch.float32)
        for indice in range(frames.shape[0]):
            validas = int(mascara[indice].sum().item())
            if validas < 2:
                continue
            if torch.rand(()) < 0.5:
                perm = torch.randperm(validas)
                validas_idx = torch.nonzero(mascara[indice], as_tuple=False).flatten()
                frames[indice, :validas] = frames[indice, validas_idx[perm]]
                labels[indice] = 0.0
        return frames, labels

    def treinar(self) -> Dict[str, Any]:
        loader, validacao = self._dataloader()
        parametros = list(self.modelo.parameters()) + list(self.cabecalho_ordem.parameters())
        otimizador = torch.optim.AdamW(
            parametros,
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
        melhor = {"acuracia_ordem": -1.0}
        iterator = iter(loader)

        while passo < total_passos:
            try:
                lote = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                lote = next(iterator)

            frames, mascara = lote["frames"], lote["mascara"]
            frames, labels = self._embaralhar(frames.clone(), mascara)

            emb_video = self.modelo(frames)
            emb_texto = self.embeddings_texto(
                [t or "vídeo" for t in lote["titulos"]]
            )
            perda_contraste, _ = perda_contrastiva(
                emb_video, emb_texto, temperatura=self.config.temperatura
            )
            logits = self.cabecalho_ordem(emb_video).squeeze(-1)
            perda_ordem = F.binary_cross_entropy_with_logits(logits, labels)
            perda = perda_contraste + self.config.peso_ordem * perda_ordem

            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(parametros, 1.0)
            otimizador.step()
            escalonador.step()
            passo += 1

            if passo % 10 == 0 or passo == 1:
                registro = {
                    "passo": passo,
                    "perda": round(float(perda.item()), 4),
                    "perda_contraste": round(float(perda_contraste.item()), 4),
                    "perda_ordem": round(float(perda_ordem.item()), 4),
                    "lr": round(escalonador.get_last_lr()[0], 8),
                    "elapsed_s": round(time.time() - inicio, 1),
                }
                historico.append(registro)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(registro, ensure_ascii=False) + "\n")
                print(
                    f"  [video] passo {passo:>4}/{total_passos} | perda {perda.item():.4f} "
                    f"(contraste {perda_contraste.item():.4f} | ordem {perda_ordem.item():.4f}) | "
                    f"{registro['elapsed_s']:.0f}s"
                )

            if passo % self.config.intervalo_avaliacao == 0 or passo == total_passos:
                metricas = self.avaliar(validacao)
                print(f"  [video][validação] {metricas}")
                if metricas.get("acuracia_ordem", -1) > melhor.get("acuracia_ordem", -1):
                    melhor = metricas
                    self.salvar("melhor", metrica=metricas.get("acuracia_ordem"))

        self.salvar("final")
        resumo = {
            "modalidade": "video",
            "passos": passo,
            "tempo_s": round(time.time() - inicio, 1),
            "parametros_encoder": self.modelo.num_parameters(),
            "melhor_validacao": melhor,
            "historico": historico[-10:],
        }
        print(json.dumps(resumo, indent=2, ensure_ascii=False))
        return resumo

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def avaliar(self, validacao) -> Dict[str, float]:
        """Acurácia da ordem temporal (chave) e recuperação de texto."""
        self.modelo.eval()
        corretos = 0
        total = 0
        emb_video: List[torch.Tensor] = []
        emb_texto: List[torch.Tensor] = []
        for item in validacao:
            for embaralhado in (False, True):
                frames = item.frames.unsqueeze(0).clone()
                if embaralhado and frames.shape[1] > 1:
                    perm = torch.randperm(frames.shape[1])
                    frames = frames[:, perm]
                embedding = self.modelo(frames)[0]
                logit = float(self.cabecalho_ordem(embedding).item())
                previsto = 1 if logit > 0 else 0
                esperado = 0 if embaralhado else 1
                corretos += int(previsto == esperado)
                total += 1
                if not embaralhado:
                    emb_video.append(embedding)
                    emb_texto.append(self.embeddings_texto([item.titulo or "vídeo"])[0])
        metricas: Dict[str, float] = {
            "acuracia_ordem": round(corretos / max(total, 1), 4),
            "amostras_ordem": total,
        }
        if len(emb_video) >= 8:
            metricas.update(
                avaliar_recuperacao(
                    torch.stack(emb_video),
                    torch.stack(emb_texto),
                    temperatura=self.config.temperatura,
                )
            )
        else:
            metricas["r1"] = None
            metricas["observacao"] = (
                f"recuperação não calculada: apenas {len(emb_video)} amostras "
                "(mínimo 8 para interpretação)"
            )
        self.modelo.train()
        return metricas

    # ------------------------------------------------------------------ #
    def salvar(self, nome: str, metrica: Optional[float] = None) -> Path:
        saida = Path(self.config.saida) / nome
        saida.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"modelo": self.modelo.state_dict(), "cabecalho_ordem": self.cabecalho_ordem.state_dict()},
            saida / "modelo.pt",
        )
        (saida / "config.json").write_text(
            json.dumps(self.config.para_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (saida / "config_encoder.json").write_text(
            json.dumps(self.modelo.config.para_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (saida / "metadados.json").write_text(
            json.dumps(
                {
                    "parametros": self.modelo.num_parameters(),
                    "metrica_ordem": metrica,
                    "treinado_em": time.time(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return saida


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento da modalidade vídeo")
    parser.add_argument("--pasta", default="datasets/video/raw")
    parser.add_argument("--saida", default="models/video")
    parser.add_argument("--passos", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--checkpoint-lm", default="checkpoints/apolo-lm-tiny/checkpoints/best")
    args = parser.parse_args()

    config = ConfigTreinoVideo(
        pasta=args.pasta,
        saida=args.saida,
        passos=args.passos,
        batch_size=args.batch_size,
        lr=args.lr,
        max_frames=args.max_frames,
        checkpoint_lm=args.checkpoint_lm,
    )
    VideoTrainer(config).treinar()


if __name__ == "__main__":
    main()
