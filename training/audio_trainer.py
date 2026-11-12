"""Treinador da modalidade **ÁUDIO** (reconhecimento e associação).

O que é treinado de verdade:

* o ``MelEncoder`` (convolução 2D + Transformer sobre o mel-spectrogram);
* o ``AudioProjector`` (alinhamento com o espaço do Language Core).

Objetivo: **contrastivo** (InfoNCE) entre o embedding de áudio e o embedding
textual da transcrição, produced pelo Language Core treinado.  É um objetivo
de **associação áudio-texto** real e mensurável (recuperação).

Limite explícito: **não há decoder de geração de voz**.  Este módulo não
produz fala — apenas representa áudio e o associa a texto.

Uso::

    python -m training.audio_trainer --pasta datasets/audio/raw/librispeech/LibriSpeech \
        --saida models/audio --steps 200
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
from torch.utils.data import DataLoader, Subset  # noqa: E402

from audio.encoder import ApoloAudioModel, AudioConfig  # noqa: E402
from core.language_model import ApoloLanguageModel  # noqa: E402
from datasets.audio_loader import ItemAudio, LibriSpeechDataset  # noqa: E402
from training.contraste import avaliar_recuperacao, perda_contrastiva  # noqa: E402


@dataclass
class ConfigTreinoAudio:
    """Hiperparâmetros do treino de áudio."""

    pasta: str = "datasets/audio/raw/librispeech/LibriSpeech"
    saida: str = "models/audio"
    tokenizer: str = "datasets_storage/processed/tokenizer"
    checkpoint_lm: str = "checkpoints/apolo-lm-tiny/checkpoints/best"
    n_mels: int = 80
    max_seconds: float = 3.0
    batch_size: int = 4
    passos: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: int = 20
    temperatura: float = 0.07
    max_itens: Optional[int] = None
    fracao_validacao: float = 0.2
    seed: int = 1337
    dim_projetor: int = 128
    dim_encoder: int = 192
    n_layer: int = 3
    n_head: int = 4
    max_tokens: int = 32
    intervalo_avaliacao: int = 50
    log: str = "logs/train_audio.jsonl"

    def para_dict(self) -> Dict[str, Any]:
        return asdict(self)


def collate_audio(lote: List[ItemAudio], max_frames: int = 300) -> Dict[str, Any]:
    """Agrupa mels com padding (recorte + preenchimento) e transcrições."""
    mels = []
    for item in lote:
        mel = item.mel
        if mel.shape[1] > max_frames:
            mel = mel[:, :max_frames]
        mels.append(mel)
    largura = max(m.shape[1] for m in mels)
    lote_mel = torch.zeros(len(mels), mels[0].shape[0], largura)
    mascara = torch.zeros(len(mels), largura)
    for indice, mel in enumerate(mels):
        lote_mel[indice, :, : mel.shape[1]] = mel
        mascara[indice, : mel.shape[1]] = 1.0
    return {
        "mels": lote_mel,
        "mascara": mascara,
        "transcricoes": [item.transcricao for item in lote],
        "identificadores": [item.identificador for item in lote],
    }


class AudioTrainer:
    """Treina o encoder de áudio + projetor com objetivo contrastivo."""

    def __init__(self, config: ConfigTreinoAudio) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.device = torch.device("cpu")

        cfg_audio = AudioConfig(
            n_mels=config.n_mels,
            max_seconds=config.max_seconds,
            dim=config.dim_encoder,
            n_layer=config.n_layer,
            n_head=config.n_head,
            dim_projetor=config.dim_projetor,
        )
        self.modelo = ApoloAudioModel(cfg_audio).to(self.device)
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
        dataset = LibriSpeechDataset(
            self.config.pasta,
            n_mels=self.config.n_mels,
            max_seconds=self.config.max_seconds,
            max_itens=self.config.max_itens,
        )
        total = len(dataset)
        if total == 0:
            raise SystemExit(f"nenhum par áudio/texto encontrado em {self.config.pasta}")
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
            collate_fn=lambda lote: collate_audio(lote),
        )
        return loader, Subset(dataset, indices[corte:])

    # ------------------------------------------------------------------ #
    def treinar(self) -> Dict[str, Any]:
        loader, validacao = self._dataloader()
        otimizador = torch.optim.AdamW(
            self.modelo.parameters(),
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
        melhor = {"r1": -1.0}
        iterator = iter(loader)

        while passo < total_passos:
            try:
                lote = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                lote = next(iterator)

            emb_audio = self.modelo(lote["mels"])
            emb_texto = self.embeddings_texto(
                [t or "fala" for t in lote["transcricoes"]]
            )
            perda, _ = perda_contrastiva(
                emb_audio, emb_texto, temperatura=self.config.temperatura
            )

            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(self.modelo.parameters(), 1.0)
            otimizador.step()
            escalonador.step()
            passo += 1

            if passo % 10 == 0 or passo == 1:
                registro = {
                    "passo": passo,
                    "perda": round(float(perda.item()), 4),
                    "lr": round(escalonador.get_last_lr()[0], 8),
                    "elapsed_s": round(time.time() - inicio, 1),
                }
                historico.append(registro)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(registro, ensure_ascii=False) + "\n")
                print(
                    f"  [audio] passo {passo:>4}/{total_passos} | perda {perda.item():.4f} | "
                    f"{registro['elapsed_s']:.0f}s"
                )

            if passo % self.config.intervalo_avaliacao == 0 or passo == total_passos:
                metricas = self.avaliar(validacao)
                print(f"  [audio][validação] {metricas}")
                if metricas["r1"] > melhor["r1"]:
                    melhor = metricas
                    self.salvar("melhor", metrica=metricas["r1"])

        self.salvar("final")
        resumo = {
            "modalidade": "audio",
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
        self.modelo.eval()
        emb_audio: List[torch.Tensor] = []
        emb_texto: List[torch.Tensor] = []
        for item in validacao:
            emb_audio.append(self.modelo(item.mel.unsqueeze(0))[0])
            emb_texto.append(self.embeddings_texto([item.transcricao or "fala"])[0])
        if not emb_audio:
            return {"r1": 0.0, "amostras": 0}
        metricas = avaliar_recuperacao(
            torch.stack(emb_audio),
            torch.stack(emb_texto),
            temperatura=self.config.temperatura,
        )
        self.modelo.train()
        return metricas

    # ------------------------------------------------------------------ #
    def salvar(self, nome: str, metrica: Optional[float] = None) -> Path:
        saida = Path(self.config.saida) / nome
        saida.mkdir(parents=True, exist_ok=True)
        torch.save(self.modelo.state_dict(), saida / "modelo.pt")
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
                    "metrica_r1": metrica,
                    "treinado_em": time.time(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return saida


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento da modalidade áudio")
    parser.add_argument("--pasta", default="datasets/audio/raw/librispeech/LibriSpeech")
    parser.add_argument("--saida", default="models/audio")
    parser.add_argument("--passos", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-itens", type=int, default=None)
    parser.add_argument("--checkpoint-lm", default="checkpoints/apolo-lm-tiny/checkpoints/best")
    args = parser.parse_args()

    config = ConfigTreinoAudio(
        pasta=args.pasta,
        saida=args.saida,
        passos=args.passos,
        batch_size=args.batch_size,
        lr=args.lr,
        max_itens=args.max_itens,
        checkpoint_lm=args.checkpoint_lm,
    )
    AudioTrainer(config).treinar()


if __name__ == "__main__":
    main()
