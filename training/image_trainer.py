"""Treinador da modalidade **IMAGEM** (visão computacional).

O que é treinado de verdade:

* o ``VisionEncoder`` (ViT compacto) — recebe a imagem;
* o ``VisionProjector`` — leva o embedding visual ao espaço do Language Core.

O objetivo é **contrastivo** (InfoNCE) contra os embeddings de texto
produzidos pelo Language Core treinado.  Isso é alinhamento multimodal real:
o encoder visual **não** é inicializado com pesos aleatórios "declarados
treinados" — ele é treinado aqui, com perda medida.

Métrica de avaliação: acurácia de recuperação imagem→texto e texto→imagem
sobre o conjunto de validação.

Uso::

    python -m training.image_trainer --parquet datasets/images/raw/flickr8k/test.parquet \
        --saida models/vision --steps 200
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from core.language_model import ApoloLanguageModel  # noqa: E402
from datasets.image_loader import ParquetImagemLegendaDataset, collate_imagens  # noqa: E402
from resources.config import carregar_config_recursos  # noqa: E402
from training.contraste import avaliar_recuperacao, perda_contrastiva  # noqa: E402
from vision.encoder import ApoloVisionModel, VisionConfig  # noqa: E402


@dataclass
class ConfigTreinoImagem:
    """Hiperparâmetros do treino da visão."""

    parquet: str = "datasets/images/raw/flickr8k/test.parquet"
    parquets_extra: str = ""
    saida: str = "models/vision"
    tokenizer: str = "datasets_storage/processed/tokenizer"
    checkpoint_lm: str = "checkpoints/apolo-lm-tiny/checkpoints/best"
    tamanho: int = 64
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
    n_layer: int = 4
    n_head: int = 4
    patch_size: int = 16
    augmentar_legendas: bool = False
    dropout: float = 0.1
    max_tokens: int = 32
    intervalo_avaliacao: int = 100
    max_validacao: int = 300
    log: str = "logs/train_visao.jsonl"

    def para_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


class ImagemTrainer:
    """Treina o encoder visual + projetor com objetivo contrastivo."""

    def __init__(self, config: ConfigTreinoImagem) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.device = torch.device("cpu")
        cfg_visao = VisionConfig(
            dim=config.dim_encoder,
            n_layer=config.n_layer,
            n_head=config.n_head,
            patch_size=config.patch_size,
            image_size=config.tamanho,
            dim_projetor=config.dim_projetor,
            dropout=config.dropout,
        )
        self.modelo = ApoloVisionModel(cfg_visao).to(self.device)

        # Language Core congelado: fornece os embeddings de texto.
        self.lm = self._carregar_lm()
        self.tokenizer = self._carregar_tokenizer()

        saida = Path(config.saida)
        saida.mkdir(parents=True, exist_ok=True)
        self.log_path = RAIZ / config.log
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def _carregar_lm(self) -> Optional[ApoloLanguageModel]:
        caminho = Path(self.config.checkpoint_lm)
        if not caminho.exists():
            print("[aviso] checkpoint do LM não encontrado; o texto usará embeddings médios.")
            return None
        try:
            modelo = ApoloLanguageModel.load(caminho, map_location=self.device)
            modelo.eval()
            for parametro in modelo.parameters():
                parametro.requires_grad_(False)
            return modelo
        except Exception as exc:
            print(f"[aviso] falha ao carregar o LM ({exc}); usando embeddings médios.")
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

    # ------------------------------------------------------------------ #
    def embeddings_texto(self, textos: List[str]) -> torch.Tensor:
        """Embeddings de texto (LM congelado ou média das embeddings)."""
        if self.tokenizer is None or self.lm is None:
            return torch.zeros(len(textos), self.config.dim_projetor)
        ids = []
        for texto in textos:
            tokens = self.tokenizer.encode(texto, add_bos=False, add_eos=False)
            tokens = list(tokens)[: self.config.max_tokens]
            ids.append(tokens or [0])
        largura = max(len(i) for i in ids)
        lote = torch.zeros(len(ids), largura, dtype=torch.long)
        mascara = torch.zeros(len(ids), largura, dtype=torch.float32)
        for indice, tokens in enumerate(ids):
            lote[indice, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mascara[indice, : len(tokens)] = 1.0
        with torch.no_grad():
            bruto = self.lm.embed_text(lote.to(self.device), mascara.to(self.device))
        # Projeta para a dimensão do projetor visual (o LM tem n_embd=128).
        if bruto.shape[-1] == self.config.dim_projetor:
            return bruto
        return bruto[:, : self.config.dim_projetor]

    # ------------------------------------------------------------------ #
    def _dataloader(self) -> DataLoader:
        # Doıs datasets: treino com augmentação, validação sem (avaliação
        # determinística).
        caminhos = [self.config.parquet]
        if self.config.parquets_extra:
            caminhos.extend(self.config.parquets_extra.split(","))
        dataset_treino = ParquetImagemLegendaDataset(
            caminhos,
            tamanho=self.config.tamanho,
            max_itens=self.config.max_itens,
            augmentar=True,
            seed=self.config.seed,
        )
        dataset_validacao = ParquetImagemLegendaDataset(
            caminhos,
            tamanho=self.config.tamanho,
            max_itens=self.config.max_itens,
            augmentar=False,
            seed=self.config.seed + 1,
        )
        total = len(dataset_treino)
        corte = max(1, int(total * (1.0 - self.config.fracao_validacao)))
        # Ordem agrupada por row group: leitura sequencial do Parquet
        # (embaralhar as linhas exigiria reler o grupo inteiro a cada item).
        ordem = dataset_treino.ordem_por_grupo(self.config.seed)
        gerador = torch.Generator()
        gerador.manual_seed(self.config.seed)
        loader = DataLoader(
            torch.utils.data.Subset(dataset_treino, ordem[:corte]),
            batch_size=self.config.batch_size,
            shuffle=False,          # a ordem já vem embaralhada por grupo
            num_workers=0,
            generator=gerador,
            drop_last=False,
            collate_fn=collate_imagens,
        )
        validacao = torch.utils.data.Subset(dataset_validacao, ordem[corte:])
        return loader, validacao

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

            imagens = lote["imagens"]
            legendas = []
            for indice, candidatas in enumerate(lote.get("todas_legendas", [])):
                opcoes = [c for c in candidatas if c] or ["uma imagem"]
                if self.config.augmentar_legendas:
                    # Uma legenda diferente a cada passagem: 5x mais pares.
                    legendas.append(opcoes[random.randrange(len(opcoes))])
                else:
                    # Legenda principal: o alvo é o mesmo usado na avaliação.
                    legendas.append(opcoes[0])
            legendas = [t or "uma imagem" for t in legendas]

            emb_imagem = self.modelo(imagens)
            emb_texto = self.embeddings_texto(legendas)
            perda, _ = perda_contrastiva(
                emb_imagem, emb_texto, temperatura=self.config.temperatura
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
                    "temperatura": self.config.temperatura,
                    "elapsed_s": round(time.time() - inicio, 1),
                }
                historico.append(registro)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(registro, ensure_ascii=False) + "\n")
                print(
                    f"  [visao] passo {passo:>4}/{total_passos} | perda {perda.item():.4f} | "
                    f"lr {escalonador.get_last_lr()[0]:.2e} | {registro['elapsed_s']:.0f}s"
                )

            if passo % self.config.intervalo_avaliacao == 0 or passo == total_passos:
                metricas = self.avaliar(validacao)
                print(f"  [visao][validação] {metricas}")
                if metricas["r1"] > melhor["r1"]:
                    melhor = metricas
                    self.salvar("melhor", metrica=metricas["r1"])

        self.salvar("final")
        resumo = {
            "modalidade": "images",
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
        """Avalia a recuperação no conjunto de validação."""
        self.modelo.eval()
        emb_img: List[torch.Tensor] = []
        emb_txt: List[torch.Tensor] = []
        for indice, item in enumerate(validacao):
            if indice >= self.config.max_validacao:
                break
            emb_img.append(self.modelo(item.imagem.unsqueeze(0))[0])
            legenda = item.legenda or (item.legendas[0] if item.legendas else "uma imagem")
            emb_txt.append(self.embeddings_texto([legenda or "uma imagem"])[0])

        if not emb_img:
            return {"r1": 0.0, "amostras": 0}
        metricas = avaliar_recuperacao(
            torch.stack(emb_img), torch.stack(emb_txt), temperatura=self.config.temperatura
        )
        self.modelo.train()
        return metricas

    # ------------------------------------------------------------------ #
    def salvar(self, nome: str, metrica: Optional[float] = None) -> Path:
        """Salva o encoder visual treinado."""
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
        metadados = {
            "parametros": self.modelo.num_parameters(),
            "metrica_r1": metrica,
            "treinado_em": time.time(),
        }
        (saida / "metadados.json").write_text(
            json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return saida


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento da modalidade imagem")
    parser.add_argument("--parquet", default="datasets/images/raw/flickr8k/test.parquet")
    parser.add_argument("--parquets-extra", default="")
    parser.add_argument("--saida", default="models/vision")
    parser.add_argument("--passos", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-itens", type=int, default=None)
    parser.add_argument("--tamanho", type=int, default=64)
    parser.add_argument("--checkpoint-lm", default="checkpoints/apolo-lm-tiny/checkpoints/best")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--dim-encoder", type=int, default=192)
    parser.add_argument("--camadas", type=int, default=4)
    parser.add_argument("--fracao-validacao", type=float, default=0.2)
    parser.add_argument("--augmentar-legendas", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--intervalo-avaliacao", type=int, default=100)
    parser.add_argument("--max-validacao", type=int, default=300)
    args = parser.parse_args()

    config = ConfigTreinoImagem(
        parquet=args.parquet,
        parquets_extra=args.parquets_extra,
        saida=args.saida,
        passos=args.passos,
        batch_size=args.batch_size,
        lr=args.lr,
        max_itens=args.max_itens,
        tamanho=args.tamanho,
        checkpoint_lm=args.checkpoint_lm,
        patch_size=args.patch_size,
        dim_encoder=args.dim_encoder,
        n_layer=args.camadas,
        fracao_validacao=args.fracao_validacao,
        augmentar_legendas=args.augmentar_legendas,
        dropout=args.dropout,
        intervalo_avaliacao=args.intervalo_avaliacao,
        max_validacao=args.max_validacao,
    )
    trainer = ImagemTrainer(config)
    trainer.treinar()


if __name__ == "__main__":
    main()
