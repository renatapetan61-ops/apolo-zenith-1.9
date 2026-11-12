"""Apolo Language Core: modelo de linguagem decoder-only Transformer.

Este módulo materializa o núcleo neural compartilhado do ecossistema Apolo
Zenith 1.9.  Ele é um Transformer causal real (treinável), com RoPE,
RMSNorm, SwiGLU, KV Cache e geração incremental de tokens.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import ModelConfig
from core.embeddings import TokenEmbedding
from core.kv_cache import KVCache
from core.transformer import Transformer


class ApoloLanguageModel(nn.Module):
    """Modelo de linguagem autoregressivo do Apolo.

    O forward recebe tokens ``(B, T)`` e devolve logits ``(B, T, vocab_size)``.
    Quando ``targets`` são informados, devolve também a perda de entropia
    cruzada (usada diretamente pelo loop de treino).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = TokenEmbedding(config.vocab_size, config.n_embd)
        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            # Compartilhamento de pesos reduz parâmetros e estabiliza modelos
            # pequenos (o gradiente soma as duas direções).
            self.lm_head.weight = self.tok_emb.embedding.weight

        self.apply(self._init_weights)
        self._scale_residual_projections()

    # ------------------------------------------------------------------ #
    # Inicialização
    # ------------------------------------------------------------------ #
    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _scale_residual_projections(self) -> None:
        """Escala as projeções residuais por 1/sqrt(2 * n_layer).

        Técnica clássica (GPT-2) que evita o crescimento da variância das
        ativações em modelos com muitas camadas.
        """
        escala = 1.0 / math.sqrt(2 * self.config.n_layer)
        for block in self.transformer.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=self.config.init_std * escala)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=self.config.init_std * escala)

    # ------------------------------------------------------------------ #
    # Embeddings de texto (usados pelo alinhamento multimodal)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def embed_text(
        self,
        idx: torch.Tensor,
        mascara: torch.Tensor | None = None,
        agregar: str = "media",
    ) -> torch.Tensor:
        """Devolve o embedding de um texto (representação semântica).

        É o **mesmo** Transformer do Language Core, sem a cabeça de saída.
        Serve para alinhar multimodalidade: o embedding textual precisa
        estar no mesmo espaço do embedding de imagem/áudio.

        ``idx``: (B, T) tokens.  ``mascara``: (B, T) com 1 nos tokens válidos.
        ``agregar``: ``"media"`` (média ponderada) ou ``"ultimo"``.
        """
        if idx.dim() != 2:
            raise ValueError(f"idx deve ter shape (B, T), recebido {tuple(idx.shape)}")
        x = self.tok_emb(idx)
        x = self.transformer(x)
        if agregar == "ultimo":
            return x[:, -1]
        if mascara is None:
            return x.mean(dim=1)
        mascara = mascara.to(x.dtype).unsqueeze(-1)
        soma = (x * mascara).sum(dim=1)
        total = mascara.sum(dim=1).clamp(min=1e-6)
        return soma / total

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if idx.dim() != 2:
            raise ValueError(f"idx deve ter shape (B, T), recebido {tuple(idx.shape)}")
        b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(
                f"Sequência com {t} tokens excede block_size={self.config.block_size}"
            )

        offset = cache.length if cache is not None else 0
        x = self.tok_emb(idx)
        x = self.transformer(x, cache=cache, position_offset=offset)
        logits = self.lm_head(x)

        loss: torch.Tensor | None = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets deve ter o mesmo shape de idx")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=-100,
            )

        if cache is not None:
            cache.length = offset + t
        return logits, loss

    # ------------------------------------------------------------------ #
    # Geração
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        repetition_penalty: float = 1.0,
        stop_token_ids: Sequence[int] | None = None,
        generator: torch.Generator | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Gera tokens autoregressivamente.

        ``temperature=0`` faz decodificação gulosa (argmax).  O KV Cache é
        utilizado por padrão, reduzindo o custo por token de O(T^2) para O(T).
        """
        was_training = self.training
        self.eval()
        device = idx.device
        b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError("Prompt excede block_size do modelo")

        cache: KVCache | None = None
        if use_cache:
            cache = KVCache(
                n_layer=self.config.n_layer,
                max_seq_len=self.config.block_size,
                batch_size=b,
                n_kv_head=self.config.n_kv_head or self.config.n_head,
                head_dim=self.config.head_dim,
                device=device,
                dtype=next(self.parameters()).dtype,
            )

        saida = idx
        gerados: list[torch.Tensor] = []
        stops = set(stop_token_ids or [])
        limite = self.config.block_size
        atual = idx  # tokens enviados ao modelo no passo atual

        for _ in range(max_new_tokens):
            logits, _ = self(atual, cache=cache)
            logit = logits[:, -1, :].float()

            if repetition_penalty and repetition_penalty != 1.0:
                for token in set(gerados_token_ids(gerados)):
                    valores = logit[:, token]
                    logit[:, token] = torch.where(
                        valores > 0, valores / repetition_penalty, valores * repetition_penalty
                    )

            if temperature <= 0:
                proximo = torch.argmax(logit, dim=-1, keepdim=True)
            else:
                logit = logit / temperature
                if top_k is not None and top_k > 0:
                    k = min(top_k, logit.size(-1))
                    valores, indices = torch.topk(logit, k, dim=-1)
                    mascara = torch.full_like(logit, float("-inf"))
                    mascara.scatter_(1, indices, valores)
                    logit = mascara
                if top_p is not None and 0 < top_p < 1.0:
                    ordenados, indices = torch.sort(logit, descending=True, dim=-1)
                    probs = torch.softmax(ordenados, dim=-1)
                    acum = torch.cumsum(probs, dim=-1)
                    remover = acum - probs > top_p
                    remover[:, 0] = False
                    ordenados[remover] = float("-inf")
                    logit = torch.full_like(logit, float("-inf")).scatter_(1, indices, ordenados)
                probs = torch.softmax(logit, dim=-1)
                proximo = torch.multinomial(probs, num_samples=1, generator=generator)

            gerados.append(proximo)
            saida = proximo
            if cache is not None:
                # Com cache, basta enviar o novo token.
                atual = proximo
            else:
                # Sem cache, é preciso reenviar todo o contexto.
                atual = torch.cat([atual, proximo], dim=1)[:, -limite:]
            if proximo.numel() == 1 and int(proximo.item()) in stops:
                break
            if cache is not None and cache.length >= limite:
                break

        resultado = torch.cat(gerados, dim=1) if gerados else torch.empty((b, 0), dtype=torch.long, device=device)
        if was_training:
            self.train()
        return resultado

    # ------------------------------------------------------------------ #
    # Utilidades
    # ------------------------------------------------------------------ #
    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.tok_emb.embedding.weight.numel()
            if not self.config.tie_embeddings:
                total += self.config.vocab_size * self.config.n_embd
        return total

    def parameter_groups_for_decay(self, weight_decay: float) -> list[dict]:
        """Separa pesos que recebem decaimento dos que não recebem.

        Matrizes (>=2 dimensões) recebem decaimento; normalizações, biases e
        embeddings ficam de fora, conforme prática de treinar LLMs.
        """
        decay, sem_decay = [], []
        for nome, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay.append(param)
            else:
                sem_decay.append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": sem_decay, "weight_decay": 0.0},
        ]

    # ------------------------------------------------------------------ #
    # Persistência
    # ------------------------------------------------------------------ #
    def save(self, caminho: str | Path) -> Path:
        """Salva pesos + configuração em um único diretório de checkpoint."""
        caminho = Path(caminho)
        caminho.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), caminho / "model.pt")
        (caminho / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return caminho

    @classmethod
    def load(cls, caminho: str | Path, map_location: str | torch.device = "cpu") -> "ApoloLanguageModel":
        """Carrega um modelo salvo por :meth:`save` ou por um checkpoint de treino.

        Aceita tanto o diretório com ``config.json`` quanto o formato do
        ``CheckpointManager`` (configuração dentro de ``meta.json``).
        """
        caminho = Path(caminho)
        config_path = caminho / "config.json"
        if config_path.exists():
            config = ModelConfig.load(config_path)
        else:
            meta_path = caminho / "meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint inválido em {caminho}: 'config.json' ou 'meta.json' não encontrado."
                )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "model_config" not in meta:
                raise ValueError(f"Checkpoint em {caminho} não contém 'model_config'.")
            config = ModelConfig.from_dict(meta["model_config"])
        model = cls(config)
        estado = torch.load(caminho / "model.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(estado)
        model.eval()
        return model


def gerados_token_ids(gerados: Iterable[torch.Tensor]) -> list[int]:
    """Achata a lista de tokens gerados em uma lista de inteiros."""
    saida: list[int] = []
    for tensor in gerados:
        if tensor.numel() == 1:
            saida.append(int(tensor.item()))
        else:
            saida.extend(int(t) for t in tensor.reshape(-1).tolist())
    return saida
