"""Configuração central dos modelos do Apolo Language Core.

A configuração é serializável em JSON para que checkpoints, experiments e
servidores possam versionar a arquitetura exata que gerou cada conjunto de
pesos.  Nenhum valor é inventado em tempo de execução: todos os campos possuem
padrões explícitos e podem ser sobrescritos por arquivos YAML/CLI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Hiperparâmetros de arquitetura do Transformer decoder-only.

    Atributos
    ---------
    vocab_size:
        Tamanho do vocabulário do tokenizer.  Deve corresponder exatamente ao
        tokenizer carregado junto do checkpoint.
    n_layer:
        Número de blocos Transformer.
    n_head:
        Número de cabeças de auto-atenção.
    n_embd:
        Dimensão do espaço residual.
    n_kv_head:
        Número de cabeças de chave/valor (GQA).  ``None`` equivale a ``n_head``.
    block_size:
        Comprimento máximo de contexto (em tokens).
    rope_theta:
        Base da codificação posicional rotacional (RoPE).
    rms_norm_eps:
       psilon do RMSNorm.
    dropout:
        Taxa de dropout aplicada nas projeções e no MLP.
    tie_embeddings:
        Compartilha os pesos entre a tabela de embeddings e a cabeça de saída.
    bias:
        Usa bias nas projeções lineares.
    mlp_hidden:
        Dimensão interna do MLP SwiGLU.  Calculada automaticamente se ``None``.
    mlp_multiple_of:
        Arredondamento da dimensão interna do MLP (multiploot para hardware).
    init_std:
        Desvio padrão da inicialização normal dos pesos.
    """

    vocab_size: int = 8192
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    n_kv_head: int | None = None
    block_size: int = 256
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    bias: bool = False
    mlp_hidden: int | None = None
    mlp_multiple_of: int = 64
    init_std: float = 0.02
    use_cache: bool = True

    # ------------------------------------------------------------------ #
    # Validação e normalização
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.n_embd <= 0 or self.n_head <= 0 or self.n_layer <= 0:
            raise ValueError("n_embd, n_head e n_layer devem ser positivos")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) precisa ser divisível por n_head ({self.n_head})"
            )
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) precisa ser divisível por n_kv_head ({self.n_kv_head})"
            )
        if self.block_size <= 0:
            raise ValueError("block_size deve ser positivo")
        if self.mlp_hidden is None:
            # Regra estilo LLaMA: dim interna ~ 8/3 * d_model, arredondada.
            bruto = int(8 * self.n_embd / 3)
            mult = max(1, self.mlp_multiple_of)
            self.mlp_hidden = ((bruto + mult - 1) // mult) * mult
        if self.mlp_hidden <= 0:
            raise ValueError("mlp_hidden deve ser positivo")

    @property
    def head_dim(self) -> int:
        """Dimensão de cada cabeça de atenção."""
        return self.n_embd // self.n_head

    @property
    def n_parameters(self) -> int:
        """Número estimado de parâmetros (sem contar biases)."""
        d = self.n_embd
        h = self.mlp_hidden
        kv = self.n_kv_head * self.head_dim
        # Embeddings + cabeça de saída (com ou sem compartilhamento)
        embed = self.vocab_size * d
        lm_head = 0 if self.tie_embeddings else self.vocab_size * d
        # Por bloco: atenção (q, k, v, proj) + MLP SwiGLU (gate, up, down) + 2 norms
        attn = d * d + 2 * d * kv + d * d
        mlp = 3 * d * h
        norms = 2 * d
        total = embed + lm_head + self.n_layer * (attn + mlp + norms) + d
        return total

    # ------------------------------------------------------------------ #
    # Serialização
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        campos = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        desconhecidos = set(data) - campos
        if desconhecidos:
            raise ValueError(f"Campos desconhecidos na configuração: {sorted(desconhecidos)}")
        return cls(**data)

    def save(self, caminho: str | Path) -> Path:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return caminho

    @classmethod
    def load(cls, caminho: str | Path) -> "ModelConfig":
        dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
        return cls.from_dict(dados)
