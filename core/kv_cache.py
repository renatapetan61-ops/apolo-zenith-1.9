"""Cache de chave/valor (KV Cache) para inferência incremental.

O cache é pré-alocado uma única vez e reutilizado durante toda a geração,
evitando o custo quadrático de recalcular o contexto a cada token.  É o mesmo
mecanismo usado na versão mobile do Apolo.
"""

from __future__ import annotations

import torch


class KVCache:
    """Armazena as chaves e valores de cada camada para autoregressão."""

    def __init__(
        self,
        n_layer: int,
        max_seq_len: int,
        batch_size: int,
        n_kv_head: int,
        head_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.n_layer = n_layer
        self.max_seq_len = max_seq_len
        self.length = 0
        shape = (batch_size, n_kv_head, max_seq_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Insere as chaves/valores da camada e devolve o histórico completo."""
        pos = self.length
        n = k.shape[-2]
        if pos + n > self.max_seq_len:
            raise ValueError(
                f"KV Cache excedido: {pos + n} > {self.max_seq_len}. "
                "Aumente block_size ou reduza max_new_tokens."
            )
        self.k[layer][:, :, pos : pos + n] = k
        self.v[layer][:, :, pos : pos + n] = v
        return self.k[layer][:, :, : pos + n], self.v[layer][:, :, : pos + n]

    def reset(self) -> None:
        """Limpa o cache mantendo a memória alocada."""
        for tensor in self.k:
            tensor.zero_()
        for tensor in self.v:
            tensor.zero_()
        self.length = 0

    def crop(self, tamanho: int) -> None:
        """Mantém apenas os primeiros ``tamanho`` tokens (branching de-beam)."""
        self.length = min(self.length, tamanho)
