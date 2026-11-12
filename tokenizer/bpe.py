"""Implementação de Byte Level BPE (Byte Pair Encoding) do Apolo.

O tokenizer trabalha em nível de *bytes* UTF-8: cada byte vira um token base
(256 tokens) e o BPE aprende a fundir pares de bytes em tokens maiores.  Isso
garante cobertura total de qualquer idioma sem depender de um pré-tokenizador
específico por língua.

O treinamento do BPE é incremental: as contagens de pares são mantidas em
``Counter`` e atualizadas apenas para os pares afetados por cada fusão,
tornando o processo viável para vocabulários de 8k a 32k tokens.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tokenizer.vocabulary import SPECIAL_TOKENS


# --------------------------------------------------------------------------- #
# Pré-tokenização
# --------------------------------------------------------------------------- #
def _ler_token_texto(texto: str, i: int) -> Tuple[int, str]:
    """Lê um token de texto (palavra/número/pontuação) a partir da posição ``i``."""
    n = len(texto)
    c = texto[i]
    if c.isalnum() or c == "_":
        j = i
        while j < n:
            ch = texto[j]
            if ch.isalnum() or ch == "_":
                j += 1
            elif ch == "'" and j + 1 < n and texto[j + 1].isalpha():
                j += 2  # apóstrofo interno (ex.: "d'água", "l'água")
            else:
                break
        return j, texto[i:j]
    # Pontuação: Agrupa caracteres especiais consecutivos.
    j = i
    while j < n:
        ch = texto[j]
        if ch.isspace() or ch.isalnum() or ch == "_":
            break
        if ch == "'" and j + 1 < n and texto[j + 1].isalpha():
            break
        j += 1
    if j == i:
        j = i + 1
    return j, texto[i:j]


def pretokenize(texto: str) -> List[str]:
    """Divide o texto em peças pré-tokenizáveis.

    Espaços iniciais são "colados" à palavra seguinte (estilo GPT-2), o que
    reduz o número de tokens por frase em português.
    """
    pecas: List[str] = []
    i = 0
    n = len(texto)
    while i < n:
        c = texto[i]
        if c in " \t":
            j = i
            while j < n and texto[j] in " \t":
                j += 1
            espacos = texto[i:j]
            if j < n and texto[j] not in "\r\n":
                fim, token = _ler_token_texto(texto, j)
                pecas.append(espacos + token)
                i = fim
            else:
                pecas.append(espacos)
                i = j
        elif c in "\r\n":
            j = i
            while j < n and texto[j] in "\r\n":
                j += 1
            pecas.append(texto[i:j])
            i = j
        else:
            fim, token = _ler_token_texto(texto, i)
            pecas.append(token)
            i = fim
    return pecas


# --------------------------------------------------------------------------- #
# Núcleo BPE
# --------------------------------------------------------------------------- #
class ByteLevelBPE:
    """Modelo BPE em nível de byte, com fusões ordenadas."""

    def __init__(self, special_tokens: Optional[Dict[str, int]] = None) -> None:
        self.special_tokens: Dict[str, int] = dict(special_tokens or SPECIAL_TOKENS)
        self.id_to_bytes: Dict[int, bytes] = {}
        for nome, idx in self.special_tokens.items():
            self.id_to_bytes[idx] = nome.encode("utf-8")
        self.byte_base = len(self.special_tokens)
        for b in range(256):
            self.id_to_bytes[self.byte_base + b] = bytes([b])
        self.merges: Dict[Tuple[int, int], int] = {}
        self.merge_rank: Dict[Tuple[int, int], int] = {}
        self._cache: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------ #
    # Propriedades
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return len(self.id_to_bytes)

    def token_bytes(self, token_id: int) -> bytes:
        return self.id_to_bytes[token_id]

    # ------------------------------------------------------------------ #
    # Treinamento
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fundir_seq(seq: Sequence[int], par: Tuple[int, int], novo_id: int) -> Tuple[int, ...]:
        """Substitui todas as ocorrências não sobrepostas de ``par`` na sequência.

        ``novo_id`` é o id do token resultante da fusão; ele substitui o par
        original (jamais apenas uma das metades).
        """
        a, b = par
        saida: List[int] = []
        i = 0
        n = len(seq)
        while i < n:
            if i + 1 < n and seq[i] == a and seq[i + 1] == b:
                saida.append(novo_id)
                i += 2
            else:
                saida.append(seq[i])
                i += 1
        return tuple(saida)

    def train(
        self,
        contador: Counter,
        vocab_size_alvo: int,
        min_frequencia: int = 2,
        verbose: bool = False,
    ) -> None:
        """Treina as fusões BPE a partir de um contador de peças.

        ``contador`` mapeia peça pré-tokenizada -> frequência.
        """
        if vocab_size_alvo <= self.vocab_size:
            raise ValueError(
                f"vocab_size_alvo ({vocab_size_alvo}) deve ser maior que o tamanho base "
                f"({self.vocab_size})"
            )

        pecas = list(contador.keys())
        freqs = [contador[p] for p in pecas]
        seqs: List[Tuple[int, ...]] = [
            tuple(self.byte_base + b for b in p.encode("utf-8")) for p in pecas
        ]

        contagem: Counter = Counter()
        ocorrencias: Dict[Tuple[int, int], set] = defaultdict(set)
        for idx, seq in enumerate(seqs):
            for par in zip(seq, seq[1:]):
                contagem[par] += freqs[idx]
                ocorrencias[par].add(idx)

        heap: List[Tuple[int, Tuple[int, int]]] = [(-c, p) for p, c in contagem.items()]
        heapq.heapify(heap)

        def _empilhar(par: Tuple[int, int]) -> None:
            c = contagem.get(par)
            if c:
                heapq.heappush(heap, (-c, par))

        while len(self.id_to_bytes) < vocab_size_alvo and heap:
            neg, par = heapq.heappop(heap)
            atual = contagem.get(par, 0)
            if atual == 0 or -neg != atual:
                continue  # entrada obsoleta (lazy deletion)
            if atual < min_frequencia:
                break

            novo_id = len(self.id_to_bytes)
            self.id_to_bytes[novo_id] = self.id_to_bytes[par[0]] + self.id_to_bytes[par[1]]
            self.merges[par] = novo_id
            self.merge_rank[par] = len(self.merge_rank)

            for idx in list(ocorrencias.get(par, ())):
                seq = seqs[idx]
                f = freqs[idx]
                for p in zip(seq, seq[1:]):
                    contagem[p] -= f
                    if contagem[p] <= 0:
                        contagem.pop(p, None)
                    occ = ocorrencias.get(p)
                    if occ is not None:
                        occ.discard(idx)
                    _empilhar(p)
                novo = self._fundir_seq(seq, par, novo_id)
                seqs[idx] = novo
                for p in zip(novo, novo[1:]):
                    contagem[p] += f
                    ocorrencias[p].add(idx)
                    _empilhar(p)
            contagem.pop(par, None)
            ocorrencias.pop(par, None)

            if verbose and len(self.id_to_bytes) % 1000 == 0:
                print(f"  [bpe] vocab={len(self.id_to_bytes)} última fusão={par} freq={atual}")

    # ------------------------------------------------------------------ #
    # Codificação / decodificação
    # ------------------------------------------------------------------ #
    def encode_piece(self, peca: str) -> List[int]:
        """Codifica uma única peça pré-tokenizada."""
        if peca in self._cache:
            return self._cache[peca]
        ids = [self.byte_base + b for b in peca.encode("utf-8")]
        if len(ids) > 1:
            while True:
                melhor_rank = None
                melhor_par: Optional[Tuple[int, int]] = None
                for par in zip(ids, ids[1:]):
                    rank = self.merge_rank.get(par)
                    if rank is not None and (melhor_rank is None or rank < melhor_rank):
                        melhor_rank = rank
                        melhor_par = par
                if melhor_par is None:
                    break
                ids = list(self._fundir_seq(ids, melhor_par, self.merges[melhor_par]))
        self._cache[peca] = ids
        return ids

    def encode_text(self, texto: str) -> List[int]:
        """Codifica texto puro (sem tratar tokens especiais)."""
        ids: List[int] = []
        for peca in pretokenize(texto):
            ids.extend(self.encode_piece(peca))
        return ids

    def decode_ids(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """Decodifica IDs em texto (UTF-8, tolerante a bytes inválidos)."""
        buffer = bytearray()
        especiais = set(self.special_tokens.values())
        for token_id in ids:
            if skip_special and token_id in especiais:
                continue
            buffer.extend(self.id_to_bytes.get(token_id, b""))
        return buffer.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # Persistência
    # ------------------------------------------------------------------ #
    def save(self, caminho: str | Path) -> Path:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        dados = {
            "format": "apolo-bpe-v1",
            "special_tokens": self.special_tokens,
            "byte_base": self.byte_base,
            "merges": [[a, b, novo] for (a, b), novo in sorted(self.merges.items(), key=lambda kv: self.merge_rank[kv[0]])],
            "vocab_size": self.vocab_size,
        }
        caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
        return caminho

    @classmethod
    def load(cls, caminho: str | Path) -> "ByteLevelBPE":
        dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
        if dados.get("format") != "apolo-bpe-v1":
            raise ValueError(f"Formato de tokenizer desconhecido: {dados.get('format')}")
        bpe = cls(special_tokens=dados["special_tokens"])
        for a, b, novo in dados["merges"]:
            bpe.merges[(a, b)] = novo
            bpe.merge_rank[(a, b)] = len(bpe.merge_rank)
            bpe.id_to_bytes[novo] = bpe.id_to_bytes[a] + bpe.id_to_bytes[b]
        return bpe
