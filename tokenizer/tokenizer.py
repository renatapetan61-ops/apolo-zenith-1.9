"""Tokenizer do Apolo Zenith 1.9.

Fachada de alto nível sobre o :class:`~tokenizer.bpe.ByteLevelBPE`, responsável
por:

* treino do vocabulário a partir de textos (português, inglês e outros);
* codificação/decodificação com tratamento de tokens especiais;
* persistência em disco (formato JSON autocontido);
* intercâmbio com os módulos multimodais (vision/audio/video/tools).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tokenizer.bpe import ByteLevelBPE, pretokenize
from tokenizer.vocabulary import SPECIAL_TOKENS, ids_especiales


class BPETokenizer:
    """Tokenizer byte-level BPE do Apolo."""

    def __init__(self, bpe: ByteLevelBPE) -> None:
        self.bpe = bpe
        self.special_tokens: Dict[str, int] = dict(bpe.special_tokens)
        self._pad_id = self.special_tokens.get("<|pad|>")
        self._bos_id = self.special_tokens.get("<|bos|>")
        self._eos_id = self.special_tokens.get("<|eos|>")
        self._unk_id = self.special_tokens.get("<|unk|>")
        self._pad_token = "<|pad|>"
        self._re_special = re.compile(
            "(" + "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)) + ")"
        )

    # ------------------------------------------------------------------ #
    # Propriedades
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return self.bpe.vocab_size

    @property
    def pad_id(self) -> Optional[int]:
        return self._pad_id

    @property
    def bos_id(self) -> Optional[int]:
        return self._bos_id

    @property
    def eos_id(self) -> Optional[int]:
        return self._eos_id

    @property
    def unk_id(self) -> Optional[int]:
        return self._unk_id

    def token_id(self, nome: str) -> Optional[int]:
        return self.special_tokens.get(nome)

    # ------------------------------------------------------------------ #
    # Codificação
    # ------------------------------------------------------------------ #
    def encode(
        self,
        texto: str,
        add_bos: bool = False,
        add_eos: bool = False,
        allow_special: bool = False,
    ) -> List[int]:
        """Codifica texto em IDs.

        Tokens especiais presentes no texto só são convertidos em IDs quando
        ``allow_special=True`` (por segurança, o texto do usuário não consegue
        injetar controles de formato).
        """
        ids: List[int] = []
        if add_bos and self._bos_id is not None:
            ids.append(self._bos_id)
        for parte in self._re_special.split(texto):
            if not parte:
                continue
            if parte in self.special_tokens:
                if allow_special:
                    ids.append(self.special_tokens[parte])
                else:
                    ids.extend(self.bpe.encode_text(parte))
            else:
                ids.extend(self.bpe.encode_text(parte))
        if add_eos and self._eos_id is not None:
            ids.append(self._eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self.bpe.decode_ids(ids, skip_special=skip_special_tokens)

    def encode_batch(self, textos: Sequence[str], **kwargs) -> List[List[int]]:
        return [self.encode(t, **kwargs) for t in textos]

    # ------------------------------------------------------------------ #
    # Treino
    # ------------------------------------------------------------------ #
    @classmethod
    def train(
        cls,
        textos: Iterable[str],
        vocab_size: int = 8192,
        special_tokens: Optional[Dict[str, int]] = None,
        min_frequencia: int = 2,
        max_pieces: int = 400_000,
        verbose: bool = False,
    ) -> "BPETokenizer":
        """Treina o tokenizer a partir de um iterável de textos.

        ``max_pieces`` limita o número de peças distintas usadas para aprender
        as fusões (controle de tempo/memória em corpora muito grandes).
        """
        especiais = dict(special_tokens or ids_especiales())
        bpe = ByteLevelBPE(special_tokens=especiais)
        contador: Counter = Counter()
        total = 0
        for texto in textos:
            for peca in pretokenize(texto):
                contador[peca] += 1
                total += 1
        if verbose:
            print(f"  [tokenizer] peças distintas={len(contador)} tokens brutos={total}")
        if len(contador) > max_pieces:
            # Mantém as peças mais frequentes ( heurística simples de cobertura).
            mais_comuns = contador.most_common(max_pieces)
            contador = Counter(dict(mais_comuns))
            if verbose:
                print(f"  [tokenizer] limitado a {max_pieces} peças distintas")
        bpe.train(contador, vocab_size_alvo=vocab_size, min_frequencia=min_frequencia, verbose=verbose)
        if verbose:
            print(f"  [tokenizer] vocabulário final={bpe.vocab_size}")
        return cls(bpe)

    # ------------------------------------------------------------------ #
    # Persistência
    # ------------------------------------------------------------------ #
    def save(self, diretorio: str | Path) -> Path:
        diretorio = Path(diretorio)
        diretorio.mkdir(parents=True, exist_ok=True)
        caminho = diretorio / "tokenizer.json"
        self.bpe.save(caminho)
        metadados = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
        }
        (diretorio / "tokenizer_meta.json").write_text(
            json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return caminho

    @classmethod
    def load(cls, diretorio: str | Path) -> "BPETokenizer":
        diretorio = Path(diretorio)
        caminho = diretorio / "tokenizer.json"
        if not caminho.exists():
            raise FileNotFoundError(f"Tokenizer não encontrado em {caminho}")
        return cls(ByteLevelBPE.load(caminho))

    # ------------------------------------------------------------------ #
    # Utilidades
    # ------------------------------------------------------------------ #
    def encode_dataset(self, textos: Iterable[str], add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Codifica uma lista de textos em um fluxo único de tokens."""
        fluxo: List[int] = []
        for texto in textos:
            fluxo.extend(self.encode(texto, add_bos=add_bos, add_eos=add_eos))
        return fluxo

    def __len__(self) -> int:  # pragma: no cover - conveniência
        return self.vocab_size

    def __repr__(self) -> str:  # pragma: no cover - conveniência
        return f"BPETokenizer(vocab_size={self.vocab_size})"
