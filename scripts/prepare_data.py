"""Prepara o corpus para o Apolo Zenith 1.9.

Executa o pipeline completo de dados:

1. lê o manifesto de proveniência/licenças;
2. normaliza, remove PII e filtra documentos de baixa qualidade;
3. remove duplicatas (exatas e aproximadas);
4. divide em treino/validação/teste;
5. treina o tokenizer BPE **apenas** no split de treino;
6. codifica cada split em binário uint16 pronto para o Trainer.

Uso::

    python -m scripts.prepare_data --manifesto datasets_storage/manifest.json \
        --saida datasets_storage/processed --vocab 8192
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.text_loader import (  # noqa: E402
    deduplicar,
    dividir_documentos,
    normalizar_texto,
    qualidade_texto,
    remover_pii,
    salvar_tokens,
)
from tokenizer.tokenizer import BPETokenizer  # noqa: E402


def carregar_documentos(manifesto: Path, raiz: Path) -> List[Dict]:
    """Lê o manifesto e devolve a lista de documentos com metadados."""
    registros = json.loads(manifesto.read_text(encoding="utf-8"))
    documentos: List[Dict] = []
    for registro in registros:
        caminho = raiz / registro["arquivo"]
        if not caminho.exists():
            print(f"  [aviso] arquivo ausente: {caminho}")
            continue
        documentos.append({"registro": registro, "caminho": caminho})
    return documentos


def processar(documentos: List[Dict], raiz: Path) -> tuple[List[str], Dict]:
    """Aplica limpeza, filtro de qualidade, PII e deduplicação."""
    textos: List[str] = []
    stats = {"lidos": 0, "aprovados": 0, "rejeitados": 0, "motivos": {}}
    for item in documentos:
        stats["lidos"] += 1
        registro = item["registro"]
        bruto = item["caminho"].read_text(encoding="utf-8", errors="replace")
        fonte = registro.get("fonte", "")
        if fonte.startswith("gutenberg"):
            fonte_tipo = "gutenberg"
        elif fonte.startswith("wikisource"):
            fonte_tipo = "wikisource"
        else:
            fonte_tipo = "gutenberg"
        texto = normalizar_texto(bruto, fonte=fonte_tipo)
        texto = remover_pii(texto)
        aprovado, motivo = qualidade_texto(texto)
        if not aprovado:
            stats["rejeitados"] += 1
            stats["motivos"][motivo] = stats["motivos"].get(motivo, 0) + 1
            continue
        stats["aprovados"] += 1
        textos.append(texto)
    return textos, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara o corpus de treino do Apolo")
    parser.add_argument("--manifesto", default="datasets_storage/manifest.json")
    parser.add_argument("--raiz", default="datasets_storage")
    parser.add_argument("--saida", default="datasets_storage/processed")
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--min-frequencia", type=int, default=3)
    parser.add_argument("--proporcoes", default="0.97,0.015,0.015")
    args = parser.parse_args()

    inicio = time.time()
    raiz = Path(args.raiz)
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    print("[1/5] carregando documentos do manifesto…")
    documentos = carregar_documentos(Path(args.manifesto), raiz)

    print("[2/5] limpando, filtrando qualidade e removendo PII…")
    textos, stats = processar(documentos, raiz)
    print(f"      {stats['aprovados']}/{stats['lidos']} documentos aprovados; motivos={stats['motivos']}")

    print("[3/5] removendo duplicatas…")
    textos, removidas = deduplicar(textos)
    print(f"      {removidas} duplicatas removidas; {len(textos)} documentos finais")

    print("[4/5] dividindo em treino/validação/teste…")
    proporcoes = tuple(float(x) for x in args.proporcoes.split(","))
    splits = dividir_documentos(textos, proporcoes=proporcoes)  # type: ignore[arg-type]

    for nome, docs in splits.items():
        caminho = saida / f"{nome}.txt"
        caminho.write_text("\n\n".join(docs), encoding="utf-8")
        print(f"      {nome}: {len(docs)} documentos, {caminho.stat().st_size/1024/1024:.2f} MB")

    print(f"[5/5] treinando tokenizer BPE (vocab={args.vocab})…")
    tokenizer = BPETokenizer.train(
        splits["train"],
        vocab_size=args.vocab,
        min_frequencia=args.min_frequencia,
        verbose=True,
    )
    pasta_tokenizer = saida / "tokenizer"
    tokenizer.save(pasta_tokenizer)
    print(f"      tokenizer salvo em {pasta_tokenizer} (vocab={tokenizer.vocab_size})")

    contagens: Dict[str, int] = {}
    for nome, docs in splits.items():
        ids = tokenizer.encode_dataset(docs, add_bos=True, add_eos=True)
        caminho = saida / f"{nome}.bin"
        salvar_tokens(ids, caminho)
        contagens[nome] = len(ids)
        print(f"      {nome}: {len(ids)} tokens -> {caminho}")

    resumo = {
        "documentos": stats,
        "duplicadas_removidas": removidas,
        "tokens": contagens,
        "vocab_size": tokenizer.vocab_size,
        "tempo_s": round(time.time() - inicio, 2),
    }
    (saida / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
