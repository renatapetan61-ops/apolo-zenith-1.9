"""Harness de benchmarks do Apolo Zenith 1.9.

O harness executa **inferência real** sobre arquivos de benchmark presentes
localmente (formato JSONL) e calcula métricas reproduzíveis.  Nenhum resultado
é inventado: benchmarks ausentes simplesmente não são avaliados, e o relatório
registra explicitamente quais foram executados.

Formatos suportados (JSONL, um exemplo por linha):

* ``perda``      – ``{"texto": "...", "tokens": 123}`` (perplexidade)
* ``mult_escolha`` – ``{"pergunta": "...", "opcoes": ["a","b"], "resposta": 1}``
* ``matematica`` – ``{"pergunta": "...", "resposta": "42"}``
* ``codigo``     – ``{"task_id": "...", "prompt": "...", "testes": ["assert ..."]}``

Benchmarks oficiais (MMLU, GSM8K, MATH, HumanEval, MBPP, SWE-bench, MMBench,
MMMU) devem ser obtidos pelo usuário em suas fontes e convertidos para estes
formatos; a conversão automática não é feita aqui para respeitar licenças.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from evaluation.metrics import (
    acuracia_numerica,
    agregar_metricas,
    pass_at_k,
    perplexity,
)


# --------------------------------------------------------------------------- #
# Carregamento
# --------------------------------------------------------------------------- #
def carregar_jsonl(caminho: str | Path) -> List[Dict]:
    """Carrega um arquivo JSONL de benchmark."""
    caminho = Path(caminho)
    exemplos: List[Dict] = []
    with caminho.open("r", encoding="utf-8") as fh:
        for linha in fh:
            linha = linha.strip()
            if linha:
                exemplos.append(json.loads(linha))
    return exemplos


@dataclass
class Exemplo:
    """Exemplo normalizado de benchmark."""

    task_id: str
    prompt: str
    referencia: str = ""
    opcoes: List[str] = field(default_factory=list)
    testes: List[str] = field(default_factory=list)
    bruto: Dict[str, Any] = field(default_factory=dict)


def normalizar_exemplos(brutos: Sequence[Dict]) -> List[Exemplo]:
    """Converte registros brutos em :class:`Exemplo` padronizados."""
    saida: List[Exemplo] = []
    for i, bruto in enumerate(brutos):
        task_id = str(bruto.get("task_id", bruto.get("id", i)))
        prompt = bruto.get("prompt") or bruto.get("pergunta") or bruto.get("question") or bruto.get("text") or ""
        referencia = bruto.get("resposta") or bruto.get("answer") or bruto.get("reference") or bruto.get("target") or ""
        if isinstance(referencia, (dict, list)):
            referencia = json.dumps(referencia, ensure_ascii=False)
        opcoes = bruto.get("opcoes") or bruto.get("choices") or bruto.get("options") or []
        testes = bruto.get("testes") or bruto.get("tests") or []
        if isinstance(testes, str):
            testes = [testes]
        saida.append(
            Exemplo(
                task_id=task_id,
                prompt=str(prompt),
                referencia=str(referencia),
                opcoes=[str(o) for o in opcoes],
                testes=list(testes),
                bruto=bruto,
            )
        )
    return saida


# --------------------------------------------------------------------------- #
# Geração
# --------------------------------------------------------------------------- #
def _construir_prompt(exemplo: Exemplo, estilo: str) -> str:
    """Monta o prompt conforme o estilo da tarefa."""
    if estilo == "mult_escolha":
        opcoes = "\n".join(f"{chr(65+i)}) {o}" for i, o in enumerate(exemplo.opcoes))
        return (
            f"{exemplo.prompt}\n\n{opcoes}\n\n"
            "Responda apenas com a letra da alternativa correta.\nResposta:"
        )
    if estilo == "matematica":
        return (
            f"{exemplo.prompt}\n\n"
            "Resolva o problema passo a passo e finalize com 'Resposta: <valor>'.\n"
            "Resposta:"
        )
    if estilo == "codigo":
        return (
            f"{exemplo.prompt}\n\n"
            "Escreva apenas o código necessário, sem comentários extras:\n"
        )
    return exemplo.prompt


@torch.no_grad()
def gerar_respostas(
    modelo,
    tokenizer,
    exemplos: Sequence[Exemplo],
    estilo: str,
    max_new_tokens: int = 256,
    temperatura: float = 0.0,
    seed: int = 42,
) -> List[str]:
    """Gera respostas reais do modelo para cada exemplo."""
    modelo.eval()
    gerador = torch.Generator().manual_seed(seed)
    respostas: List[str] = []
    for exemplo in exemplos:
        prompt = _construir_prompt(exemplo, estilo)
        ids = tokenizer.encode(prompt)
        if not ids:
            respostas.append("")
            continue
        entrada = torch.tensor([ids[-modelo.config.block_size :]], dtype=torch.long)
        saida = modelo.generate(
            entrada,
            max_new_tokens=max_new_tokens,
            temperature=temperatura,
            top_k=None if temperatura == 0 else 50,
            top_p=None if temperatura == 0 else 0.95,
            stop_token_ids=[tokenizer.eos_id] if tokenizer.eos_id is not None else None,
            generator=gerador,
        )
        respostas.append(tokenizer.decode(saida[0].tolist()))
    return respostas


# --------------------------------------------------------------------------- #
# Execução de código (para pass@k)
# --------------------------------------------------------------------------- #
def executar_testes(codigo: str, testes: Sequence[str], timeout: int = 10) -> bool:
    """Executa o código gerado contra os testes em um subprocesso isolado.

    Retorna ``True`` apenas se todos os testes passarem.
    """
    if not testes:
        return False
    programa = "\n\n".join([codigo, *testes])
    with tempfile.TemporaryDirectory() as pasta:
        arquivo = Path(pasta) / "candidato.py"
        arquivo.write_text(programa, encoding="utf-8")
        try:
            resultado = subprocess.run(
                [sys.executable, str(arquivo)],
                capture_output=True,
                timeout=timeout,
                cwd=pasta,
            )
            return resultado.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except OSError:
            return False


# --------------------------------------------------------------------------- #
# Tarefas
# --------------------------------------------------------------------------- #
def avaliar_perda(modelo, tokenizer, exemplos: Sequence[Exemplo], block_size: int = 256) -> Dict:
    """Calcula a perplexidade média sobre os exemplos (avaliação causal real)."""
    modelo.eval()
    perdas: List[float] = []
    total_tokens = 0
    with torch.no_grad():
        for exemplo in exemplos:
            ids = tokenizer.encode(exemplo.prompt)
            if len(ids) < 2:
                continue
            entrada = torch.tensor([ids[:-1]], dtype=torch.long)
            alvos = torch.tensor([ids[1:]], dtype=torch.long)
            logits, _ = modelo(entrada)
            # logits[j] prevê o token ids[j+1].
            perda = torch.nn.functional.cross_entropy(
                logits[0].float(), alvos[0], reduction="mean"
            )
            perdas.append(float(perda.item()))
            total_tokens += len(ids) - 1
    if not perdas:
        return {"perda": None, "perplexidade": None, "n": 0}
    media = sum(perdas) / len(perdas)
    return {
        "perda": round(media, 6),
        "perplexidade": round(perplexity(media), 4),
        "n": len(perdas),
        "tokens": total_tokens,
    }


def avaliar_mult_escolha(
    modelo, tokenizer, exemplos: Sequence[Exemplo], max_new_tokens: int = 8
) -> Dict:
    """Avalia múltipla escolha por log-likelihood de cada alternativa.

    A escolha é feita pela maior log-probabilidade da continuação (critério
    determinístico, equivalente ao uso no MMLU).
    """
    modelo.eval()
    acertos: List[float] = []
    with torch.no_grad():
        for exemplo in exemplos:
            if not exemplo.opcoes:
                continue
            prompt = _construir_prompt(exemplo, "mult_escolha")
            ids_prompt = tokenizer.encode(prompt)
            if not ids_prompt:
                continue
            melhores: List[float] = []
            for opcao in exemplo.opcoes:
                ids_opcao = tokenizer.encode(" " + opcao)
                if not ids_opcao:
                    melhores.append(-1e9)
                    continue
                # O contexto precisa caber em block_size, preservando a opção.
                max_prompt = max(1, modelo.config.block_size - len(ids_opcao) - 1)
                ids_prompt_ctx = ids_prompt[-max_prompt:]
                entrada = ids_prompt_ctx + ids_opcao
                if len(entrada) < 2:
                    melhores.append(-1e9)
                    continue
                tensor = torch.tensor([entrada], dtype=torch.long)
                logits, _ = modelo(tensor)
                inicio = len(ids_prompt_ctx)  # índice do 1º token da opção
                logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
                fatia = logprobs[inicio - 1 : inicio - 1 + len(ids_opcao)]
                alvos = tensor[0, inicio : inicio + len(ids_opcao)]
                sel = fatia.gather(1, alvos.unsqueeze(1))
                melhores.append(float(sel.sum().item()))
            predicao = chr(65 + max(range(len(melhores)), key=lambda i: melhores[i]))
            referencia = exemplo.referencia.strip().upper()
            acertos.append(1.0 if referencia.startswith(predicao) else 0.0)
    resumo = agregar_metricas(acertos)
    resumo["tarefa"] = "mult_escolha"
    return resumo


def avaliar_matematica(modelo, tokenizer, exemplos: Sequence[Exemplo], max_new_tokens: int = 256) -> Dict:
    """Avalia acurácia numérica em problemas de matemática."""
    respostas = gerar_respostas(modelo, tokenizer, exemplos, "matematica", max_new_tokens)
    acertos = [acuracia_numerica(r, e.referencia) for r, e in zip(respostas, exemplos)]
    resumo = agregar_metricas(acertos)
    resumo["tarefa"] = "matematica"
    resumo["amostras"] = [
        {"task_id": e.task_id, "resposta_modelo": r[:200], "referencia": e.referencia}
        for e, r in list(zip(exemplos, respostas))[:10]
    ]
    return resumo


def avaliar_codigo(
    modelo,
    tokenizer,
    exemplos: Sequence[Exemplo],
    max_new_tokens: int = 384,
    n_amostras: int = 1,
    temperatura: float = 0.2,
    seed: int = 42,
) -> Dict:
    """Avalia pass@k executando os testes reais de cada problema."""
    scores: List[float] = []
    n_total = n_amostras
    for exemplo in exemplos:
        if not exemplo.testes:
            continue
        resolvidos = 0
        for _ in range(n_amostras):
            respostas = gerar_respostas(
                modelo,
                tokenizer,
                [exemplo],
                "codigo",
                max_new_tokens=max_new_tokens,
                temperatura=temperatura,
                seed=seed,
            )
            codigo = respostas[0]
            if executar_testes(codigo, exemplo.testes):
                resolvidos += 1
        scores.append(pass_at_k(n_total, resolvidos, 1))
    resumo = agregar_metricas(scores)
    resumo["tarefa"] = "codigo"
    resumo["pass@1"] = round(sum(scores) / len(scores), 6) if scores else 0.0
    return resumo


# --------------------------------------------------------------------------- #
# Orquestração
# --------------------------------------------------------------------------- #
TAREFAS: Dict[str, Callable] = {
    "perda": avaliar_perda,
    "mult_escolha": avaliar_mult_escolha,
    "matematica": avaliar_matematica,
    "codigo": avaliar_codigo,
}


def avaliar_benchmark(
    modelo,
    tokenizer,
    caminho: str | Path,
    tarefa: str = "perda",
    max_exemplos: int = 100,
    max_new_tokens: int = 256,
    seed: int = 42,
) -> Dict:
    """Executa um benchmark local e devolve o relatório da tarefa."""
    if tarefa not in TAREFAS:
        raise ValueError(f"Tarefa desconhecida: {tarefa}. Disponíveis: {sorted(TAREFAS)}")
    exemplos = normalizar_exemplos(carregar_jsonl(caminho))[:max_exemplos]
    if tarefa == "perda":
        return avaliar_perda(modelo, tokenizer, exemplos)
    if tarefa == "mult_escolha":
        return avaliar_mult_escolha(modelo, tokenizer, exemplos, max_new_tokens)
    if tarefa == "matematica":
        return avaliar_matematica(modelo, tokenizer, exemplos, max_new_tokens)
    return avaliar_codigo(modelo, tokenizer, exemplos, max_new_tokens, seed=seed)
