"""Validação independente do checkpoint textual do Apolo Zenith.

Verifica integridade SHA-256, carregamento estrito, compatibilidade arquitetural
com o checkpoint de código, identidade do tokenizer, loss/perplexidade em probes
bilíngues e geração determinística após reinicialização.
As métricas dos probes são diagnósticas, não substituem um benchmark holdout amplo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from core.language_model import ApoloLanguageModel
from tokenizer.tokenizer import BPETokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_compatibility(code_model: ApoloLanguageModel, text_model: ApoloLanguageModel) -> dict[str, Any]:
    fields = ("vocab_size", "n_layer", "n_head", "n_embd", "n_kv_head",
              "block_size", "tie_embeddings", "mlp_hidden")
    code_cfg, text_cfg = code_model.config.to_dict(), text_model.config.to_dict()
    differences = {
        key: {"code": code_cfg.get(key), "text": text_cfg.get(key)}
        for key in fields if code_cfg.get(key) != text_cfg.get(key)
    }
    code_state, text_state = code_model.state_dict(), text_model.state_dict()
    missing = sorted(set(code_state) - set(text_state))
    unexpected = sorted(set(text_state) - set(code_state))
    shape_mismatches = {
        key: {"code": list(code_state[key].shape), "text": list(text_state[key].shape)}
        for key in set(code_state) & set(text_state)
        if code_state[key].shape != text_state[key].shape
    }
    changed = 0
    changed_parameters = 0
    squared_delta = 0.0
    for key in set(code_state) & set(text_state):
        left, right = code_state[key].detach().cpu(), text_state[key].detach().cpu()
        if left.shape != right.shape:
            continue
        if not torch.equal(left, right):
            changed += 1
            changed_parameters += right.numel()
            squared_delta += float(torch.sum((right.float() - left.float()) ** 2).item())
    compatible = not (differences or missing or unexpected or shape_mismatches)
    return {
        "ok": compatible,
        "config_differences": differences,
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
        "shape_mismatches": shape_mismatches,
        "tensors_changed_during_text_training": changed,
        "parameters_in_changed_tensors": changed_parameters,
        "weight_delta_l2": round(math.sqrt(squared_delta), 8),
        "note": "A compatibilidade exige a mesma arquitetura e formas; pesos alterados demonstram continuidade, não qualidade.",
    }


@torch.inference_mode()
def evaluate_probe(model: ApoloLanguageModel, tokenizer: BPETokenizer, text: str) -> dict[str, Any]:
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    if len(ids) < 2:
        raise ValueError("Probe produziu menos de dois tokens.")
    limit = model.config.block_size
    ids = ids[: limit + 1]
    if len(ids) < 2:
        raise ValueError("block_size insuficiente para avaliação.")
    x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0)
    _, loss = model(x, y)
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError("Loss inválida no probe.")
    value = float(loss.item())
    return {
        "loss": round(value, 6),
        "perplexity": round(math.exp(min(value, 30.0)), 6),
        "tokens": int(y.numel()),
    }


@torch.inference_mode()
def generate_text(model: ApoloLanguageModel, tokenizer: BPETokenizer, prompt: str,
                  max_new_tokens: int, use_cache: bool) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_bos=True)
    ids = ids[-model.config.block_size:]
    if not ids:
        raise ValueError("Prompt não gerou tokens.")
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    generated = model.generate(
        input_ids, max_new_tokens=max_new_tokens, temperature=0,
        top_k=None, top_p=None, use_cache=use_cache,
    )
    token_ids = [int(item) for item in generated[0].tolist()]
    if any(item < 0 or item >= model.config.vocab_size for item in token_ids):
        raise RuntimeError("Geração produziu ID fora do vocabulário.")
    decoded = tokenizer.decode(token_ids)
    if not decoded.strip():
        raise RuntimeError("Geração não produziu texto decodificável.")
    return {"token_ids": token_ids, "text": decoded, "tokens_generated": len(token_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-text", required=True)
    parser.add_argument("--checkpoint-code", required=True)
    parser.add_argument("--tokenizer-original", default="tokenizers/tokenizer")
    parser.add_argument("--output", default="reports/text-checkpoint-validation.json")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    text_dir, code_dir = Path(args.checkpoint_text), Path(args.checkpoint_code)
    for directory in (text_dir, code_dir):
        for filename in ("model.pt", "config.json"):
            path = directory / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Arquivo obrigatório ausente/vazio: {path}")

    text_model = ApoloLanguageModel.load(text_dir, map_location="cpu")
    code_model = ApoloLanguageModel.load(code_dir, map_location="cpu")
    if text_model.config.vocab_size != code_model.config.vocab_size:
        raise RuntimeError("Vocabularies diferentes entre checkpoint de código e texto.")

    embedded_tokenizer = text_dir / "tokenizer"
    tokenizer_dir = embedded_tokenizer if (embedded_tokenizer / "tokenizer.json").is_file() else Path(args.tokenizer_original)
    tokenizer = BPETokenizer.load(tokenizer_dir)
    original = BPETokenizer.load(args.tokenizer_original)
    if tokenizer.vocab_size != text_model.config.vocab_size:
        raise RuntimeError("Tokenizer e checkpoint textual possuem vocabulários incompatíveis.")
    tokenizer_hash = sha256_file(tokenizer_dir / "tokenizer.json")
    original_hash = sha256_file(Path(args.tokenizer_original) / "tokenizer.json")
    if tokenizer_hash != original_hash:
        raise RuntimeError("O tokenizer incorporado no checkpoint difere do tokenizer original do repositório.")

    compatibility = config_compatibility(code_model, text_model)
    if not compatibility["ok"]:
        raise RuntimeError("Incompatibilidade estrutural entre os checkpoints: " + json.dumps(compatibility, ensure_ascii=False))
    if compatibility["tensors_changed_during_text_training"] == 0:
        raise RuntimeError("Nenhum tensor mudou; não há evidência de continuação do treinamento textual.")

    text_model.eval()
    code_model.eval()
    probes = {
        "pt-BR": "A inteligência artificial aprende padrões da linguagem e utiliza o contexto para produzir respostas.",
        "en": "Artificial intelligence learns language patterns and uses context to produce coherent responses.",
    }
    metrics = {language: evaluate_probe(text_model, tokenizer, prompt) for language, prompt in probes.items()}

    generations = {}
    for language, prompt in {
        "pt-BR": "A inteligência artificial",
        "en": "Artificial intelligence",
    }.items():
        first = generate_text(text_model, tokenizer, prompt, args.max_new_tokens, True)
        # Reinicializa a instância a partir do disco, como em um processo novo.
        restarted = ApoloLanguageModel.load(text_dir, map_location="cpu")
        second = generate_text(restarted, tokenizer, prompt, args.max_new_tokens, True)
        if first["token_ids"] != second["token_ids"]:
            raise RuntimeError(f"Geração após reinicialização não determinística ({language}).")
        without_cache = generate_text(restarted, tokenizer, prompt, args.max_new_tokens, False)
        generations[language] = {
            "prompt": prompt,
            "with_kv_cache": first,
            "after_reload": second,
            "without_kv_cache": without_cache,
            "reload_matches": first["token_ids"] == second["token_ids"],
            "cache_matches": first["token_ids"] == without_cache["token_ids"],
        }
        if not generations[language]["cache_matches"]:
            raise RuntimeError(f"KV cache e geração sem cache divergiram ({language}).")

    report = {
        "status": "success",
        "text_checkpoint": str(text_dir),
        "code_checkpoint": str(code_dir),
        "sha256": {
            "text_model_pt": sha256_file(text_dir / "model.pt"),
            "code_model_pt": sha256_file(code_dir / "model.pt"),
            "tokenizer": tokenizer_hash,
            "original_tokenizer": original_hash,
        },
        "parameters": text_model.num_parameters(),
        "vocab_size": tokenizer.vocab_size,
        "compatibility": compatibility,
        "bilingual_probe_metrics": metrics,
        "generation": generations,
        "limitations": [
            "Loss/perplexidade são medidas em probes fixos curtos, não em um conjunto holdout amplo.",
            "Geração determinística confirma carregamento e execução; não comprova qualidade semântica.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
