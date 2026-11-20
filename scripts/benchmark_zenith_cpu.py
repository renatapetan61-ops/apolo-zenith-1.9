"""Benchmark real de 1.000 passos do treinador textual em CPU.

Executa o mesmo Trainer/PyTorch e o dataset processado, mede tempo de parede,
validações efetivamente gravadas e falha se qualquer avaliação esperada faltar.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dados", default="datasets_storage/processed")
    parser.add_argument("--tokenizer", default="datasets_storage/processed/tokenizer")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--output", default="reports/cpu-1000-step-benchmark.json")
    args = parser.parse_args()
    if args.steps != 1000:
        raise SystemExit("Este benchmark de aceitação exige exatamente 1.000 passos.")
    data = Path(args.dados)
    for name in ("train.bin", "val.bin"):
        if not (data / name).is_file() or (data / name).stat().st_size == 0:
            raise SystemExit(f"Dataset obrigatório ausente/vazio: {data / name}")

    with tempfile.TemporaryDirectory(prefix="zenith-cpu-benchmark-") as tmp:
        out = Path(tmp) / "run"
        env = os.environ.copy()
        env.update({
            "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1",
        })
        cmd = [
            sys.executable, "-m", "training.text_trainer",
            "--dados", str(data), "--tokenizer", args.tokenizer,
            "--saida", str(out), "--steps", "1000",
            "--batch-size", "4", "--block-size", "128",
            "--lr", "0.00005", "--min-lr", "0.000005",
            "--warmup-steps", "200", "--early-stopping-patience", "0",
            "--eval-interval", str(args.eval_interval),
            "--save-interval", str(args.save_interval), "--sobrescrever",
        ]
        started = time.monotonic()
        proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
        elapsed = time.monotonic() - started
        if proc.returncode:
            raise RuntimeError(
                "Benchmark CPU falhou.\nSTDOUT:\n" + proc.stdout[-12000:]
                + "\nSTDERR:\n" + proc.stderr[-12000:]
            )
        summary_path = out / "resumo_treino.json"
        if not summary_path.is_file():
            raise RuntimeError("Treinador não produziu resumo_treino.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics_path = out / "metricas.jsonl"
        records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        evaluations = [row for row in records if row.get("fase") == "validacao" and isinstance(row.get("val_loss"), (int, float))]
        expected = args.steps // args.eval_interval
        if int(summary.get("passos", -1)) != 1000:
            raise RuntimeError(f"Benchmark terminou em {summary.get('passos')} passos, esperado 1000")
        if len(evaluations) < expected:
            raise RuntimeError(f"Apenas {len(evaluations)} avaliações registradas; esperadas >= {expected}")
        if not isinstance(summary.get("melhor_perda_validacao"), (int, float)) or not (0 < summary["melhor_perda_validacao"] < float("inf")):
            raise RuntimeError("Métrica de validação ausente/inválida")
        report = {
            "status": "passed", "device": "cpu", "steps": 1000,
            "elapsed_seconds": round(elapsed, 3),
            "seconds_per_step": round(elapsed / 1000, 6),
            "evaluation_count": len(evaluations),
            "evaluation_interval": args.eval_interval,
            "best_validation_loss": summary["melhor_perda_validacao"],
            "tokens_processed": summary.get("tokens_processados"),
            "tokens_per_second": summary.get("tokens_por_segundo"),
            "model_parameters": summary.get("parametros"),
            "session_timeout_budget_minutes": 300,
            "source_command": cmd,
        }
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
