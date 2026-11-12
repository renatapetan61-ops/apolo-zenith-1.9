"""Ponto de entrada do Apolo Zenith 1.9.

CLI unificada para as operações do ecossistema::

    python main.py hardware --json
    python main.py hardware-report          # gera hardware_report.json
    python main.py datasets --listar        # catálogo de datasets
    python main.py datasets download --datasets flickr8k_test
    python main.py datasets prepare --todas
    python main.py train --config configs/training.yaml
    python main.py train-visao --passos 200
    python main.py train-audio --passos 200
    python main.py train-video --passos 150
    python main.py train-codigo --steps 400
    python main.py train-multimodal --passos 200
    python main.py evaluate --checkpoint checkpoints/apolo-lm-tiny/checkpoints/best
    python main.py evaluate-multimodal --todas
    python main.py scheduler status
    python main.py scheduler run --todas
    python main.py recursos
    python main.py serve --porta 8080
    python main.py testes
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))


def _executar_modulo(modulo: str, argumentos: list[str]) -> int:
    """Executa a ``main()`` de um módulo repassando os argumentos."""
    import importlib

    mod = importlib.import_module(modulo)
    sys.argv = [modulo] + argumentos
    try:
        mod.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="apolo",
        description="Apolo Zenith 1.9 — IA multimodal do projeto Apolo",
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    # ------------------------------------------------------------ hardware
    p_hw = sub.add_parser("hardware", help="detecta o hardware disponível (celular/servidor)")
    p_hw.add_argument("--json", action="store_true", help="saída em JSON")
    p_hw.add_argument("--testar-rede", action="store_true", help="mede a vazão de download")
    p_hw.add_argument("--parametros", type=int, default=2_000_000)

    sub.add_parser(
        "hardware-report",
        help="gera o hardware_report.json com medições reais + teste de capacidade",
    )

    # ------------------------------------------------------------- recursos
    sub.add_parser("recursos", help="mostra os limites de recursos e o estado atual")

    # -------------------------------------------------------------- dados
    p_ds = sub.add_parser("datasets", help="catálogo, download e preparação dos datasets")
    ds_sub = p_ds.add_subparsers(dest="acao", required=True)
    ds_sub.add_parser("listar", help="lista o catálogo de datasets")
    p_dl = ds_sub.add_parser("download", help="baixa datasets com verificação de SHA-256")
    p_dl.add_argument("--datasets", nargs="*", default=[])
    p_dl.add_argument("--modalidades", nargs="*", default=[])
    p_pr = ds_sub.add_parser("prepare", help="prepara os datasets processados")
    p_pr.add_argument("--modalidade", default=None)
    p_pr.add_argument("--todas", action="store_true")

    # ------------------------------------------------------------- treino
    p_train = sub.add_parser("train", help="treina o Apolo Language Core (texto)")
    p_train.add_argument("--config", default="configs/training.yaml")
    p_train.add_argument("--dados", default="datasets_storage/processed")
    p_train.add_argument("--saida", default=None)
    p_train.add_argument("--retomar", default=None)
    p_train.add_argument("--max-steps", type=int, default=None)

    p_tv = sub.add_parser("train-visao", help="treina o encoder visual (ViT + projetor)")
    p_tv.add_argument("--passos", type=int, default=1000)
    p_tv.add_argument("--batch-size", type=int, default=16)
    p_tv.add_argument("--saida", default="models/vision")

    p_ta = sub.add_parser("train-audio", help="treina o encoder de áudio (mel + projetor)")
    p_ta.add_argument("--passos", type=int, default=250)
    p_ta.add_argument("--batch-size", type=int, default=4)
    p_ta.add_argument("--saida", default="models/audio")

    p_tvd = sub.add_parser("train-video", help="treina o encoder temporal de vídeo")
    p_tvd.add_argument("--passos", type=int, default=150)
    p_tvd.add_argument("--batch-size", type=int, default=2)
    p_tvd.add_argument("--saida", default="models/video")

    p_tc = sub.add_parser("train-codigo", help="treina o LM sobre o corpus de código")
    p_tc.add_argument("--steps", type=int, default=400)
    p_tc.add_argument("--dados", default="datasets/code/processed")
    p_tc.add_argument("--saida", default="checkpoints/apolo-code")

    p_tm = sub.add_parser("train-multimodal", help="treina o alinhamento multimodal")
    p_tm.add_argument("--passos", type=int, default=200)
    p_tm.add_argument("--batch-size", type=int, default=2)
    p_tm.add_argument("--saida", default="models/multimodal")

    # --------------------------------------------------------- avaliação
    p_inf = sub.add_parser("infer", help="gera texto a partir de um checkpoint")
    p_inf.add_argument("--checkpoint", required=True)
    p_inf.add_argument("--prompt", default="")
    p_inf.add_argument("--interativo", action="store_true")
    p_inf.add_argument("--max-tokens", type=int, default=128)
    p_inf.add_argument("--temperatura", type=float, default=0.8)
    p_inf.add_argument("--seed", type=int, default=42)
    p_inf.add_argument("--top-k", type=int, default=50)
    p_inf.add_argument("--top-p", type=float, default=0.95)

    p_eval = sub.add_parser("evaluate", help="avalia o modelo de texto com métricas reais")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--dados", default="datasets_storage/processed")
    p_eval.add_argument("--saida", default="logs/avaliacao.json")
    p_eval.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="arquivo.jsonl:tarefa (repetível). Tarefas: perda, mult_escolha, matematica, codigo",
    )
    p_eval.add_argument("--max-exemplos", type=int, default=50)
    p_eval.add_argument("--max-tokens", type=int, default=256)

    p_em = sub.add_parser("evaluate-multimodal", help="avalia as modalidades multimodais")
    p_em.add_argument("--modalidade", default=None,
                      choices=["images", "audio", "video", "multimodal"])
    p_em.add_argument("--todas", action="store_true")
    p_em.add_argument("--max-itens", type=int, default=800)

    # --------------------------------------------------------- scheduler
    p_sch = sub.add_parser("scheduler", help="scheduler do ciclo de 10 dias (240 h)")
    sch_sub = p_sch.add_subparsers(dest="acao", required=True)
    sch_sub.add_parser("status", help="estado atual do ciclo")
    sch_sub.add_parser("plan", help="plano de 10 dias")
    p_run = sch_sub.add_parser("run", help="executa uma fase ou todas")
    p_run.add_argument("--fase", type=int, default=None)
    p_run.add_argument("--todas", action="store_true")

    # ------------------------------------------------------------ servidor
    p_serve = sub.add_parser("serve", help="sobe a API do servidor de treinamento")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--porta", type=int, default=8080)

    sub.add_parser("testes", help="executa a suíte de testes automatizados")

    p_est = sub.add_parser("estimar", help="estima custo de memória/tempo do experimento")
    p_est.add_argument("--config", default="configs/training.yaml")
    p_est.add_argument("--metricas", default="checkpoints/apolo-lm-tiny/metricas.jsonl")
    p_est.add_argument("--max-steps", type=int, default=None)

    args = parser.parse_args()

    # ------------------------------------------------------------ hardware
    if args.comando == "hardware":
        from server.hardware_detection import main as hardware_main

        sys.argv = ["hardware"] + sys.argv[2:]
        hardware_main()
        return 0

    if args.comando == "hardware-report":
        return _executar_modulo("scripts.hardware_report", sys.argv[2:])

    if args.comando == "recursos":
        from resources.config import carregar_config_recursos
        from resources.resource_monitor import ResourceMonitor
        from resources.training_lock import TrainingLock

        cfg = carregar_config_recursos(RAIZ / "configs" / "resource_manager.yaml")
        monitor = ResourceMonitor(cfg, raiz=RAIZ)
        lock = TrainingLock(RAIZ / cfg.diretorio_lock)
        print(json.dumps(
            {
                "limites": cfg.para_dict(),
                "estado": monitor.resumo(),
                "lock": lock.status(),
            },
            indent=2,
            ensure_ascii=False,
        ))
        return 0

    # -------------------------------------------------------------- dados
    if args.comando == "datasets":
        if args.acao == "listar":
            return _executar_modulo("scripts.download_datasets", ["--listar"])
        if args.acao == "download":
            extras = []
            if args.datasets:
                extras += ["--datasets", *args.datasets]
            if args.modalidades:
                extras += ["--modalidades", *args.modalidades]
            return _executar_modulo("scripts.download_datasets", extras)
        if args.acao == "prepare":
            extras = ["--todas"] if args.todas else ["--modalidade", args.modalidade]
            return _executar_modulo("scripts.prepare_multimodal", extras)

    # ------------------------------------------------------------- treino
    if args.comando == "train":
        extra = ["--config", args.config, "--dados", args.dados]
        if args.saida:
            extra += ["--saida", args.saida]
        if args.retomar:
            extra += ["--retomar", args.retomar]
        if args.max_steps is not None:
            extra += ["--max-steps", str(args.max_steps)]
        return _executar_modulo("scripts.train", extra)

    if args.comando == "train-visao":
        return _executar_modulo(
            "training.image_trainer",
            ["--passos", str(args.passos), "--batch-size", str(args.batch_size),
             "--saida", args.saida],
        )

    if args.comando == "train-audio":
        return _executar_modulo(
            "training.audio_trainer",
            ["--passos", str(args.passos), "--batch-size", str(args.batch_size),
             "--saida", args.saida],
        )

    if args.comando == "train-video":
        return _executar_modulo(
            "training.video_trainer",
            ["--passos", str(args.passos), "--batch-size", str(args.batch_size),
             "--saida", args.saida],
        )

    if args.comando == "train-codigo":
        return _executar_modulo(
            "training.code_trainer",
            ["--steps", str(args.steps), "--dados", args.dados, "--saida", args.saida],
        )

    if args.comando == "train-multimodal":
        return _executar_modulo(
            "training.multimodal_trainer",
            ["--passos", str(args.passos), "--batch-size", str(args.batch_size),
             "--saida", args.saida],
        )

    # --------------------------------------------------------- inferência
    if args.comando == "infer":
        extra = [
            "--checkpoint", args.checkpoint,
            "--max-tokens", str(args.max_tokens),
            "--temperatura", str(args.temperatura),
            "--seed", str(args.seed),
            "--top-k", str(args.top_k),
            "--top-p", str(args.top_p),
        ]
        if args.interativo:
            extra.append("--interativo")
        else:
            extra += ["--prompt", args.prompt]
        return _executar_modulo("scripts.infer", extra)

    # --------------------------------------------------------- avaliação
    if args.comando == "evaluate":
        extra = [
            "--checkpoint", args.checkpoint,
            "--dados", args.dados,
            "--saida", args.saida,
            "--max-exemplos", str(args.max_exemplos),
            "--max-tokens", str(args.max_tokens),
        ]
        for benchmark in args.benchmark:
            extra += ["--benchmark", benchmark]
        return _executar_modulo("scripts.evaluate", extra)

    if args.comando == "evaluate-multimodal":
        extra = ["--max-itens", str(args.max_itens)]
        if args.todas:
            extra.append("--todas")
        elif args.modalidade:
            extra += ["--modalidade", args.modalidade]
        return _executar_modulo("scripts.evaluate_multimodal", extra)

    # --------------------------------------------------------- scheduler
    if args.comando == "scheduler":
        extras = [args.acao]
        if args.acao == "run":
            if args.fase is not None:
                extras += ["--fase", str(args.fase)]
            if args.todas:
                extras.append("--todas")
        return _executar_modulo("orchestrator.scheduler", extras)

    # ------------------------------------------------------------ servidor
    if args.comando == "serve":
        sys.argv = ["api", "--host", args.host, "--porta", str(args.porta)]
        from server.api import main as api_main

        api_main()
        return 0

    if args.comando == "testes":
        resultado = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(RAIZ / "tests")], cwd=str(RAIZ)
        )
        return resultado.returncode

    if args.comando == "estimar":
        from core.config import ModelConfig
        from scripts.train import aplicar, carregar_config
        from training.estimativas import estimar_treino
        from training.trainer import TrainingConfig

        bruto = carregar_config(args.config)
        modelo_config = ModelConfig.from_dict(bruto.get("model", {}))
        treino_config = aplicar(TrainingConfig(), dict(bruto.get("training", {})))
        if args.max_steps is not None:
            treino_config.max_steps = args.max_steps

        vazoes = []
        caminho = Path(args.metricas)
        if caminho.exists():
            for linha in caminho.read_text(encoding="utf-8").splitlines():
                try:
                    registro = json.loads(linha)
                except json.JSONDecodeError:
                    continue
                if registro.get("fase") == "treino" and registro.get("tokens_s"):
                    vazoes.append(float(registro["tokens_s"]))
        vazao = sum(vazoes[-10:]) / len(vazoes[-10:]) if vazoes else None
        estimativa = estimar_treino(modelo_config, treino_config, tokens_por_segundo=vazao)
        print(json.dumps(estimativa.para_dict(), indent=2, ensure_ascii=False))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
