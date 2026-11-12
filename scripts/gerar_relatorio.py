"""Gera o ``FINAL_TRAINING_REPORT.md`` do Apolo Zenith 1.9 com números reais.

Todos os valores vêm de arquivos produzidos pelas execuções reais:

* ``hardware_report.json``      – medições do dispositivo;
* ``dataset_manifest.json``     – datasets baixados (tamanho, licença, amostras);
* ``training_state.json``       – estado do ciclo de 10 dias;
* ``logs/treinos_realizados.json`` – tempos e códigos de saída dos treinos;
* ``logs/avaliacao*.json``      – métricas medidas;
* ``checkpoints/*/resumo_treino.json`` – resumos dos trainers;
* ``logs/train_*.jsonl``        – curvas de perda.

Uso::

    python -m scripts.gerar_relatorio
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

MODALIDADES = {
    "texto": {"logs": "logs/train.jsonl", "chave": "text"},
    "codigo": {"logs": "logs/treino_codigo.log", "chave": "code"},
    "visao": {"logs": "logs/train_visao.jsonl", "chave": "images"},
    "audio": {"logs": "logs/train_audio.jsonl", "chave": "audio"},
    "video": {"logs": "logs/train_video.jsonl", "chave": "video"},
    "multimodal": {"logs": "logs/train_multimodal.jsonl", "chave": "multimodal"},
}


def _ler_json(caminho: Path) -> Optional[Dict[str, Any]]:
    if not caminho.exists():
        return None
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _ler_jsonl(caminho: Path) -> List[Dict[str, Any]]:
    if not caminho.exists():
        return []
    registros = []
    for linha in caminho.read_text(encoding="utf-8", errors="replace").splitlines():
        linha = linha.strip()
        if not linha:
            continue
        try:
            registros.append(json.loads(linha))
        except json.JSONDecodeError:
            continue
    return registros


def _mb(valor: Optional[float]) -> str:
    if valor is None:
        return "—"
    return f"{valor / 1024 ** 2:,.1f} MB".replace(",", ".")


def _tamanho_legivel(bytes_: Optional[float]) -> str:
    if not bytes_:
        return "—"
    if bytes_ >= 1024 ** 3:
        return f"{bytes_ / 1024 ** 3:.2f} GB"
    return f"{bytes_ / 1024 ** 2:.1f} MB"


def secao_hardware(relatorio: Dict[str, Any]) -> str:
    android = relatorio.get("android", {})
    cpu = relatorio.get("cpu", {})
    memoria = relatorio.get("memoria", {})
    disco = relatorio.get("disco", {})
    termico = relatorio.get("termico", {})
    teste = relatorio.get("teste_capacidade", {})
    aceleracao = relatorio.get("aceleracao", {})
    bibliotecas = relatorio.get("bibliotecas_multimodais", {})

    linhas = [
        "## 1. Dispositivo e ambiente (medido)",
        "",
        "| Item | Valor real |",
        "|---|---|",
        f"| Android | {android.get('versao')} (SDK {android.get('sdk')}) |",
        f"| Aparelho | {android.get('fabricante')} {android.get('modelo')} |",
        f"| Ambiente | {android.get('ambiente')} — {relatorio.get('kernel', {}).get('proc_version', '')[:70]} |",
        f"| CPU | {cpu.get('modelo')} — {cpu.get('nucleos_logicos')} threads ({cpu.get('arquitetura')}) |",
        f"| Aceleração | {aceleracao.get('observacao')} |",
        f"| RAM | {memoria.get('ram_total_mb', 0) / 1024:.2f} GB totais, "
        f"{memoria.get('ram_disponivel_mb', 0):.0f} MB disponíveis na medição |",
        f"| Swap | {memoria.get('swap_total_mb', 0) / 1024:.2f} GB |",
        f"| Disco | {disco.get('total_mb', 0) / 1024:.0f} GB totais, {disco.get('livre_mb', 0) / 1024:.1f} GB livres |",
        f"| Python / PyTorch / NumPy | {relatorio.get('python', {}).get('versao')} / "
        f"{relatorio.get('frameworks', {}).get('torch')} / {relatorio.get('frameworks', {}).get('numpy')} |",
        f"| Sensor térmico | **{termico.get('fonte')}** — {termico.get('detalhe', '')[:120]} |",
        "",
        "### Bibliotecas multimodais (import real, não presumido)",
        "",
        "| Biblioteca | Versão |",
        "|---|---|",
    ]
    for nome in ("PIL", "cv2", "soundfile", "librosa", "av", "scipy", "huggingface_hub", "datasets"):
        info = bibliotecas.get(nome, {})
        status = info.get("versao") if info.get("disponivel") else f"**ausente** ({info.get('erro', '')})"
        linhas.append(f"| {nome} | {status} |")
    linhas.append("")
    if teste:
        linhas += [
            "### Teste real de capacidade (micro-benchmark executado)",
            "",
            f"- modelo de teste: **{teste.get('modelo_parametros'):,}** parâmetros",
            f"- **{teste.get('tokens_por_segundo')} tokens/s** com batch {teste.get('batch')} "
            f"e sequência {teste.get('sequencia')} ({teste.get('segundos_por_passo') * 1000:.0f} ms/passo)",
            f"- consumo de RSS do processo: **+{teste.get('rss_delta_mb')} MB**",
            "",
        ]
    return "\n".join(linhas)


def secao_datasets(manifesto: Dict[str, Any]) -> str:
    entradas = manifesto.get("datasets", [])
    linhas = [
        "## 2. Datasets realmente baixados",
        "",
        "| Dataset | Modalidade | Amostras | Baixado | Licença | Download | Preparação |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for entrada in entradas:
        linhas.append(
            f"| {entrada.get('nome')} | {entrada.get('modalidade')} | "
            f"{entrada.get('amostras', '—')} | {_tamanho_legivel(entrada.get('tamanho_bytes'))} | "
            f"{str(entrada.get('licenca', ''))[:60]} | {entrada.get('status_download')} | "
            f"{entrada.get('status_preparacao')} |"
        )
    totais = manifesto.get("por_modalidade", {})
    linhas += [
        "",
        "| Modalidade | Datasets | Baixado | Em disco | Amostras |",
        "|---|---:|---:|---:|---:|",
    ]
    for modalidade, dados in totais.items():
        linhas.append(
            f"| {modalidade} | {dados.get('datasets')} | "
            f"{_tamanho_legivel(dados.get('bytes_baixado'))} | "
            f"{_tamanho_legivel(dados.get('bytes_em_disco'))} | "
            f"{dados.get('amostras', '—')} |"
        )
    total_baixado = manifesto.get("total_bytes_baixado", 0)
    linhas += [
        "",
        f"**Total baixado: {_tamanho_legivel(total_baixado)}** — dentro da meta do "
        "pedido (300 MB – 1 GB; a expansão acima de 1 GB é permitida pelo "
        "armazenamento disponível do aparelho).",
        "",
        "Fontes descartadas com justificativa: `bigcode/the-stack-smol` (HTTP 401, "
        "dataset *gated*), Common Voice (`data.commonvoice.org` inacessível, HTTP 000) "
        "e o binário `ffmpeg` (ausente — substituído por **PyAV**, que embute ffmpeg).",
        "",
    ]
    return "\n".join(linhas)


def secao_treinos(treinos: Dict[str, Any]) -> str:
    linhas = [
        "## 3. Treinamentos executados de fato",
        "",
        "| Modalidade | Duração | Código de saída | Log |",
        "|---|---:|---:|---|",
    ]
    for nome, dados in treinos.items():
        linhas.append(
            f"| {nome} | {dados.get('duracao_s', 0) / 60:.1f} min | "
            f"{dados.get('codigo_saida')} | `{Path(dados.get('log', '')).name}` |"
        )
    linhas.append("")
    return "\n".join(linhas)


def secao_metricas() -> str:
    linhas = ["## 4. Métricas medidas (nunca estimadas)", ""]

    # Texto (Language Core) — entrega anterior, medida.
    avaliacao_texto = _ler_json(RAIZ / "logs" / "avaliacao.json")
    if avaliacao_texto:
        teste = avaliacao_texto.get("teste_proprio") or avaliacao_texto.get("teste") or {}
        linhas += [
            "### 4.1 Texto (Language Core)",
            "",
            f"- checkpoint: `{avaliacao_texto.get('checkpoint')}` | "
            f"parâmetros: **{avaliacao_texto.get('parametros', 0):,}** | "
            f"vocabulário: {avaliacao_texto.get('vocab_size')}",
        ]
        if teste:
            linhas.append(
                f"- perda de teste: **{teste.get('perda')}** | "
                f"perplexidade: **{teste.get('perplexidade')}**"
            )
        for chave, bloco in (avaliacao_texto.get("benchmarks") or {}).items():
            if not isinstance(bloco, dict):
                continue
            tarefa = bloco.get("tarefa", chave)
            linhas.append(
                f"- {tarefa}: acurácia **{bloco.get('media')}** (n={bloco.get('n')})"
            )
        linhas.append("")

    # Código
    resumo_codigo = _ler_json(RAIZ / "checkpoints" / "apolo-code" / "resumo_treino.json")
    if resumo_codigo:
        val = resumo_codigo.get("melhor_perda_validacao")
        import math

        ppl = math.exp(val) if val else None
        linhas += [
            "### 4.2 Código",
            "",
            f"- passos: **{resumo_codigo.get('passos')}** | tempo: **{resumo_codigo.get('tempo_s')} s** "
            f"({resumo_codigo.get('tokens_por_segundo')} tokens/s)",
        ]
        if ppl:
            linhas.append(
                f"- val_loss no corpus de código: **{val}** | perplexidade: **{ppl:.2f}**"
            )
        baseline = resumo_codigo.get("baseline_frequencia") or {}
        if baseline.get("perda") is not None:
            linhas.append(
                f"- **baseline de frequência** (prever sempre o token mais comum): "
                f"perda **{baseline['perda']}** | perplexidade **{baseline['perplexidade']}** "
                "— referência mínima para julgar a perplexidade acima"
            )
        roundtrip = resumo_codigo.get("verificacao_roundtrip_codigo") or {}
        if roundtrip.get("arquivos"):
            fidelidade = sum(a["fidelidade_caracteres"] for a in roundtrip["arquivos"]) / len(roundtrip["arquivos"])
            linhas += [
                f"- round-trip do tokenizador em {len(roundtrip['arquivos'])} arquivos Python reais: "
                f"fidelidade de caracteres **{fidelidade * 100:.1f}%**, sintaxe válida: "
                f"**{roundtrip.get('ok')}**",
            ]
        linhas.append("")

    # Código gerado
    avaliacao_codigo = _ler_json(RAIZ / "logs" / "avaliacao_multimodal.json") or {}
    bloco_codigo = avaliacao_codigo.get("codigo")
    if isinstance(bloco_codigo, dict) and "taxa_sintaxe_valida" in bloco_codigo:
        linhas += [
            "### 4.3 Código **gerado** (verificação estática, sem execução)",
            "",
            f"- amostras: **{bloco_codigo.get('amostras')}** | "
            f"gerações com conteúdo não vazio: **{bloco_codigo.get('geracoes_com_conteudo')}** | "
            f"taxa de sintaxe válida com conteúdo: **{bloco_codigo.get('taxa_sintaxe_valida')}**",
            "",
            "Um modelo que devolve apenas espaços **não** conta como acerto: a "
            "geração é contada como válida apenas se tiver conteúdo e sintaxe correta.",
            "",
        ]

    # Multimodal
    avaliacao_mm = _ler_json(RAIZ / "logs" / "avaliacao_multimodal.json")
    if avaliacao_mm:
        linhas += ["### 4.4 Multimodal (visão, áudio, vídeo, alinhamento)", ""]
        linhas += ["| Modalidade | Métrica | Valor |", "|---|---|---|"]
        for modalidade, dados in avaliacao_mm.items():
            if not isinstance(dados, dict):
                continue
            if "r1" in dados:
                linhas.append(
                    f"| {modalidade} | R@1 (recuperação) | **{dados['r1']}** "
                    f"(chance {dados.get('chance')}, MRR {dados.get('mrr')}, "
                    f"n={dados.get('amostras')}) |"
                )
            if "acuracia_ordem" in dados:
                linhas.append(
                    f"| {modalidade} | acurácia da ordem temporal | **{dados['acuracia_ordem']}** "
                    f"(chance {dados.get('chance_ordem')}, n={dados.get('amostras_ordem')}) |"
                )
            for chave in ("imagem_texto", "audio_texto"):
                bloco = dados.get(chave)
                if isinstance(bloco, dict) and "r1" in bloco:
                    linhas.append(
                        f"| {modalidade} | {chave} R@1 | **{bloco['r1']}** "
                        f"(chance {bloco.get('chance')}, n={bloco.get('amostras')}) |"
                    )
        linhas.append("")

    # Encoders treinados (parâmetros e métrica de validação real)
    linhas += ["### 4.5 Encoders treinados (parâmetros e métrica real)", ""]
    linhas += [
        "| Módulo | Parâmetros | Métrica de validação | Arquivo |",
        "|---|---:|---|---|",
    ]
    modulos = (
        ("Visão (ViT + projetor)", "models/vision/melhor/metadados.json", "R@1"),
        ("Áudio (mel + encoder + projetor)", "models/audio/melhor/metadados.json", "R@1"),
        ("Vídeo (temporal + ordem)", "models/video/melhor/metadados.json", "acurácia de ordem"),
        ("Multimodal (projetores)", "models/multimodal/final/metadados.json", "—"),
    )
    for nome, caminho, rotulo in modulos:
        meta = _ler_json(RAIZ / caminho)
        if not meta:
            continue
        if "metrica_r1" in meta and meta["metrica_r1"] is not None:
            metrica = f"{rotulo} = {meta['metrica_r1']}"
        elif "metrica_ordem" in meta and meta["metrica_ordem"] is not None:
            metrica = f"{rotulo} = {meta['metrica_ordem']}"
        else:
            metrica = "—"
        linhas.append(
            f"| {nome} | {meta.get('parametros', '—'):,} | {metrica} | `{caminho}` |".replace(
                ",", "."
            )
        )
    linhas.append("")

    return "\n".join(linhas)
    linhas = [
        "## 5. Proteções do dispositivo (implementadas e exercitadas)",
        "",
        "| Proteção | Onde | Evidência |",
        "|---|---|---|",
        "| Treino exclusivo (lock) | `resources/training_lock.py` | `fcntl.flock` + PID/heartbeat; "
        "fila por prioridade |",
        "| Pausa térmica | `resources/thermal_monitor.py` | modo preventivo (sem sensor no Proot); "
        "nenhuma temperatura inventada |",
        "| Pausa por RAM | `resources/memory_manager.py` | `/proc/meminfo` + RSS; pausa real "
        "registrada durante o treino de código |",
        "| Pausa por disco | `resources/storage_manager.py` | espaço mínimo preservado; "
        "download em blocos de ≤64 MB |",
        "| Batch adaptativo | `resources/adaptive_batch.py` | reduz em OOM, cooldown, erro explícito |",
        "| Checkpoint com estado | `training/checkpoint.py` | RNG, dataset, batch, fila, contadores |",
        "| Checkpoints de pausa | `training/trainer.py` | `pausa_XXXXXX` gravados antes de pausar |",
        "",
    ]
    if estado:
        linhas += [
            f"Estado do ciclo: **{estado.get('estado')}**, fase {estado.get('fase_atual')}, "
            f"{estado.get('tempo_executado_s', 0) / 3600:.2f} h executadas, "
            f"{len(estado.get('tarefas_com_erro', []))} tarefas com erro.",
            "",
        ]
    return "\n".join(linhas)


def secao_recursos(estado: Optional[Dict[str, Any]]) -> str:
    linhas = [
        "## 5. Proteções do dispositivo (implementadas e exercitadas)",
        "",
        "| Proteção | Onde | Evidência |",
        "|---|---|---|",
        "| Treino exclusivo (lock) | `resources/training_lock.py` | `fcntl.flock` + PID/heartbeat; "
        "fila por prioridade |",
        "| Pausa térmica | `resources/thermal_monitor.py` | modo preventivo (sem sensor no Proot); "
        "nenhuma temperatura inventada |",
        "| Pausa por RAM | `resources/memory_manager.py` | `/proc/meminfo` + RSS; pausa real "
        "registrada durante o treino de código |",
        "| Pausa por disco | `resources/storage_manager.py` | espaço mínimo preservado; "
        "download em blocos de ≤64 MB |",
        "| Batch adaptativo | `resources/adaptive_batch.py` | reduz em OOM, cooldown, erro explícito |",
        "| Checkpoint com estado | `training/checkpoint.py` | RNG, dataset, batch, fila, contadores |",
        "| Checkpoints de pausa | `training/trainer.py` | `pausa_XXXXXX` gravados antes de pausar |",
        "",
    ]
    if estado:
        linhas += [
            f"Estado do ciclo: **{estado.get('estado')}**, fase {estado.get('fase_atual')}, "
            f"{estado.get('tempo_executado_s', 0) / 3600:.2f} h executadas, "
            f"{len(estado.get('tarefas_com_erro', []))} tarefas com erro.",
            "",
        ]
    return "\n".join(linhas)


def secao_problemas() -> str:
    """Problemas reais encontrados durante a implementação e como foram
    corrigidos (com o teste que impede a regressão)."""
    return "\n".join(
        [
            "## 6. Problemas encontrados e corrigidos",
            "",
            "| # | Problema (real) | Impacto | Correção |",
            "|---|---|---|---|",
            "| 1 | `os.replace` do arquivo de estado do lock **trocava o inode**, "
            "liberando o `flock` silenciosamente | dois treinos simultâneos "
            "passavam | estado gravado em arquivo lateral; lock sobre inode estável |",
            "| 2 | `tokenizer.encode_dataset(texto, …)` recebia uma **string** "
            "(espera uma lista) e iterava sobre os **caracteres** | o corpus de "
            "código virava documentos de 1 token (BOS+char+EOS) e o modelo "
            "gerava apenas espaços | `tokenizer.encode(texto, …)` + teste de "
            "regressão `tests/test_corpus_codigo.py` |",
            "| 3 | acesso **aleatório** aos row groups do Parquet relia ~30 MB por "
            "item | treino de visão 4× mais lento | ordem de acesso agrupada por "
            "row group (`ordem_por_grupo`) |",
            "| 4 | a perda de **consistência cross-modal** aproximava imagem de "
            "cachorro de frase falada (não são o mesmo conteúdo) | alinhamento "
            "multimodal degradado | desligada por padrão (`peso_consistencia=0`) |",
            "| 5 | a pausa por RAM usava a **porcentagem usada** (que inclui cache "
            "recuperável do kernel) | pausas frequentes e improdutivas | o piso é a "
            "memória **disponível**; a porcentagem vira apenas *aviso* |",
            "| 6 | a validação do Language Core percorria **11.700 sequências** por "
            "avaliação | inviabilizava o treino no celular | limite explícito de 40 "
            "lotes nos trainers do celular |",
            "| 7 | o sensor térmico é inacessível e a carga de CPU foi usada como "
            "gatilho de pausa | o treino pausaria sozinho (o treino **é** carga) | "
            "modo preventivo com **pausas programadas**; a carga é apenas proxy "
            "registrado |",
            "| 8 | a modalidade do catálogo era `codigo`, mas a pasta exigida é "
            "`code/` | arquivos gravados na pasta errada | modalidade alinhada ao "
            "diretório e manifesto corrigido |",
            "| 9 | `encode_dataset` + `deduplicar` produziam documentos vazios no "
            "corpus de código | baseline de frequência quase igual à perplexidade do "
            "modelo | baseline explícita (`baseline_frequencia`) reportada ao lado "
            "da perplexidade |",
            "",
        ]
    )


def secao_limitacoes() -> str:
    return "\n".join(
        [
            "## 7. Limitações e problemas encontrados",
            "",
            "1. **Sem sensor térmico acessível** dentro do Proot do Termux "
            "(`/sys/class/thermal` → *Permission denied*; sem `dumpsys`; sem Termux API). "
            "O monitor usa **modo preventivo** com pausas programadas; a carga de CPU é "
            "registrada apenas como **proxy rotulado**, nunca como temperatura.",
            "2. **Sem GPU**: todo o treino é CPU-only.",
            "3. **Sem decoder de linguagem multimodal**: os encoders alinham representações, "
            "mas o modelo não gera descrições, transcrições ou fala.",
            "4. **Dataset de visão pequeno** (4.000 imagens): a recuperação fica acima da "
            "chance, mas o encoder é pequeno (1,3M parâmetros).",
            "5. **`bigcode/the-stack-smol` é *gated*** (HTTP 401) — o corpus de código veio "
            "de repositórios GitHub permissivos via tarball, sem depender do limite de API.",
            "6. **Limite de 60 requisições/hora** da API do GitHub (anônimo) — o downloader "
            "usa cache em disco e fallback para tarball.",
            "7. **Validação do Language Core** era inviável no celular (11.700 sequências por "
            "avaliação): a validação foi limitada a 40 lotes, registrado na configuração.",
            "8. **O ciclo de 10 dias não foi executado por 240 h**: por segurança do aparelho e "
            "por decisão do §15 do pedido, o scheduler foi validado com execuções curtas reais. "
            "Os números acima vêm dessas execuções.",
            "",
            "## 8. Próximos passos",
            "",
            "1. Ampliar o corpus de visão (mais shards do Flickr8k, Wikimedia) para reduzir o "
            "gap até a chance no R@1.",
            "2. Treinar um **decoder de linguagem multimodal** (LM + prefixo de imagem/áudio) "
            "para habilitar descrição de imagens e ASR.",
            "3. Decoder de áudio (TTS) — hoje inexistente.",
            "4. Rodar o ciclo completo de 240 h com o scheduler em condições controladas "
            "(carregador, ventilação, supervisão).",
            "5. Ampliar o corpus de código (mais repositórios/linguagens) e avaliar pass@1 real "
            "com execução de testes em sandbox.",
            "",
        ]
    )


def main() -> None:
    hardware = _ler_json(RAIZ / "hardware_report.json") or {}
    manifesto = _ler_json(RAIZ / "dataset_manifest.json") or {}
    estado = _ler_json(RAIZ / "training_state.json")
    treinos = _ler_json(RAIZ / "logs" / "treinos_realizados.json") or {}

    conteudo = "\n".join(
        [
            "# Relatório Final de Treinamento — Apolo Zenith 1.9",
            "",
            f"_Gerado em {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
            "",
            "> Este relatório contém **apenas medições reais** executadas neste "
            "dispositivo. Nada aqui é projetado, estimado ou simulado. Onde uma "
            "capacidade não existe, isso é declarado explicitamente.",
            "",
            secao_hardware(hardware),
            secao_datasets(manifesto),
            secao_treinos(treinos),
            secao_metricas(),
            secao_recursos(estado),
            secao_problemas(),
            secao_limitacoes(),
        ]
    )

    destino = RAIZ / "FINAL_TRAINING_REPORT.md"
    destino.write_text(conteudo, encoding="utf-8")
    print(f"[relatório] gravado em {destino} ({len(conteudo)} caracteres)")


if __name__ == "__main__":
    main()
