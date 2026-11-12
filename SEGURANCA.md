# Regras de Segurança Permanentes — Apolo Zenith 1.9

Estas 10 regras valem para **qualquer** execução de treino, dataset ou
inferência neste dispositivo. Elas são implementadas no código (não são
apenas recomendações) e verificadas pelos testes em `tests/test_resources.py`
e `tests/test_multimodal.py`.

| # | Regra | Onde é implementada | Como é verificada |
|---|---|---|---|
| 1 | **Um treinamento por vez** — nunca dois processos de treino simultâneos | `resources/training_lock.py` (`fcntl.flock` + PID/heartbeat), `server/job_manager.py` | `test_lock_bloqueia_treino_simultaneo`, `test_lock_detecta_dono_morto` |
| 2 | **Streaming de dados** — o dataset nunca é carregado inteiro na RAM | `resources/lazy_dataset.py`, `datasets/{image,audio,video,code}_loader.py` (memmap, row groups, leitura sob demanda) | `test_lazy_loading_nao_carrega_tudo`, `test_streaming_jsonl_ignora_linhas_corrompidas` |
| 3 | **Lazy loading com cache limitado por bytes** e verificação SHA-256 (arquivo corrompido é ignorado com log, nunca derruba o treino) | `resources/lazy_dataset.py` | `test_lazy_ignora_shard_corrompido`, `test_integridade_sha256_detecta_alteracao` |
| 4 | **Batch pequeno (1–4)** com **batch adaptativo**: um OOM reduz o lote e a repetição é limitada (sem retry infinito) | `resources/adaptive_batch.py`, `training/trainer.py` | `test_batch_reduzido_apos_oom_simulado`, `test_oom_sem_retentativa_infinita_levanta_erro` |
| 5 | **Checkpoints frequentes** com estado completo (RNG, dataset, batch, fila) e gravação atômica validada; o checkpoint anterior nunca é perdido | `training/checkpoint.py`, `training/trainer.py` | `test_checkpoint_gravacao_atomica_e_validada`, `test_checkpoint_preserva_o_melhor` |
| 6 | **Pausa térmica** — sem sensor acessível, o modo preventivo aplica pausas programadas; **nenhuma temperatura é inventada** | `resources/thermal_monitor.py`, `resources/resource_monitor.py` | `test_pausa_termica_quando_temperatura_excede_limite`, `test_modo_preventivo_sem_sensor_nao_inventa_temperatura` |
| 7 | **Controle de RAM** — RSS, `/proc/meminfo` e cgroup; o treino pausa antes de o sistema sufocar o aparelho | `resources/memory_manager.py` | `test_monitor_de_recursos_entra_em_pausa_termica`, `test_operacoes_nao_exigem_root` |
| 8 | **Controle de armazenamento** — espaço mínimo preservado, downloads em blocos (≤64 MB/passo), rotação que preserva `best`/`final` e **nunca** apaga dados originais | `resources/storage_manager.py`, `datasets/downloader.py` | `test_falta_de_armazenamento_pausa_o_treino`, `test_nao_apaga_arquivos_originais_nem_checkpoints` |
| 9 | **Retomada após interrupção** — o progresso é persistido (dataset, batch, fila, RNG) e o treino continua de onde parou | `resources/incremental_processor.py`, `training/checkpoint.py`, `orchestrator/scheduler.py` | `test_processamento_incremental_retoma`, `test_retomada_restaura_passo_e_estado`, `test_scheduler_persiste_estado` |
| 10 | **Nunca exceder os limites do aparelho** — CPU-only assumido, sem CUDA presumido, sem root, sem loop infinito; o scheduler executa um plano finito | `resources/`, `orchestrator/scheduler.py` | `test_execucao_e_cpu_only`, `test_scheduler_nao_executa_loop_infinito`, `test_operacoes_nao_exigem_root` |

## Regras adicionais (complementares)

11. **Integridade do dataset** — download em blocos com SHA-256 calculado
    durante o download; arquivo com hash divergente é descartado e registrado
    em `logs/errors.log`.
12. **Nenhum dado original é alterado** — `datasets/*/raw/` é somente leitura
    para o pipeline; os processados vão para `datasets/*/processed/`.
13. **Nenhuma capacidade é declarada sem medição** — os módulos reportam
    `implementado` em `core/status.py`; o que não existe (decoder multimodal,
    geração de voz, geração de imagem/vídeo) continua levantado erro
    explícito.
14. **Erros são auditáveis** — `logs/errors.log` (JSONL), `logs/resource_events.jsonl`
    (transições de estado) e `logs/dataset_downloads.log`.

## Limites padrão (`configs/resource_manager.yaml`)

```yaml
termicos:      # sem sensor no Proot → modo preventivo
  temperatura_maxima_c: 42.0
  pausa_preventiva_min: 5.0
  ciclo_trabalho_min: 30.0
memoria:
  ram_max_utilizacao_percent: 80.0   # aviso; a pausa usa a memória *disponível*
  ram_minima_disponivel_mb: 512      # piso crítico
  rss_max_mb: 2048
armazenamento:
  espaco_minimo_livre_mb: 2048
  download_max_por_passo_mb: 64
batch:
  batch_minimo: 1
  batch_maximo: 4
  seq_len_maxima: 64
```

## Como inspecionar o estado em tempo real

```bash
python main.py recursos      # limites, estado do monitor e do lock
tail -f logs/resource_events.jsonl
tail -f logs/treino_*.log
```
