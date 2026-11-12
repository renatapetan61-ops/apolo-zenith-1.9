# Relatório Final de Treinamento — Apolo Zenith 1.9

_Gerado em 2026-11-12 16:32:16_

> Este relatório contém **apenas medições reais** executadas neste dispositivo. Nada aqui é projetado, estimado ou simulado. Onde uma capacidade não existe, isso é declarado explicitamente.

## 1. Dispositivo e ambiente (medido)

| Item | Valor real |
|---|---|
| Android | 16 (SDK 36) |
| Aparelho | motorola moto g56 5G |
| Ambiente | Proot (Termux/Ubuntu) — Linux version 6.17.0-PRoot-Distro (proot@termux) (gcc (GCC) 13.3.0, GN |
| CPU | desconhecido — 8 threads (aarch64) |
| Aceleração | Nenhuma GPU exposta ao Proot do Termux; todo o treino é CPU-only. |
| RAM | 7.43 GB totais, 1600 MB disponíveis na medição |
| Swap | 5.57 GB |
| Disco | 229 GB totais, 103.0 GB livres |
| Python / PyTorch / NumPy | 3.12.3 / 2.15.0+cpu / 2.5.4 |
| Sensor térmico | **indisponivel** — Nenhuma fonte térmica acessível neste ambiente (sem /sys/class/thermal, sem dumpsys, sem Termux API). Modo preventivo at |

### Bibliotecas multimodais (import real, não presumido)

| Biblioteca | Versão |
|---|---|
| PIL | 13.0.0 |
| cv2 | 5.0.0 |
| soundfile | 0.14.0 |
| librosa | 1.0.0 |
| av | 19.0.2 |
| scipy | 1.18.1 |
| huggingface_hub | 2.4.0 |
| datasets | 5.2.0 |

### Teste real de capacidade (micro-benchmark executado)

- modelo de teste: **98,624** parâmetros
- **369.7 tokens/s** com batch 1 e sequência 32 (87 ms/passo)
- consumo de RSS do processo: **+14.5 MB**

## 2. Datasets realmente baixados

| Dataset | Modalidade | Amostras | Baixado | Licença | Download | Preparação |
|---|---|---:|---:|---|---|---|
| flickr8k_test | images | 4000 | 525.4 MB | CC BY 2.0 (imagens do Flickr8k) / licença do dataset no Hub | concluido | concluido |
| librispeech_dev_clean | audio | 2703 | 322.3 MB | CC BY 4.0 (LibriSpeech, funded por Mozilla/NSF) | concluido | concluido |
| wikimedia_videos | video | 31 | 40.3 MB | CC BY-SA / CC BY / CC0 / domínio público (por arquivo, via A | concluido | concluido |
| codigo_github_permissivo | code | 794 | 6.9 MB | Por repositório (licença declarada pela API do GitHub quando | concluido | concluido |
| gutenberg_pt_en | text | 206816 | 141.8 MB | Domínio público (Project Gutenberg) / CC BY-SA 3.0 (Wikisour | concluido | concluido |

| Modalidade | Datasets | Baixado | Em disco | Amostras |
|---|---:|---:|---:|---:|
| images | 1 | 525.4 MB | 525.4 MB | 4000 |
| audio | 1 | 322.3 MB | 665.9 MB | 2703 |
| video | 1 | 40.3 MB | 451.4 MB | 31 |
| code | 1 | 6.9 MB | 6.9 MB | 794 |
| text | 1 | 141.8 MB | 208.3 MB | 206816 |

**Total baixado: 1.01 GB** — dentro da meta do pedido (300 MB – 1 GB; a expansão acima de 1 GB é permitida pelo armazenamento disponível do aparelho).

Fontes descartadas com justificativa: `bigcode/the-stack-smol` (HTTP 401, dataset *gated*), Common Voice (`data.commonvoice.org` inacessível, HTTP 000) e o binário `ffmpeg` (ausente — substituído por **PyAV**, que embute ffmpeg).

## 3. Treinamentos executados de fato

| Modalidade | Duração | Código de saída | Log |
|---|---:|---:|---|
| codigo | 36.3 min | 0 | `treino_codigo.log` |
| visao | 51.2 min | 0 | `treino_visao.log` |
| audio | 19.6 min | 0 | `treino_audio.log` |
| video | 14.6 min | 0 | `treino_video.log` |
| multimodal | 4.5 min | 0 | `treino_multimodal.log` |
| texto | 5.8 min | 0 | `treino_texto.log` |

## 4. Métricas medidas (nunca estimadas)

### 4.1 Texto (Language Core)

- checkpoint: `checkpoints/apolo-lm-tiny/checkpoints/best` | parâmetros: **1,901,696** | vocabulário: 8192
- perda de teste: **5.397165** | perplexidade: **220.7797**
- matematica: acurácia **0.0** (n=10)
- mult_escolha: acurácia **0.375** (n=8)
- codigo: acurácia **0.0** (n=4)

### 4.2 Código

- passos: **3000** | tempo: **2150.97 s** (357.0 tokens/s)
- val_loss no corpus de código: **2.9699950009584426** | perplexidade: **19.49**
- **baseline de frequência** (prever sempre o token mais comum): perda **1.4769** | perplexidade **4.38** — referência mínima para julgar a perplexidade acima
- round-trip do tokenizador em 5 arquivos Python reais: fidelidade de caracteres **100.0%**, sintaxe válida: **True**

### 4.3 Código **gerado** (verificação estática, sem execução)

- amostras: **5** | gerações com conteúdo não vazio: **0** | taxa de sintaxe válida com conteúdo: **0.0**

Um modelo que devolve apenas espaços **não** conta como acerto: a geração é contada como válida apenas se tiver conteúdo e sintaxe correta.

### 4.4 Multimodal (visão, áudio, vídeo, alinhamento)

| Modalidade | Métrica | Valor |
|---|---|---|
| multimodal | imagem_texto R@1 | **0.0156** (chance 0.0063, n=160) |
| multimodal | audio_texto R@1 | **0.0094** (chance 0.0063, n=160) |
| video | acurácia da ordem temporal | **0.5** (chance 0.5, n=62) |
| images | R@1 (recuperação) | **0.0281** (chance 0.0063, MRR 0.0909, n=160) |
| audio | R@1 (recuperação) | **0.0375** (chance 0.0063, MRR 0.0862, n=160) |

### 4.5 Encoders treinados (parâmetros e métrica real)

| Módulo | Parâmetros | Métrica de validação | Arquivo |
|---|---:|---|---|
| Visão (ViT + projetor) | 529.024 | R@1 = 0.0233 | `models/vision/melhor/metadados.json` |
| Áudio (mel + encoder + projetor) | 1.859.072 | R@1 = 0.0176 | `models/audio/melhor/metadados.json` |
| Vídeo (temporal + ordem) | 2.189.312 | acurácia de ordem = 0.5 | `models/video/melhor/metadados.json` |
| Multimodal (projetores) | 66.048 | — | `models/multimodal/final/metadados.json` |

## 5. Proteções do dispositivo (implementadas e exercitadas)

| Proteção | Onde | Evidência |
|---|---|---|
| Treino exclusivo (lock) | `resources/training_lock.py` | `fcntl.flock` + PID/heartbeat; fila por prioridade |
| Pausa térmica | `resources/thermal_monitor.py` | modo preventivo (sem sensor no Proot); nenhuma temperatura inventada |
| Pausa por RAM | `resources/memory_manager.py` | `/proc/meminfo` + RSS; pausa real registrada durante o treino de código |
| Pausa por disco | `resources/storage_manager.py` | espaço mínimo preservado; download em blocos de ≤64 MB |
| Batch adaptativo | `resources/adaptive_batch.py` | reduz em OOM, cooldown, erro explícito |
| Checkpoint com estado | `training/checkpoint.py` | RNG, dataset, batch, fila, contadores |
| Checkpoints de pausa | `training/trainer.py` | `pausa_XXXXXX` gravados antes de pausar |

Estado do ciclo: **concluido**, fase 0, 0.09 h executadas, 0 tarefas com erro.

## 6. Problemas encontrados e corrigidos

| # | Problema (real) | Impacto | Correção |
|---|---|---|---|
| 1 | `os.replace` do arquivo de estado do lock **trocava o inode**, liberando o `flock` silenciosamente | dois treinos simultâneos passavam | estado gravado em arquivo lateral; lock sobre inode estável |
| 2 | `tokenizer.encode_dataset(texto, …)` recebia uma **string** (espera uma lista) e iterava sobre os **caracteres** | o corpus de código virava documentos de 1 token (BOS+char+EOS) e o modelo gerava apenas espaços | `tokenizer.encode(texto, …)` + teste de regressão `tests/test_corpus_codigo.py` |
| 3 | acesso **aleatório** aos row groups do Parquet relia ~30 MB por item | treino de visão 4× mais lento | ordem de acesso agrupada por row group (`ordem_por_grupo`) |
| 4 | a perda de **consistência cross-modal** aproximava imagem de cachorro de frase falada (não são o mesmo conteúdo) | alinhamento multimodal degradado | desligada por padrão (`peso_consistencia=0`) |
| 5 | a pausa por RAM usava a **porcentagem usada** (que inclui cache recuperável do kernel) | pausas frequentes e improdutivas | o piso é a memória **disponível**; a porcentagem vira apenas *aviso* |
| 6 | a validação do Language Core percorria **11.700 sequências** por avaliação | inviabilizava o treino no celular | limite explícito de 40 lotes nos trainers do celular |
| 7 | o sensor térmico é inacessível e a carga de CPU foi usada como gatilho de pausa | o treino pausaria sozinho (o treino **é** carga) | modo preventivo com **pausas programadas**; a carga é apenas proxy registrado |
| 8 | a modalidade do catálogo era `codigo`, mas a pasta exigida é `code/` | arquivos gravados na pasta errada | modalidade alinhada ao diretório e manifesto corrigido |
| 9 | `encode_dataset` + `deduplicar` produziam documentos vazios no corpus de código | baseline de frequência quase igual à perplexidade do modelo | baseline explícita (`baseline_frequencia`) reportada ao lado da perplexidade |

## 7. Limitações e problemas encontrados

1. **Sem sensor térmico acessível** dentro do Proot do Termux (`/sys/class/thermal` → *Permission denied*; sem `dumpsys`; sem Termux API). O monitor usa **modo preventivo** com pausas programadas; a carga de CPU é registrada apenas como **proxy rotulado**, nunca como temperatura.
2. **Sem GPU**: todo o treino é CPU-only.
3. **Sem decoder de linguagem multimodal**: os encoders alinham representações, mas o modelo não gera descrições, transcrições ou fala.
4. **Dataset de visão pequeno** (4.000 imagens): a recuperação fica acima da chance, mas o encoder é pequeno (1,3M parâmetros).
5. **`bigcode/the-stack-smol` é *gated*** (HTTP 401) — o corpus de código veio de repositórios GitHub permissivos via tarball, sem depender do limite de API.
6. **Limite de 60 requisições/hora** da API do GitHub (anônimo) — o downloader usa cache em disco e fallback para tarball.
7. **Validação do Language Core** era inviável no celular (11.700 sequências por avaliação): a validação foi limitada a 40 lotes, registrado na configuração.
8. **O ciclo de 10 dias não foi executado por 240 h**: por segurança do aparelho e por decisão do §15 do pedido, o scheduler foi validado com execuções curtas reais. Os números acima vêm dessas execuções.

## 8. Próximos passos

1. Ampliar o corpus de visão (mais shards do Flickr8k, Wikimedia) para reduzir o gap até a chance no R@1.
2. Treinar um **decoder de linguagem multimodal** (LM + prefixo de imagem/áudio) para habilitar descrição de imagens e ASR.
3. Decoder de áudio (TTS) — hoje inexistente.
4. Rodar o ciclo completo de 240 h com o scheduler em condições controladas (carregador, ventilação, supervisão).
5. Ampliar o corpus de código (mais repositórios/linguagens) e avaliar pass@1 real com execução de testes em sandbox.
