# Apolo Zenith 1.9

IA multimodal própria em desenvolvimento — **Language Core + pipeline
multimodal real (5 modalidades) + gerenciamento de recursos do dispositivo**.

> **Declaração de estado:** este repositório contém um **Language Core real**
> (Transformer decoder-only treinado de fato), um tokenizer BPE próprio,
> pipeline de dados com controle de licenças, **encoders multimodais
> efetivamente treinados** (visão ViT + projetor, áudio mel-spectrogram +
> projetor, vídeo temporal + projetor), **gerenciamento de recursos do
> celular** (lock de exclusividade, guardas térmico/RAM/disco, batch
> adaptativo, checkpoints com estado completo), servidor de treinamento,
> cliente mobile e harness de avaliação.
>
> **O que ainda NÃO existe:** decoder de linguagem multimodal (o modelo
> **alinha representações**, mas não gera descrições de imagens nem
> transcrições), geração de voz, geração de imagens/vídeo e um
> Understanding Transformer que componha as modalidades dentro do Language
> Core. Nenhuma dessas capacidades é declarada como funcional.
> O sensor térmico do aparelho **não é acessível** dentro do Proot do Termux:
> o monitor trabalha em **modo preventivo** e nunca inventa temperatura.

---

## 1. Índice

- [2. Status real do ecossistema](#2-status-real-do-ecossistema)
- [3. Arquitetura neural implementada](#3-arquitetura-neural-implementada)
- [4. Hardware detectado e implicações](#4-hardware-detectado-e-implicações)
- [5. Instalação](#5-instalação)
- [6. Pipeline de dados](#6-pipeline-de-dados)
- [7. Treinamento (Fase 1)](#7-treinamento-fase-1)
- [8. Inferência](#8-inferência)
- [9. Avaliação](#9-avaliação)
- [10. Servidor externo e cliente mobile](#10-servidor-externo-e-cliente-mobile)
- [11. Otimização para mobile](#11-otimização-para-mobile)
- [12. Testes](#12-testes)
- [13. Estrutura do projeto](#13-estrutura-do-projeto)
- [14. Roadmap das fases 2–10](#14-roadmap-das-fases-210)
- [15. Gerenciamento de recursos do dispositivo](#15-gerenciamento-de-recursos-do-dispositivo)
- [16. Pipeline multimodal (5 modalidades)](#16-pipeline-multimodal-5-modalidades)
- [17. Scheduler do ciclo de 10 dias](#17-scheduler-do-ciclo-de-10-dias)
- [18. Limitações conhecidas](#18-limitações-conhecidas)

---

## 2. Status real do ecossistema

| Fase | Módulo | Status | Evidência |
|------|--------|--------|-----------|
| 1 | **Apolo Language Core** | ✅ Implementado e treinado | val_loss 8,94 (aleatório) → **5,23**; checkpoint + testes de aprendizado |
| 1 | **Tokenizer BPE byte-level** | ✅ Implementado | 8.192 tokens, round-trip exato em PT/EN/Unicode |
| 1 | **Pipeline de dados** | ✅ Implementado | 5.515 docs coletados, 5.170 aprovados, 12,76M tokens de treino |
| 1 | **Servidor + API + mobile** | ✅ Implementado | endpoints testados, jobs monitorados |
| 1 | **Harness de avaliação** | ✅ Implementado | perplexidade, múltipla escolha, matemática, pass@k com execução de testes |
| 3 | **Coding Intelligence** | ✅ Treinado | corpus real de 13 linguagens, val_loss 2,97 (ppl 19,49) — **acima** da baseline de frequência (ppl 4,38); código gerado ainda sem conteúdo |
| 4 | **Apolo Vision** | ✅ Encoder treinado | ViT (529k params) + projetor, perda contrastiva InfoNCE real, **R@1 2,81%** vs 0,63% de chance |
| 5 | **Apolo Audio** | ✅ Encoder treinado | mel-spectrogram + CNN/Transformer (1,86M params) + projetor, **R@1 3,75%** vs 0,63% |
| 6 | **Apolo Video** | ✅ Encoder treinado | encoder temporal (2,19M params) + cabeça de ordem temporal, R@1 12,9% (chance 3,2%); ordem temporal **na chance** |
| 9 | **Multimodal Alignment** | ✅ Projetores treinados | alinhamento conjunto visão/áudio → texto, R@1 1,56% / 0,94% vs 0,63% |
| 10 | **Gerenciamento de recursos** | ✅ Implementado | lock exclusivo, guardas térmico/RAM/disco, batch adaptativo, checkpoint com estado completo |
| 2 | Language Intelligence | ⏳ Planejado | escala de modelo, dados de instrução/raciocínio |
| 7 | Image Generator | ⏳ **Não implementado** | latent diffusion |
| 8 | Video Generator | ⏳ **Não implementado** | video diffusion |
| 10 | Decoder de linguagem multimodal | ⏳ **Não implementado** | o modelo alinha representações, mas não gera texto a partir de imagem/áudio |

**Não se declara superioridade a nenhum modelo.** Comparações exigem condições
equivalentes de prompt, decoding e conjunto de testes, e devem ser publicadas
com os artefatos de avaliação.

---

## 3. Arquitetura neural implementada

`core/` implementa um **Transformer decoder-only real**, treinável:

```
tokens ──► TokenEmbedding (8192 × d_model)
             │
             ├─► Bloco Transformer × N
             │      ├─ RMSNorm (pré-norm)
             │      ├─ Multi-Head Self-Attention causal
             │      │     ├─ RoPE (Rotary Positional Embeddings)
             │      │     ├─ suporte a GQA (n_kv_head < n_head)
             │      │     └─ KV Cache pré-alocado
             │      ├─ Residual connection
             │      ├─ RMSNorm (pré-norm)
             │      ├─ MLP SwiGLU  (down(silu(gate(x)) * up(x)))
             │      └─ Residual connection
             │
             └─► RMSNorm final ──► LM Head (pesos tied) ──► logits
```

Detalhes de implementação relevantes:

| Recurso | Onde | Observação |
|---|---|---|
| Multi-Head Self-Attention | `core/attention.py` | `scaled_dot_product_attention` com máscara causal |
| RoPE | `core/embeddings.py` | cache de sen/cos, crescimento automático |
| RMSNorm | `core/normalization.py` | acumulação em fp32 mesmo sob autocast |
| SwiGLU | `core/transformer.py` | dimensão interna `8/3·d_model` arredondada |
| KV Cache | `core/kv_cache.py` | inferência O(T) por token |
| AdamW decoupled | `training/adamw.py` | implementação própria, **verificada numericamente contra `torch.optim.AdamW`** |
| Warmup + cosine | `training/scheduler.py` | com `min_lr` |
| Grad clipping / accumulation | `training/trainer.py` | grad norm global |
| Mixed precision | `training/trainer.py` | bf16/fp16 quando o hardware suporta |
| Checkpointing atômico | `training/checkpoint.py` | escrita em `.tmp` + rename, rotação, retenção do melhor |
| DDP / FSDP | `training/distributed.py` | detecção de hardware antes de habilitar |

### Tokenizer (`tokenizer/`)

- **BPE byte-level**: 256 tokens base garantem cobertura de **qualquer** texto
  (português, inglês, cirílico, CJK, emoji) sem pré-tokenizador por idioma.
- Treinamento **incremental** com heap e contagens de par atualizadas apenas
  para os pares afetados — viável para vocabulários de 8k–32k.
- **16 tokens especiais reservados desde o início**, incluindo os marcadores
  de modalidade `<|vision|>`, `<|audio|>`, `<|video|>` e `<|tool_call|>`.
  Eles já existem no vocabulário, o que evita retrabalho quando os módulos
  multimodais entrarem.

---

## 4. Hardware detectado e implicações

Medição real (`python main.py hardware`):

| Recurso | Valor detectado | Implicação |
|---|---|---|
| SO | Ubuntu 24.04 (Proot/Termux), ARM64 | sem binários CUDA |
| CPU | 8 threads | treino em CPU viável para modelos pequenos |
| RAM | 7,4 GB total (~1,4–2,5 GB livres) | datasets via `memmap`; sem activation checkpointing pesado |
| GPU | **nenhuma** | DDP/FSDP não aplicáveis aqui |
| Disco | 229 GB (109 GB livres) | suficiente para corpus local de dezenas de GB |
| PyTorch | 2.15.0+cpu | CPU-only |

**Sobre o servidor externo (100–400 TB):** o storage **não** implica RAM nem
GPU. O `server/hardware_detection.py` mede CPU, RAM, VRAM, disco e rede antes
de recomendar estratégia (`single`, `ddp`, `fsdp`, `cpu`), e o
`training/distributed.py` só habilita DDP/FSDP quando o hardware confirma
suporte. Quando o servidor real for provisionado, rode:

```bash
python main.py hardware --json --testar-rede
```

---

## 5. Instalação

```bash
# CPU (celular / Termux / servidor mínimo)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# CUDA (servidor de treino)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## 6. Pipeline de dados

### 6.1 Coleta com controle de licenças

```bash
python -m datasets.corpus_builder --saida datasets_storage \
    --max-livros-pt 120 --max-livros-en 8 --max-wiki-mb 24
```

| Fonte | Conteúdo | Licença |
|---|---|---|
| Project Gutenberg (PT) | 120 obras em português | Domínio público |
| Project Gutenberg (EN) | 8 obras em inglês | Domínio público |
| Wikisource (PT) | 5.385 páginas | CC BY-SA 4.0 |

Cada documento é registrado em `datasets_storage/manifest.json` com **URL,
licença, SHA-256 e tamanho** — a composição do corpus é auditável.

### 6.2 Limpeza, qualidade, PII e deduplicação

```bash
python -m scripts.prepare_data --vocab 8192
```

Etapas reais executadas nesta entrega:

- **5.515** documentos lidos → **5.170 aprovados**, **345 rejeitados**
  (194 curtos demais, 138 com poucas palavras, 7 com baixa proporção
  alfabética, 6 com repetição excessiva);
- **4 duplicatas** removidas (dedup exato por hash + aproximado por **simhash**);
- **PII mascarada**: e-mails, telefones, CPF e URLs;
- Unicode normalizado em NFC, cabeçalhos do Gutenberg e marcação do Wikisource
  removidos;
- split **97% / 1,5% / 1,5%** (treino/validação/teste) com semente fixa.

Resultado: **12.759.787 tokens** de treino, **106.316** de validação,
**150.936** de teste.

### 6.3 Tokenização

O tokenizer é treinado **apenas no split de treino** (sem vazamento) e salvo em
`datasets_storage/processed/tokenizer/`. Os splits são gravados como binário
`uint16` (memmap) em `train.bin`, `val.bin`, `test.bin`.

---

## 7. Treinamento (Fase 1)

```bash
python main.py train --config configs/training.yaml --dados datasets_storage/processed
```

Configuração (`configs/training.yaml`): modelo de **1.901.696 parâmetros**
(4 camadas, 128 dims, 4 cabeças, contexto 256, vocab 8.192), AdamW
(lr 6e-4, wd 0,1, β=(0,9; 0,95)), warmup 80 passos + cosine, grad clipping 1,0,
batch 16 × 256 tokens = 4.096 tokens/passo, avaliação a cada 50 passos,
early stopping com paciência de 5 avaliações.

Exemplo de saída real (ver `logs/train_real.log`):

```
passo    20/700 | perda 8.6975 | lr 1.95e-04 | 1422 tok/s
passo   100/700 | perda 7.1056 | lr 5.99e-04 | 1372 tok/s
[validação] passo 100 | perda 7.0907 | ppl 1200.70
```

### 7.1 Resultado real desta entrega

O treino foi executado em **duas etapas** (a primeira foi interrompida por uma
reinicialização do ambiente — o mecanismo de retomada foi exercido de fato):

| Etapa | Passos | Tokens | val_loss | Perplexidade |
|---|---|---|---|---|
| Inicial (aleatório) | 0 | — | 8,94 | ~5.200 |
| Treino 1 (passo 1→200) | 200 | 0,82M | 6,45 | 630 |
| Treino 1 (passo 600) | 600 | 2,46M | 5,92 | 374 |
| **Treino 2 (passo 1800)** | **1800** | **7,37M** | **5,23** | **186** |

Curva de perda real (`checkpoints/apolo-lm-tiny/metricas.jsonl`):

```
passo    20/600 | perda 8.6975 | lr 1.95e-04 | 1422 tok/s
passo   100/600 | perda 7.1056 | lr 5.99e-04 | 1372 tok/s
[validação] passo 100 | perda 7.0907 | ppl 1200.70
passo   600/600 | perda 5.9002 | lr 6.00e-05 | 1214 tok/s
[validação] passo 600 | perda 5.9243 | ppl 374.00
passo  1800/1800 | perda 5.0781 | lr 6.00e-05 | 1199 tok/s
[validação] passo 1800 | perda 5.2277 | ppl 186.36
```

Custo: **1,19 mil tokens/s** em CPU (sem GPU), 4.130 s na segunda etapa.

### 7.2 Retomada de treino

```bash
python main.py train --config configs/training.yaml \
    --retomar checkpoints/apolo-lm-tiny/checkpoints/final
```

O checkpoint carrega pesos, estado do AdamW, estado do scheduler e metadados.
A gravação é atômica — uma interrupção no meio da escrita não corrompe o
checkpoint anterior.

> **Trava de segurança:** se já existir um experimento no diretório de saída,
> o trainer **recusa** iniciar (`--sobrescrever` é necessário para substituir
> de propósito). Isso evita que um job acidental apague um modelo já
> treinado.

---

## 8. Inferência

```bash
python main.py infer --checkpoint checkpoints/apolo-lm-tiny/checkpoints/best \
    --prompt "Era uma vez" --max-tokens 120 --temperatura 0.8

# Sessão interativa
python main.py infer --checkpoint checkpoints/apolo-lm-tiny/checkpoints/best --interativo
```

A geração usa KV Cache (uma passada de atenção por token), top-k/top-p,
penalidade de repetição e semente reproduzível.

---

## 9. Avaliação

```bash
python main.py evaluate --checkpoint checkpoints/apolo-lm-tiny/checkpoints/best \
    --dados datasets_storage/processed --saida logs/avaliacao.json
```

Tarefas disponíveis no harness (`evaluation/benchmarks.py`):

| Tarefa | Métrica | Observação |
|---|---|---|
| `perda` | perplexidade | entropia cruzada real sobre o texto |
| `mult_escolha` | acurácia | escolha por maior log-likelihood de cada alternativa |
| `matematica` | acurácia numérica | extração e comparação de valores |
| `codigo` | pass@1 | **executa os testes** em subprocesso isolado |

Datasets de exemplo prontos (autoral, não oficiais) em
`evaluation/datasets/`. Para benchmarks oficiais (MMLU, GSM8K, MATH, HumanEval,
MBPP, SWE-bench, MMBench, MMMU), converta para JSONL local — a conversão
automática não é feita aqui para respeitar as licenças de cada conjunto.

### 9.1 Resultados reais (checkpoint de 1.901.696 parâmetros)

Relatório completo em `logs/avaliacao.json`:

| Métrica | Valor | Observação |
|---|---|---|
| perplexidade (split de teste, 10.240 tokens) | **220,78** (perda 5,397) | conjunto de teste nunca visto no treino |
| perplexidade (validação) | **186,36** (perda 5,228) | passo 1800 |
| múltipla escolha (8 itens autorais) | **37,5%** | acima do acaso (25%) |
| matemática (10 itens autorais) | **0%** | sem capacidade aritmética |
| código — pass@1 (4 itens autorais) | **0%** | sem capacidade de programação |

Exemplo de geração (prompt → continuação, temperatura 0,6, semente 3):

```
Era uma vez um rei que morava num castelo
.
A mão de um homem, e o caminho, em que o mesmo era o que se não fosse, a casa.
E o seu ar, ao mesmo tempo, era o que não não não pode ser ser.
```

**Leitura honesta:** o modelo aprendeu a *estrutura* do português (morfologia,
funções sintáticas, conectivos, pontuação e passagem entre documentos), mas
**não** aprendeu semântica, aritmética ou programação. Isso é o resultado
esperado para 1,9M parâmetros e 7,4M tokens de treino — e é exatamente o
motivo pelo qual as Fases 2 e 3 (escala de modelo + dados de raciocínio/código)
existem no roadmap. Nenhuma superioridade é declarada.

### 9.2 Estimativas de custo

```bash
python main.py estimar --max-steps 1800
```

```
"tempo_estimado_h": 1.659,      # baseado em 1234 tokens/s MEDIDOS
"memoria": {"pesos_gb": 0.007084, "adamw_gb": 0.014169, "total_gb": 0.028337}
```

Sem medição de vazão, o estimador **não inventa** um tempo: devolve `null` e
explica o que falta.

---

## 10. Servidor externo e cliente mobile

### 10.1 API do servidor

```bash
python main.py serve --host 0.0.0.0 --porta 8080
```

| Endpoint | Método | Função |
|---|---|---|
| `/api/health` | GET | estado do serviço |
| `/api/hardware` | GET | relatório de hardware + recomendação de estratégia |
| `/api/metrics` | GET | métricas do sistema + últimas métricas de treino |
| `/api/models` | GET | checkpoints disponíveis (tamanho, passo, métricas) |
| `/api/jobs` | GET/POST | listar / iniciar jobs de treino |
| `/api/jobs/<id>` | GET | status + log do job |
| `/api/jobs/<id>/cancelar` | POST | cancelar job (SIGTERM no grupo de processos) |
| `/api/inferir` | POST | inferência de texto em um checkpoint |

### 10.2 Cliente mobile

```python
from mobile.mobile_client import ClienteServidorApolo

cliente = ClienteServidorApolo(host="servidor.local", porta=8080)
print(cliente.hardware().dados["cpu"])
cliente.iniciar_job(["python", "-m", "scripts.train", "--config", "configs/training.yaml"], "treino-1")
print(cliente.inferir("checkpoints/apolo-lm-tiny/checkpoints/best", "Era uma vez").dados)
```

O celular **não treina**: ele configura experimentos, envia comandos, monitora
métricas, gerencia checkpoints/datasets e executa inferência local de modelos
compactados.

---

## 11. Otimização para mobile

```bash
python -m scripts.quantize --checkpoint checkpoints/apolo-lm-tiny/checkpoints/best \
    --saida checkpoints/apolo-lm-tiny/mobile --quantizacao int8
```

`mobile/inference.py` implementa:

- **quantização dinâmica INT8** com detecção real de backend (quando o
  backend não suporta, o sistema **informa e mantém fp32** — nunca finge);
- tentativa de **INT4** via `torchao` (opcional);
- **limitação de threads** (`torch.set_num_threads`) para economia de energia;
- **KV Cache** para respostas rápidas;
- estimação de RAM ocupada pelo modelo carregado.

Export real desta entrega: **6,11 MB** (contra 7,27 MB em fp32). A redução é
limitada porque a tabela de embeddings (8.192 × 128 = 1,05M params) **não** é
quantizada pela quantização dinâmica — ela domina o tamanho do modelo. Em
modelos maiores, a compressão de embeddings será feita à parte.

`configs/mobile.yaml` define os limites (threads, tokens por minuto, pausa por
temperatura).

---

## 12. Testes

```bash
python -m pytest -q            # ou: python main.py testes
```

Cobertura atual — **96 testes, todos passando** (nenhum ignorado — o checkpoint
treinado está presente, então os testes de aprendizado realmente executam):

| Arquivo | O que prova |
|---|---|
| `tests/test_tokenizer.py` | round-trip exato (PT/EN/Unicode), fallback byte, tokens especiais, persistência, compressão |
| `tests/test_transformer.py` | shapes, **causalidade**, **equivalência do KV Cache**, geração com e sem cache idêntica, tying de embeddings, contagem de parâmetros, save/load |
| `tests/test_optimizer.py` | **AdamW próprio ≡ `torch.optim.AdamW`** (10 passos, atol 1e-5), decay desacoplado, scheduler warmup/cosine |
| `tests/test_training.py` | a perda cai no treino real, checkpoint salva/carrega, retomada, rotação, **avaliação do script ≡ avaliação do trainer** (regressão de off-by-one) |
| `tests/test_learning.py` | **o checkpoint treinado tem perda menor que o modelo aleatório**, prefere o texto real ao embaralhado, curva de perda registrada |
| `tests/test_datasets.py` | limpeza, PII, qualidade, dedup por simhash, split determinístico |
| `tests/test_server.py` | detecção de hardware, jobs, ciclo completo da API |
| `tests/test_mobile.py` | quantização INT8 real, inferência local, cliente HTTP contra a API |
| `tests/test_benchmarks.py` | harness de benchmarks, execução real de testes, **perda da avaliação ≡ perda do modelo** |
| `tests/test_metrics.py` | pass@k, exact match, F1, perplexidade |
| `tests/test_estimativas.py` | estimativas de memória/tempo (e recusa em estimar sem medição) |
| `tests/test_modulos_status.py` | módulos das Fases 3–8 **falham explicitamente** em vez de simular capacidade |

---

## 13. Estrutura do projeto

```
apolo-zenith-1.9/
├── core/                 # Language Core: config, atenção, RoPE, RMSNorm, SwiGLU, KV Cache
├── tokenizer/            # BPE byte-level + vocabulário especial
├── training/             # AdamW, scheduler, trainer, checkpoint, métricas, DDP/FSDP
│                          # + text/code/image/audio/video/multimodal_trainer.py
├── resources/            # config, thermal, memory, storage, resource_monitor,
│                          # training_lock, adaptive_batch, lazy_dataset, incremental
├── orchestrator/         # scheduler do ciclo de 10 dias (training_state.json)
├── datasets/             # coleta com licenças, limpeza, dedup, empacotamento
│                          # + catalog/downloader/manifest + loaders multimodais
├── server/               # detecção de hardware, API, jobs (com lock), monitoramento
├── mobile/               # inferência local (INT8/INT4), cliente do servidor
├── evaluation/           # métricas + harness de benchmarks + datasets de exemplo
├── vision/ audio/ video/ # encoders **implementados e treinados**
├── generation/ coding/   # Fases 7–8: estrutura e contrato (não implementados)
├── configs/              # training.yaml, mobile.yaml, multimodal.yaml,
│                          # resource_manager.yaml (limites do dispositivo)
├── scripts/              # prepare_data, train, infer, evaluate, quantize,
│                          # hardware_report, download_datasets, prepare_multimodal,
│                          # evaluate_multimodal, treinar_tudo, gerar_relatorio
├── tests/                # suíte automatizada
├── checkpoints/ logs/ datasets_storage/ models/ tokenizers/
├── main.py               # CLI unificada
└── requirements.txt
```

---

## 14. Roadmap das fases 2–10

`configs/multimodal.yaml` é a **fonte de verdade do status** de cada fase e
atualiza o critério de conclusão junto com a implementação. Resumo:

1. **Language Core** ✅ — Transformer treinado com perda decrescente.
2. **Language Intelligence** — escala do modelo, dados de instrução/raciocínio,
   benchmarks MMLU/GSM8K/MATH.
3. **Coding Intelligence** ✅ — corpus real de 13 linguagens com licenças
   verificadas, tokenizer com round-trip fiel e perplexidade medida no
   corpus de código (com baseline de frequência como referência).
4. **Vision Integration** ✅ — ViT + projetor treinados com InfoNCE;
   recuperação imagem↔texto medida em split held-out. **Sem decoder de
   linguagem** (não descreve imagens).
5. **Audio Integration** ✅ — mel-spectrogram + encoder + projetor treinados;
   recuperação áudio↔texto medida. **Sem decoder de voz**.
6. **Video Integration** ✅ — encoder temporal + cabeça de ordem temporal;
   acurácia de ordenação medida. **Sem sumário textual**.
7. **Image Generation** — latent diffusion com FID/CLIP reportados.
8. **Video Generation** — video diffusion com consistência temporal.
9. **Multimodal Alignment** — raciocínio conjunto sobre as modalidades.
10. **Optimization** — quantização, cache, perfil de latência/energia.

Os tokens de modalidade já existem no vocabulário, de modo que a entrada de
cada módulo não exige re-treinar o tokenizer.

---

## 15. Gerenciamento de recursos do dispositivo

O pacote `resources/` implementa as proteções exigidas para treinar em um
celular. Os limites ficam em `configs/resource_manager.yaml`.

| Mecanismo | Arquivo | O que faz |
|---|---|---|
| **Exclusividade de treino** | `resources/training_lock.py` | `fcntl.flock` + PID/heartbeat; detecta dono morto; fila por prioridade (`linguagem > codigo > visao > audio > video > multimodal`) |
| **Monitor térmico** | `resources/thermal_monitor.py` | tenta `/sys/class/thermal`, `dumpsys`, Termux API e `getprop`; **sem sensor → modo preventivo** (pausas programadas), nunca inventa temperatura |
| **Gestão de RAM** | `resources/memory_manager.py` | `/proc/meminfo`, RSS de `/proc/self/status`, cgroup (quando visível), `RLIMIT_AS` |
| **Gestão de disco** | `resources/storage_manager.py` | espaço livre, download em blocos (≤64 MB/passo), limpeza **só** de temporários, rotação de checkpoints preservando `best`/`final` |
| **Batch adaptativo** | `resources/adaptive_batch.py` | batch 1–4, reduz pela metade em OOM, cooldown e **erro explícito** (sem retry infinito) |
| **Orquestrador** | `resources/resource_monitor.py` | estados `IDLE/RUNNING/WARNING/PAUSED_*/RECOVERING/COMPLETED/FAILED`, log em `logs/resource_events.jsonl` |
| **Lazy loading** | `resources/lazy_dataset.py` | cache por **bytes**, SHA-256, arquivo corrompido ignorado com log |
| **Incremental** | `resources/incremental_processor.py` | blocos com progresso persistido e retomada |

### Integração no loop de treino

`training/trainer.py` agora:

1. verifica térmica/RAM/disco a cada 15 s;
2. **pausa com segurança**: grava checkpoint **antes** de pausar e retoma
   quando as métricas normalizam (com teto de `pausa_maxima_s`);
3. reage a OOM reduzindo o batch e recriando o loader;
4. grava checkpoints com **estado completo**: RNG (Python/NumPy/PyTorch),
   progresso do dataset, batch adaptativo, fila e contadores.

`server/job_manager.py` adquire o lock antes de lançar o processo de treino —
dois treinamentos simultâneos são **impossíveis**.

```bash
python main.py recursos      # limites + estado + situação do lock
```

---

## 16. Pipeline multimodal (5 modalidades)

Cada modalidade tem **dados reais**, um **encoder próprio**, um **objetivo de
treinamento específico** e uma **métrica medida**.

| Modalidade | Dados (reais) | Encoder | Objetivo | Métrica |
|---|---|---|---|---|
| **Texto** | Gutenberg PT/EN + Wikisource (142 MB) | Language Core (1,9M params) | Cross-entropy autoregressiva | perplexidade 220,8 (teste) |
| **Código** | 13 linguagens em repositórios permissivos (794 arquivos) | mesmo Language Core | Cross-entropy sobre código | val_loss 2,97 (ppl 19,49) |
| **Imagens** | Flickr8k (4.000 imagens + 5 legendas) | ViT (patch 16, 529k params) + projetor | InfoNCE imagem↔texto | R@1 2,81% (chance 0,63%) |
| **Áudio** | LibriSpeech dev-clean (2.703 falas) | mel-spectrogram + CNN/Transformer (1,86M params) + projetor | InfoNCE áudio↔texto | R@1 3,75% (chance 0,63%) |
| **Vídeo** | Wikimedia Commons (31 vídeos, 217 frames) | encoder temporal (2,19M params) + cabeça de ordem | ordem temporal + InfoNCE título | R@1 12,9%; ordem 50% (chance) |

### Resultados medidos (split held-out, execução real)

| Modalidade | Métrica | Valor | Chance | n |
|---|---|---:|---:|---:|
| **Imagens** | R@1 de recuperação imagem↔texto | **2,81%** | 0,63% | 160 |
| **Áudio** | R@1 de recuperação áudio↔texto | **3,75%** | 0,63% | 160 |
| **Vídeo** | R@1 de recuperação vídeo↔título | **12,9%** | 3,23% | 31 |
| **Vídeo** | acurácia da ordem temporal | **50%** | 50% | 62 |
| **Multimodal** (projetores) | R@1 imagem↔texto | **1,56%** | 0,63% | 160 |
| **Multimodal** (projetores) | R@1 áudio↔texto | **0,94%** | 0,63% | 160 |
| **Código** | perplexidade no corpus de código | **19,49** | 4,38 (baseline de frequência) | — |
| **Código gerado** | taxa de sintaxe válida **com conteúdo** | **0%** | — | 5 |

> **Leitura honesta:** a visão e o áudio ficam **acima da chance** (4–6×), mas
> ainda distantes de um alinhamento útil — o encoder tem ~0,5–1,9M parâmetros e
> o dataset de visão tem 4.000 imagens. O **vídeo não passou da chance** na
> tarefa de ordem temporal (31 vídeos não bastam). O LM de código **não
> vence a baseline de frequência** dentro do orçamento de CPU do aparelho, e
> **não gera código com conteúdo** — por isso a taxa de sintaxe válida é 0%
> (gerações só com espaços são contadas como falha, nunca como acerto).

### Estrutura

```
datasets/{text,code,images,audio,video}/raw|processed|manifests
models/{language,vision,audio,video,multimodal}
training/{text,code,image,audio,video,multimodal}_trainer.py
dataset_manifest.json   hardware_report.json   training_state.json
```

### Comandos

```bash
python main.py datasets listar                 # catálogo com licenças
python main.py datasets download --datasets flickr8k_test
python main.py datasets prepare --todas
python main.py train-visao      --passos 1000
python main.py train-audio      --passos 250
python main.py train-video      --passos 150
python main.py train-codigo     --steps 400
python main.py train-multimodal --passos 200
python main.py evaluate-multimodal --todas
```

### O que o modelo **não** faz

Os encoders produzem **representações** alinhadas ao espaço textual. Não há
decoder de linguagem multimodal, portanto o Apolo **não descreve imagens**,
**não transcreve áudio** e **não gera voz**. Qualquer afirmação em sentido
contrário seria falsa.

---

## 17. Scheduler do ciclo de 10 dias

`orchestrator/scheduler.py` executa o plano de 10 fases (meta de 240 h) com
recuperação após interrupção. O estado fica em `training_state.json`.

```bash
python main.py scheduler plan      # o plano de 10 dias
python main.py scheduler status    # estado, tempo executado, pausas, erros
python main.py scheduler run --fase 5
python main.py scheduler run --todas
```

O scheduler **não** é um loop infinito: ele executa um plano finito, mede o
tempo realmente gasto, registra pausas e erros, e para ao fim do plano. O
uso de `--todas` é **uma decisão explícita do usuário**.

---

## 18. Limitações conhecidas

Este é um **protótipo de pesquisa**, e as seguintes afirmações são honestas:

1. **O modelo de 1,9M parâmetros não é um sistema de IA de fronteira.** Ele
   existe para validar a pipeline de ponta a ponta: dados → tokenizer →
   treino → checkpoint → inferência → avaliação. O texto gerado tem estrutura
   morfosintática portuguesa, mas sem semântica.
2. **Não existe decoder de linguagem multimodal.** Os encoders de visão, áudio
   e vídeo são reais e treinados, e o alinhamento é medido, mas o modelo
   **não gera** descrições de imagens, transcrições de áudio nem fala. Gerar
   voz exigiria um decoder de áudio treinado, que não existe aqui.
3. **O dataset de visão é pequeno (4.000 imagens)** e o encoder tem 1,3M
   parâmetros: a recuperação de imagem↔texto fica **acima da chance**, mas
   não é Near-ImageNet. Medidas reais estão em `logs/avaliacao_multimodal.json`.
4. **O sensor térmico não é acessível** dentro do Proot do Termux
   (`/sys/class/thermal` → *Permission denied*; sem `dumpsys`; sem Termux API).
   O monitor entra em **modo preventivo** com pausas programadas e registra a
   carga de CPU apenas como **proxy rotulado** — nunca como temperatura.
5. **Sem GPU neste ambiente**, todo o treino é CPU-only (~210–400 tokens/s no
   Language Core). O mesmo código usa bf16/fp16 quando houver hardware.
6. **Os benchmarks locais de exemplo são autorais** (8 itens de múltipla
   escolha, 10 de matemática, 4 de código) e não substituem MMLU, GSM8K,
   HumanEval ou MMMU.
7. **Nenhuma superioridade é declarada.** O objetivo declarado é construir a
   melhor qualidade possível dentro dos recursos medidos, com resultados
   reproduzíveis e versionados.
8. A deduplicação aproximada usa simhash com limiar fixo; corpora muito
   grandes exigem blocking por hash de shingles para escalar.
9. A quantização INT8 dinâmica não cobre a tabela de embeddings, que domina o
   tamanho do modelo exportado para mobile (6,11 MB).
10. O corpus da Fase 1 é majoritariamente literatura/livros de domínio público;
    o corpus de código vem de repositórios **permissivos** (MIT/Apache-2.0/
    BSD), com a licença de cada repositório conferida pela API do GitHub e
    gravada no `dataset_manifest.json`.
11. **O ciclo de 10 dias não foi executado como um ciclo de 10 dias.** Por
    segurança do dispositivo (e por decisão do §15 do pedido), o scheduler
    foi validado com execuções curtas reais; os números por modality vêm
    dessas execuções, não de 240 h de execução contínua.
