# APOLO ZENITH 1.9 — corpus de 30k e ciclo de 30 dias

## Ordem obrigatória

1. O workflow **preflight** deve passar em \`main\`. Ele executa a suíte de testes, valida CLIs, checkpoint/resume e a integridade do tokenizer. Ele não inicia o relógio.
2. Executar manualmente \`APOLO ZENITH 1.9 — construir e auditar corpus de 30.000 documentos\`. A auditoria exige exatamente 30.000 documentos, no mínimo 15.000 em cada idioma, hashes únicos e íntegros, licenças aprovadas e splits com tokens mínimos. O workflow não treina o modelo.
3. Só depois dos dois resultados bem-sucedidos, executar manualmente \`APOLO ZENITH 1.9 — ciclo incremental de treinamento de 30 dias\` com \`command=start\`.
4. O primeiro start valida os artefatos de preflight/corpus e o checkpoint inicial antes de gravar \`started_at\` e \`ends_at\`. O agendamento não cria o relógio por conta própria.
5. Depois do start, o agendamento tenta uma sessão a cada seis horas. Cada sessão retoma pesos, AdamW, scheduler e RNG do checkpoint anterior, executa 1.000 passos e publica um novo artefato de estado. O artefato anterior só é excluído após o novo ser publicado.
6. Ao atingir 30 dias, o ciclo deixa de treinar e executa a avaliação independente existente com revisão humana marcada como pendente (\`false\`). Uma aprovação humana nunca é inferida automaticamente.

## Critérios preservados

- Tokenizer existente, vocabulário 8.192; não é refeito.
- Corpus exato: 30.000 documentos; mínimo 15.000 PT e 15.000 EN.
- Treinamento só inicia com auditoria aprovada.
- Early stopping desativado para que cada sessão cumpra os passos solicitados; isso não relaxa nenhum critério de qualidade.
- A aprovação continua exigindo os gates de qualidade linguística existentes e revisão semântica humana. PPL menor não substitui coerência nem revisão humana.

## Persistência e limites operacionais

O ciclo não mantém um processo único vivo por 30 dias. Ele usa execuções agendadas, checkpoint completo e estado JSON com hash, timestamp, passo e histórico. O corpus é publicado como release asset versionado e imutável (tag vinculada ao run e ao hash do manifesto), depois cacheado para as sessões seguintes; assim ele não depende da retenção de artefatos do Actions. O artefato de estado é substituído de forma segura. O checkpoint de código de referência é copiado para um release do repositório no início, evitando depender da expiração do artefato original durante o ciclo. Se o runner, agendamento, armazenamento ou download falhar, a sessão falha e o relógio não é reiniciado; a próxima execução tenta retomar do último artefato íntegro.

GitHub-hosted runners têm limite de seis horas por job. O cron é executado no branch padrão e pode atrasar em períodos de alta carga. O estado registra lacunas superiores a 12 horas entre verificações; por isso, o monitoramento é periódico e auditável, não uma garantia de processo 24/7 sem lacunas. O pacote do corpus, publicado como release asset, é limitado a 1 GB; o pacote do estado/checkpoint, armazenado como artefato do Actions, é limitado a 150 MB. O artefato de estado anterior só é excluído depois que o novo foi publicado.

## Não incluído nesta fase

Não há coleta de 100 milhões de arquivos, integração multimodal completa, visão treinada 10.000 vezes ou contexto de 3 milhões de tokens. Essas metas exigem projetos separados e medições técnicas próprias; não são anunciadas como capacidades já implementadas.
