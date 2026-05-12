# Natural Language Processing SOTA

Monorepo didatico e de portfolio tecnico com implementacoes, leituras e experimentos em Processamento de Linguagem Natural. O repositorio organiza a progressao de representacoes distribuidas, Transformers, BERT e RLHF, sempre conectando cada implementacao ao paper estudado e aos principais achados.

## Catalogo

| Trilha | Tema | Paper principal | Implementacao | Status | Resultado principal |
| --- | --- | --- | --- | --- | --- |
| [01 Word Embeddings](projects/01-word-embeddings/) | CBOW, Skip-gram, negative sampling, GloVe e avaliacao intrinseca | Word2Vec, GloVe, evaluation methods | 5 notebooks PyTorch/Gensim | Didatico completo | Comparacao pratica entre modelos preditivos, matriz global de coocorrencia e benchmarks de similaridade/analogia. |
| [02 Transformers](projects/02-transformers/) | Transformer encoder-decoder e attention | Attention Is All You Need | Notebook Colab ilustrado | Didatico completo | Implementacao minima com positional encoding, mascara causal, multi-head attention e copy task. |
| [03 BERT](projects/03-bert/) | BERT pequeno treinado do zero | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Notebook Colab e script PyTorch | Em evolucao | Fluxo didatico de MLM, NSP e fine-tuning downstream em portugues. |
| [04 RLHF](projects/04-rlhf/) | SFT, reward model, PPO e MMLU com GPT-2 | InstructGPT, PPO, Dolly, MMLU | Scripts Python + relatorios + metricas | Experimento completo | SFT melhora perplexidade no Dolly; reward model aprende preferencia sintetica; PPO otimiza reward mas fica instavel. |

## Estrutura

```text
natural-language-processing-sota/
├── docs/                 # indice de papers, trilha de leitura e politica de artefatos
├── papers/               # PDFs canonicos usados como referencia
├── projects/             # trilhas didaticas autocontidas
│   ├── 01-word-embeddings/
│   ├── 02-transformers/
│   ├── 03-bert/
│   └── 04-rlhf/
├── requirements.txt      # dependencias gerais dos notebooks/scripts didaticos
└── README.md
```

## Como Usar

Para os notebooks e scripts didaticos gerais:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para a trilha de RLHF, use o ambiente especifico:

```bash
cd projects/04-rlhf
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Os scripts de RLHF assumem execucao a partir de `projects/04-rlhf`, porque os caminhos padrao de dados, modelos e resultados sao relativos a essa pasta.

## Documentacao

- [Trilha de leitura](docs/learning-path.md)
- [Indice de papers](docs/paper-index.md)
- [Politica de artefatos](docs/artifact-policy.md)
- [Relatorios de disciplina](docs/course-reports/)

## Politica de Artefatos

Este repositorio versiona codigo, notebooks leves, documentacao, relatorios e metricas agregadas. Datasets locais, checkpoints, pesos de modelos, arquivos Arrow e dumps grandes de predicoes ficam fora do Git.

O dataset RLHF preparado a partir do Dolly 15k esta publicado no Hugging Face Hub:

```text
costadev00/dolly-15k-rlhf-instructgpt-format
```

Historico Git nao foi reescrito nesta reorganizacao. A limpeza atual vale para a arvore versionada daqui em diante.
