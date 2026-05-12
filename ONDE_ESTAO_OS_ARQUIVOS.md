# Onde Estao os Arquivos?

A pasta antiga `implementations/` foi substituida por uma organizacao por projeto dentro de `projects/`.

## Mapa Rapido

| Antes | Agora |
| --- | --- |
| `implementations/cbow_pytorch_didatico (1).ipynb` | `projects/01-word-embeddings/notebooks/cbow-pytorch-didatico.ipynb` |
| `implementations/skipgram_pytorch_didatico (1).ipynb` | `projects/01-word-embeddings/notebooks/skipgram-pytorch-didatico.ipynb` |
| `implementations/word2vec_pytorch_colab (1).ipynb` | `projects/01-word-embeddings/notebooks/word2vec-negative-sampling-pytorch.ipynb` |
| `implementations/glove_pytorch_didatico (1).ipynb` | `projects/01-word-embeddings/notebooks/glove-pytorch-didatico.ipynb` |
| `implementations/avaliacao_embeddings_gensim_cornell_colab (1).ipynb` | `projects/01-word-embeddings/notebooks/embedding-evaluation-gensim-cornell.ipynb` |
| `implementations/Annotated_Transformer_Colab_Illustrated.ipynb` | `projects/02-transformers/notebooks/annotated-transformer-colab-illustrated.ipynb` |
| `implementations/BERT_sentiment_analysis_wikipedia.ipynb` | `projects/03-bert/notebooks/bert-sentiment-analysis-wikipedia.ipynb` |
| `implementations/bert_didatico.py` | `projects/03-bert/scripts/bert-didatico.py` |
| `implementations/rlhf-project/*.py` | `projects/04-rlhf/scripts/` |
| `implementations/rlhf-project/relatorio_entrega_aula9.tex` | `projects/04-rlhf/reports/final-report/relatorio-entrega-aula9.tex` |
| `implementations/rlhf-project/references.bib` | `projects/04-rlhf/reports/final-report/references.bib` |
| `implementations/rlhf-project/*/evaluation/analysis_report.md` | `projects/04-rlhf/reports/` |
| `implementations/rlhf-project/*/metrics/*.json` | `projects/04-rlhf/results/` |
| `implementations/literature-review/*.pdf` papers | `papers/` |
| `implementations/literature-review/*.pdf` relatorios | `docs/course-reports/` |

## Projetos

- `projects/01-word-embeddings/`: CBOW, Skip-gram, Word2Vec, GloVe e avaliacao de embeddings.
- `projects/02-transformers/`: Transformer didatico.
- `projects/03-bert/`: BERT didatico e notebook de fine-tuning.
- `projects/04-rlhf/`: pipeline RLHF, scripts, relatorios, metricas e artefatos locais ignorados.

## Artigos

Os artigos ficam em `papers/`.

O indice fica em `docs/paper-index.md`.

## Observacao

Se o editor ainda mostrar abas como `rlhf-project/...` ou `implementations/...`, elas sao caminhos antigos. Abra a pasta raiz:

```text
/home/matheuscm/natural-language-processing-sota
```

