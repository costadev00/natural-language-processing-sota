# 03 BERT

## Objetivo

Estudar BERT como encoder Transformer bidirecional, com pre-treinamento auto-supervisionado e reaproveitamento em tarefas downstream. A implementacao e reduzida para fins didaticos, mas preserva as ideias centrais do artigo: WordPiece, MLM, NSP e uso do token `[CLS]` para classificacao.

## Papers

- Devlin et al. 2018, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
- Vaswani et al. 2017, *Attention Is All You Need*, como base arquitetural.

O PDF do BERT ainda nao esta versionado localmente; veja o [indice de papers](../../docs/paper-index.md) para o estado do acervo.

## Implementacao

- [`notebooks/bert-sentiment-analysis-wikipedia.ipynb`](notebooks/bert-sentiment-analysis-wikipedia.ipynb): BERT pequeno treinado do zero em portugues, com Wikipedia, MLM, NSP e fine-tuning.
- [`scripts/bert-didatico.py`](scripts/bert-didatico.py): demonstracao compacta em PyTorch para MLM e NSP.

## Como Executar

Para a demonstracao curta:

```bash
python3 projects/03-bert/scripts/bert-didatico.py
```

Para o notebook:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook projects/03-bert/notebooks/bert-sentiment-analysis-wikipedia.ipynb
```

## Resultados

- O script mostra o fluxo minimo de tokenizer, pares de sentencas, MLM e NSP.
- O notebook registra uma versao em escala reduzida com corpus real em portugues e fine-tuning downstream.
- A trilha deixa claro que BERT-base real exige corpus, hardware e tempo de treino muito maiores.

## Limitacoes

- O modelo e pequeno e treinado do zero, portanto nao reproduz desempenho de BERT-base.
- O notebook depende de datasets externos em streaming.
- Os resultados devem ser lidos como validacao de pipeline, nao como benchmark competitivo.

## Proximos Passos

- Adicionar o PDF do BERT ao acervo `papers/`.
- Separar funcoes reutilizaveis do notebook em modulo Python testavel.
- Consolidar metricas de fine-tuning em `results/`.
