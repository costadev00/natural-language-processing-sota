# 01 Word Embeddings

## Objetivo

Construir uma base didatica para representacoes distribuidas de palavras, indo de modelos preditivos simples ate avaliacao intrinseca. Esta trilha mostra como CBOW, Skip-gram, negative sampling e GloVe conectam objetivos de treinamento diferentes a propriedades do espaco vetorial.

## Papers

- Mikolov et al. 2013, *Efficient Estimation of Word Representations in Vector Space*.
- Mikolov et al. 2013, *Distributed Representations of Words and Phrases and their Compositionality*.
- Pennington, Socher e Manning 2014, *GloVe: Global Vectors for Word Representation*.
- Schnabel et al. 2015, *Evaluation Methods for Unsupervised Word Embeddings*.

Veja tambem o [indice de papers](../../docs/paper-index.md).

## Implementacao

- [`notebooks/cbow-pytorch-didatico.ipynb`](notebooks/cbow-pytorch-didatico.ipynb): CBOW minimo em PyTorch.
- [`notebooks/skipgram-pytorch-didatico.ipynb`](notebooks/skipgram-pytorch-didatico.ipynb): Skip-gram didatico.
- [`notebooks/word2vec-negative-sampling-pytorch.ipynb`](notebooks/word2vec-negative-sampling-pytorch.ipynb): Skip-gram com negative sampling.
- [`notebooks/glove-pytorch-didatico.ipynb`](notebooks/glove-pytorch-didatico.ipynb): matriz de coocorrencia e objetivo GloVe.
- [`notebooks/embedding-evaluation-gensim-cornell.ipynb`](notebooks/embedding-evaluation-gensim-cornell.ipynb): avaliacao com Gensim e dados abertos do Cornell Eval.

## Como Executar

Use o ambiente raiz do repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook projects/01-word-embeddings/notebooks
```

Os notebooks foram pensados para execucao didatica e tambem funcionam bem no Google Colab, com downloads de datasets quando necessario.

## Resultados

- CBOW e Skip-gram deixam explicita a diferenca entre prever uma palavra pelo contexto e prever contexto pela palavra central.
- Negative sampling torna o treino Word2Vec mais pratico ao substituir softmax global por pares positivos/negativos.
- GloVe evidencia a alternativa baseada em estatisticas globais de coocorrencia.
- A avaliacao mostra que similaridade, analogia e tarefas de comparacao direta capturam aspectos diferentes da qualidade dos embeddings.

## Limitacoes

- Os notebooks didaticos usam corpus pequeno ou amostras para manter tempo de execucao baixo.
- Os resultados nao pretendem competir com embeddings treinados em escala.
- Algumas avaliacoes dependem de download externo em tempo de execucao.

## Proximos Passos

- Criar uma tabela comparativa unica com metricas dos notebooks.
- Adicionar testes pequenos para funcoes de geracao de pares e coocorrencia.
- Documentar hiperparametros recomendados para execucao curta e execucao completa.
