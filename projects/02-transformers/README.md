# 02 Transformers

## Objetivo

Explicar o Transformer a partir das pecas fundamentais: positional encoding, mascara causal, scaled dot-product attention, multi-head attention e uma implementacao minima para copy task. A trilha serve como ponte entre embeddings estaticos e modelos contextuais modernos.

## Papers

- Vaswani et al. 2017, *Attention Is All You Need*.

Veja tambem o [indice de papers](../../docs/paper-index.md).

## Implementacao

- [`notebooks/annotated-transformer-colab-illustrated.ipynb`](notebooks/annotated-transformer-colab-illustrated.ipynb): notebook ilustrado para Google Colab inspirado no estilo do Annotated Transformer.

## Como Executar

Use o ambiente raiz:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook projects/02-transformers/notebooks/annotated-transformer-colab-illustrated.ipynb
```

## Resultados

- A atencao e implementada explicitamente a partir de `Q`, `K` e `V`.
- A mascara causal mostra por que o decoder nao pode olhar tokens futuros.
- A copy task oferece um teste pequeno para validar a arquitetura completa.

## Limitacoes

- O notebook usa dados sinteticos, nao um corpus linguistico real.
- A implementacao favorece clareza em vez de performance.
- Nao ha treinamento de larga escala nem avaliacao em tarefas reais.

## Proximos Passos

- Adicionar testes unitarios para shapes, mascara causal e atencao.
- Criar uma secao comparando encoder-only, decoder-only e encoder-decoder.
- Conectar a trilha diretamente ao projeto BERT.
