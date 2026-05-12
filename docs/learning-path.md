# Trilha de Leitura

## 1. Representacoes de Palavras

Comece por CBOW e Skip-gram para entender embeddings preditivos, depois avance para negative sampling e GloVe. Feche a trilha com avaliacao intrinseca para ver por que similaridade vetorial nao basta como criterio unico.

- Projeto: [01 Word Embeddings](../projects/01-word-embeddings/)
- Papers: Mikolov 2013, Pennington 2014, Schnabel 2015.

## 2. Transformers

Depois dos embeddings estaticos, estude attention como mecanismo de composicao contextual. O notebook do Transformer monta as pecas essenciais antes de chegar em modelos grandes.

- Projeto: [02 Transformers](../projects/02-transformers/)
- Paper: Vaswani et al. 2017.

## 3. BERT

Use BERT para ver a passagem de representacoes contextuais bidirecionais para tarefas downstream. A implementacao e reduzida, mas preserva tokenizer, MLM, NSP e fine-tuning.

- Projeto: [03 BERT](../projects/03-bert/)
- Paper: Devlin et al. 2018.

## 4. RLHF

Finalize com SFT, reward modeling e PPO. A trilha mostra o loop de engenharia completo com GPT-2 e Dolly, mais uma avaliacao MMLU para testar transferencia de conhecimento.

- Projeto: [04 RLHF](../projects/04-rlhf/)
- Papers: InstructGPT, GPT-2, MMLU e PPO.
