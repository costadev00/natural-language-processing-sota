# Projeto 05 - Transformer Multimodal

Este projeto reproduz, em escala reduzida, a avaliação humana pareada descrita no paper **Chameleon: Mixed-Modal Early-Fusion Foundation Models**. A comparação implementada aqui usa:

- `facebook/chameleon-7b`, executado localmente via Hugging Face Transformers.
- `gpt-5-nano`, acessado via OpenAI API.

A avaliação usa 10 prompts inspirados nas categorias do paper e salva as respostas lado a lado para votação humana.

## Arquivos principais

- `artigo.tex`: relatório final do projeto.
- `chameleon-paper.pdf`: paper usado como referência principal.
- `prompts.json`: 10 prompts da reprodução.
- `run_eval.py`: gera respostas do Chameleon e da OpenAI.
- `vote.py`: apresenta pares anonimizados A/B e registra votos.
- `export_results.py`: gera resumos em Markdown, JSON e LaTeX.
- `requirements-eval.txt`: dependências Python para rodar a avaliação.
- `outputs/`: resultados gerados pelas execuções.

## Limitação importante

O paper descreve o Chameleon como um modelo capaz de trabalhar com sequências intercaladas de texto e imagem. Porém, os checkpoints públicos disponíveis no Hugging Face são expostos no fluxo usado aqui como `image-text-to-text`: aceitam texto e imagens como entrada, mas geram texto como saída.

Por isso, prompts que pedem imagens usam blocos:

```text
<caption>...</caption>
```

Esses blocos representam a imagem que seria gerada, mas este projeto não materializa imagens reais a partir do Chameleon.

## Pré-requisitos

As chaves devem estar no arquivo `.env` na raiz do repositório:

```bash
OPENAI_API_KEY=...
HF_TOKEN=...
```

O token do Hugging Face precisa ter acesso ao modelo `facebook/chameleon-7b`. Para usar `facebook/chameleon-30b`, também é necessário ter acesso liberado e mais memória/tempo de execução.

## Instalação

A partir da raiz do repositório:

```bash
cd projects/05-multimodal-transformer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-eval.txt
```

Verifique se o ambiente está correto:

```bash
python3 run_eval.py --check-only
```

## Rodando a avaliação

Execução recomendada com o Chameleon 7B em uma única GPU:

```bash
CUDA_VISIBLE_DEVICES=3 python3 run_eval.py \
  --model facebook/chameleon-7b \
  --fallback-model facebook/chameleon-7b \
  --limit 10 \
  --no-openai-image-render \
  --max-new-tokens 350
```

O resultado esperado é um arquivo:

```bash
outputs/latest/responses.jsonl
```

com 20 linhas: 10 respostas do Chameleon e 10 respostas da OpenAI.

Se quiser fazer apenas um teste rápido:

```bash
CUDA_VISIBLE_DEVICES=3 python3 run_eval.py \
  --model facebook/chameleon-7b \
  --fallback-model facebook/chameleon-7b \
  --limit 1 \
  --no-openai-image-render \
  --max-new-tokens 350
```

## Votação pareada

Para votar manualmente nos pares A/B:

```bash
python3 vote.py outputs/latest/responses.jsonl
```

O script embaralha os lados A/B por prompt e grava:

```bash
outputs/latest/votes.csv
```

Opções de voto:

- `a`: resposta A é melhor.
- `b`: resposta B é melhor.
- `t`: empate.
- `s`: pular.
- `q`: sair.

## Exportando resultados

Depois da votação:

```bash
python3 export_results.py outputs/latest
```

Isso gera:

- `comparison.md`: respostas lado a lado.
- `summary.md`: resumo da execução.
- `summary.json`: resumo estruturado.
- `latex_table.tex`: tabela pronta para incluir no relatório.

## Resultado obtido nesta reprodução

Na execução registrada em `outputs/20260519T221427Z`, foram geradas 20 respostas para 10 prompts. A avaliação pareada anonimizada foi feita pelo autor do relatório. O resultado foi:

- `gpt-5-nano`: 10 vitórias.
- `facebook/chameleon-7b`: 0 vitórias.
- Empates: 0.

Qualitativamente, o Chameleon 7B apresentou alguns problemas recorrentes: truncamentos, repetições, artefatos textuais, falhas em seguir o formato `<caption>` e alucinações visuais em prompts com imagens. O `gpt-5-nano` foi mais consistente no seguimento das instruções e na interpretação das imagens de entrada.

## Observações operacionais

- O primeiro carregamento do Chameleon pode demorar por causa do download dos pesos.
- Para o 7B, usar uma única GPU foi mais eficiente do que espalhar o modelo em várias GPUs.
- Para o 30B, use quantização e limite de memória por GPU, por exemplo:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_eval.py \
  --model facebook/chameleon-30b \
  --limit 1 \
  --no-openai-image-render \
  --max-memory-per-gpu 18GiB
```

Na reprodução final deste projeto, foi usado o 7B por viabilidade operacional.
