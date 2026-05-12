# Dados e Artefatos RLHF

Esta pasta documenta os dados da trilha RLHF. Os datasets locais, checkpoints e modelos gerados nao sao versionados no Git.

## Dataset Publicado

O dataset preparado a partir do Dolly 15k esta no Hugging Face Hub:

```text
costadev00/dolly-15k-rlhf-instructgpt-format
```

Configs disponiveis:

- `sft`: exemplos para supervised fine-tuning.
- `rm_schema`: schema para futura rotulagem de preferencias, com `chosen` e `rejected` vazios.
- `rm_synthetic`: pares proxy em que a resposta Dolly e `chosen` e uma resposta SFT gerada e `rejected`.
- `ppo`: prompts sem resposta para rollouts PPO.

Uso direto:

```python
from datasets import load_dataset

sft = load_dataset("costadev00/dolly-15k-rlhf-instructgpt-format", "sft")
rm_synthetic = load_dataset("costadev00/dolly-15k-rlhf-instructgpt-format", "rm_synthetic")
ppo = load_dataset("costadev00/dolly-15k-rlhf-instructgpt-format", "ppo")
```

## Reconstrucao Local

Execute a partir de `projects/04-rlhf`:

```bash
python3 scripts/build-dolly-rlhf-datasets.py --overwrite
python3 scripts/build-synthetic-rm-preferences.py --overwrite
```

O segundo comando exige um modelo SFT local em `sft_gpt2/model`, gerado por:

```bash
accelerate launch --multi_gpu --num_processes 4 scripts/train-gpt2-sft.py
```

## Aviso sobre Preferencias Sinteticas

`rm_synthetic` nao e dado de preferencia humana. Ele trata operacionalmente respostas Dolly como preferidas e respostas geradas pelo SFT como rejeitadas para exercitar o pipeline de reward modeling. Os resultados do reward model e do PPO devem ser lidos dentro dessa limitacao.

## Caminhos Ignorados

Os seguintes caminhos sao esperados localmente, mas ficam fora do Git:

- `rlhf_dolly_datasets/`
- `sft_gpt2/model/` e `sft_gpt2/checkpoints/`
- `reward_gpt2/model/` e `reward_gpt2/checkpoints/`
- `ppo_gpt2/model/`
- `hf_upload/`
- dumps `predictions_*.jsonl`, `generations.jsonl`, `examples.jsonl` e `rollout_metrics.jsonl`
