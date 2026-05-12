# 04 RLHF

## Objetivo

Implementar um pipeline local de RLHF em escala didatica usando GPT-2 e Dolly 15k: preparacao de datasets, supervised fine-tuning, reward modeling, PPO e avaliacao MMLU. Esta trilha e autocontida e guarda scripts, relatorios e metricas agregadas.

## Papers

- Ouyang et al. 2022, *Training Language Models to Follow Instructions with Human Feedback*.
- Radford et al. 2019, *Language Models are Unsupervised Multitask Learners*.
- Hendrycks et al. 2021, *Measuring Massive Multitask Language Understanding*.
- Schulman et al. 2017, *Proximal Policy Optimization Algorithms*.

Veja tambem o [indice de papers](../../docs/paper-index.md).

## Implementacao

- [`scripts/build-dolly-rlhf-datasets.py`](scripts/build-dolly-rlhf-datasets.py): cria datasets SFT, RM schema e PPO a partir do Dolly.
- [`scripts/train-gpt2-sft.py`](scripts/train-gpt2-sft.py) e [`scripts/evaluate-gpt2-sft.py`](scripts/evaluate-gpt2-sft.py): treino e avaliacao SFT.
- [`scripts/build-synthetic-rm-preferences.py`](scripts/build-synthetic-rm-preferences.py): cria pares de preferencia sintetica Dolly-vs-SFT.
- [`scripts/train-gpt2-reward-model.py`](scripts/train-gpt2-reward-model.py) e [`scripts/evaluate-gpt2-reward-model.py`](scripts/evaluate-gpt2-reward-model.py): reward model com loss Bradley-Terry.
- [`scripts/train-gpt2-ppo.py`](scripts/train-gpt2-ppo.py) e [`scripts/evaluate-gpt2-ppo.py`](scripts/evaluate-gpt2-ppo.py): ajuste PPO e comparacao com SFT.
- [`scripts/evaluate-gpt2-mmlu.py`](scripts/evaluate-gpt2-mmlu.py): avaliacao zero-shot e five-shot no MMLU.
- [`scripts/upload-to-hf.py`](scripts/upload-to-hf.py): empacota e publica os datasets no Hugging Face Hub.

## Como Executar

Execute os comandos a partir desta pasta:

```bash
cd projects/04-rlhf
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Fluxo principal:

```bash
python3 scripts/build-dolly-rlhf-datasets.py --overwrite
accelerate launch --multi_gpu --num_processes 4 scripts/train-gpt2-sft.py
python3 scripts/evaluate-gpt2-sft.py
python3 scripts/build-synthetic-rm-preferences.py --overwrite
accelerate launch --multi_gpu --num_processes 4 scripts/train-gpt2-reward-model.py
python3 scripts/evaluate-gpt2-reward-model.py
accelerate launch --multi_gpu --num_processes 4 scripts/train-gpt2-ppo.py
python3 scripts/evaluate-gpt2-ppo.py
python3 scripts/evaluate-gpt2-mmlu.py
```

Os checkpoints, datasets locais e dumps grandes sao ignorados pelo Git. Veja [`data/README.md`](data/README.md).

## Resultados

Relatorios completos:

- [SFT](reports/sft/analysis-report.md)
- [Reward model](reports/reward-model/analysis-report.md)
- [PPO](reports/ppo/analysis-report.md)
- [MMLU](reports/mmlu/analysis-report.md)
- [Relatorio final LaTeX](reports/final-report/relatorio-entrega-aula9.tex)

Metricas agregadas ficam em [`results/`](results/).

Resumo dos achados:

- SFT reduziu loss de teste no Dolly de `2.7979` para `2.4550`, com perplexidade de `16.41` para `11.65`.
- O reward model atingiu acuracia pairwise de `0.7265` sobre preferencias sinteticas, com margem media `0.5231`.
- PPO aumentou o reward medio das geracoes amostradas em `0.1343`, mas piorou levemente perplexidade e teve sinais de instabilidade.
- No MMLU, o SFT nao melhorou conhecimento multitarefa: GPT-2 base five-shot ficou em `26.60%`, e GPT-2 SFT five-shot em `25.80%`.

## Limitacoes

- O reward model usa preferencias sinteticas, nao preferencias humanas reais.
- PPO mede otimizacao do reward model local, nao alinhamento humano.
- Os modelos sao pequenos e os resultados nao devem ser tratados como SOTA.
- Alguns scripts esperam GPU para treino completo.

## Proximos Passos

- Substituir preferencias sinteticas por anotacoes humanas ou pares auditados.
- Reduzir instabilidade do PPO com KL mais forte, menor learning rate e rollouts mais curtos.
- Criar testes automatizados para schemas de dataset, masking e avaliadores.
- Publicar checkpoints finais fora do Git, com hashes e instrucoes de reproducibilidade.
