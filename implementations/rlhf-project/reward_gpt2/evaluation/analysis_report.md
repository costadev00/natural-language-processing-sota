# GPT-2 Reward Model Evaluation

## Summary

- Reward model: `GPT2ForSequenceClassification`, initialized from `sft_gpt2/model`.
- Training data: `rlhf_dolly_datasets/rm_synthetic`, a synthetic proxy preference dataset.
- Preference rule: Dolly `reference_response` is `chosen`; sampled GPT-2 SFT output is `rejected`.
- Training objective: Bradley-Terry / InstructGPT pairwise loss, `-logsigmoid(reward_chosen - reward_rejected)`.
- Validation pair loss: `0.5681`.
- Test pairwise accuracy: `0.7265` over `1499` pairs.
- Test pair loss: `0.5531`.
- Mean reward margin, chosen minus rejected: `0.5231`.

This model is useful for exercising the RLHF reward-modeling and later PPO pipeline, but it is not a human-aligned production reward model because the preferences are synthetic.

## Dataset

The generated dataset was saved locally as `rlhf_dolly_datasets/rm_synthetic` and uploaded publicly as the `rm_synthetic` config in:

`costadev00/dolly-15k-rlhf-instructgpt-format`

Hub URL:

https://huggingface.co/datasets/costadev00/dolly-15k-rlhf-instructgpt-format

Upload commit:

`7634576fd73cc482d00bdbf5650356cd672883cd`

| Split | Rows | Fallback rejected responses |
| --- | ---: | ---: |
| train | 12010 | 119 |
| validation | 1502 | 14 |
| test | 1499 | 11 |

Columns:

`prompt`, `chosen`, `rejected`, `reference_response`, `category`, `source_id`, `has_context`, `ready_for_rm`, `preference_source`, `rejected_model`

Synthetic rejection generation used `sft_gpt2/model` with `temperature=0.9`, `top_p=0.95`, `max_new_tokens=160`, `seed=42`, and batch size `32`. A small number of fallback rejections were inserted when generation was empty or identical to the chosen response.

## Training Configuration

| Setting | Value |
| --- | --- |
| Base checkpoint | `sft_gpt2/model` |
| Output model | `reward_gpt2/model` |
| Epochs | `1.0` |
| Max length | `1024` |
| Learning rate | `1e-5` |
| Warmup ratio | `0.03` |
| Weight decay | `0.01` |
| Per-device train batch | `8` |
| Per-device eval batch | `8` |
| Gradient accumulation | `4` |
| GPUs | `4` |
| Precision | `bf16` |
| Best checkpoint | `reward_gpt2/checkpoints/checkpoint-94` |

Training ran through DDP with `accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16`.

## Training Metrics

| Metric | Value |
| --- | ---: |
| train loss | `0.7292` |
| train runtime | `160.91s` |
| train samples | `12010` |
| train samples/sec | `74.64` |
| validation loss | `0.5681` |
| validation samples | `1502` |
| validation samples/sec | `216.85` |

The validation loss below the random pairwise baseline of about `0.6931` shows that the reward model learned a usable synthetic preference signal. The result should be read as "can distinguish Dolly references from SFT samples", not as "understands human preference".

## Test Metrics

| Metric | Value |
| --- | ---: |
| pairwise accuracy | `0.7265` |
| pair loss | `0.5531` |
| mean chosen reward | `2.6860` |
| mean rejected reward | `2.1629` |
| mean margin | `0.5231` |
| median margin | `0.4136` |
| chosen reward std | `0.6633` |
| rejected reward std | `0.8401` |

The accuracy is comfortably above random, and the positive mean margin indicates that the model usually assigns higher scalar reward to the Dolly answer than to the generated SFT rejection.

## Category Metrics

| Category | Samples | Accuracy | Pair loss | Mean margin |
| --- | ---: | ---: | ---: | ---: |
| brainstorming | 176 | 0.8920 | 0.4152 | 0.9165 |
| classification | 213 | 0.6479 | 0.6077 | 0.2770 |
| closed_qa | 177 | 0.5876 | 0.6939 | 0.1738 |
| creative_writing | 71 | 0.8028 | 0.4954 | 0.7616 |
| general_qa | 219 | 0.7808 | 0.4863 | 0.6560 |
| information_extraction | 150 | 0.7600 | 0.5512 | 0.6560 |
| open_qa | 374 | 0.7059 | 0.5519 | 0.4840 |
| summarization | 119 | 0.7059 | 0.6139 | 0.4695 |

The strongest categories are `brainstorming` and `creative_writing`, where generated rejected answers often contain repetition, drift, or generic content. The weakest categories are `closed_qa` and `classification`, where short rejected answers can look superficially valid and the model has less room to distinguish correctness from format.

## Bias Checks

| Signal | Pearson r with margin |
| --- | ---: |
| margin vs chosen words | `0.0867` |
| margin vs rejected words | `0.3211` |
| margin vs chosen copy rate | `-0.1862` |
| margin vs rejected copy rate | `-0.1100` |

The largest simple correlation is between margin and rejected response length (`0.3211`). This suggests the RM may partially learn artifacts of long sampled responses, especially when rejected generations become repetitive. That is expected for a synthetic preference dataset and should be monitored before PPO.

## Qualitative Analysis

The reward model often catches noisy SFT generations that repeat phrases, drift from the instruction, or answer with the wrong entity. For example, in hydration and bird-identification prompts, Dolly responses receive higher reward than obviously incomplete or incorrect generated answers.

The model still makes understandable mistakes. In classification and closed QA examples, some rejected answers are short and formatted similarly to a valid answer, while Dolly references may include typos or less polished formatting. The RM sometimes rewards surface fluency or length instead of factual correctness.

## Examples

### brainstorming / databricks/databricks-dolly-15k:train:3000

**Correct:** `False`; **margin:** `-0.0062`; **chosen reward:** `2.4869`; **rejected reward:** `2.4931`.

**Prompt:** What are some common character types in D&D?

**Chosen/Dolly:** Dungeons & Dragons (commonly D&D) has many character classes, and those classes have evolved throughout the different releases. The core character classes in the 3rd edition are Barbarian, Bard, Cleric, Druid, Fighter, Monk, Paladin, Ranger, Rogue, Sorcerer, and Wizard.

**Rejected/SFT sample:** The typical character types in D&D are those that are common to many different types of creatures, such as gnomes, giants, or elves. These characters can have one or several common traits, such as fear, anger, jealousy, and fear of death...

### brainstorming / databricks/databricks-dolly-15k:train:4160

**Correct:** `True`; **margin:** `2.0378`; **chosen reward:** `4.2104`; **rejected reward:** `2.1725`.

**Prompt:** give me a list of all the ways a person can hydrate

**Chosen/Dolly:** - drink water from a glass - drink water using a straw - drink water from a hose - suck on an ice cube - have a cucumber - get an IV - drink an electrolyte

**Rejected/SFT sample:** - Drink water - Clean - Clean up your environment - Clean your hair - Clean your sleep

### classification / databricks/databricks-dolly-15k:train:453

**Correct:** `False`; **margin:** `-0.5830`; **chosen reward:** `2.0478`; **rejected reward:** `2.6307`.

**Prompt:** Categorize each of the following as either 'spicy', 'not spicy' or 'sometimes spicy': jalapenos, chilis, sriracha, chips, pizza, cocktails, fruit, milk

**Chosen/Dolly:** Jalapenos are spicy. Chilis are spicy. Sriracha is spicy. Chips are sometimes spicy. Pizza is sometimes spicy. Cocktails are sometimes spicy. Fruit is not spicy. MIlk is not spicy.

**Rejected/SFT sample:** * jalapenos * chilis * sriracha * chips * fruit * milk * jalapenos

### classification / databricks/databricks-dolly-15k:train:2158

**Correct:** `True`; **margin:** `0.3215`; **chosen reward:** `2.8376`; **rejected reward:** `2.5161`.

**Prompt:** Identify the bird from the list: Canada Dry, Canada goose, Goosebumps

**Chosen/Dolly:** Canada goose

**Rejected/SFT sample:** Canada Dry

### closed_qa / databricks/databricks-dolly-15k:train:6046

**Correct:** `False`; **margin:** `-0.1373`; **chosen reward:** `2.1521`; **rejected reward:** `2.2894`.

**Prompt:** Given this reference test, what is the daily recommended amount of magnesium for women in the U.S.? Context: In the UK, the recommended daily values for magnesium are 300 mg for men and 270 mg for women. In the U.S. the Recommended Dietary Allowances are 400 mg for men ages 19-30 and 420 mg for older; for women 310 mg for ages 19-30 and 320 mg for older.

**Chosen/Dolly:** The daily recommended amount of magnesium for women in the U.S. is 310 mg for ages 19-30 and 320 mg for older.

**Rejected/SFT sample:** * 300 mg = 300 mg for men and 270 mg for women * 400 mg = 270 mg for men

### closed_qa / databricks/databricks-dolly-15k:train:12232

**Correct:** `True`; **margin:** `0.1126`; **chosen reward:** `3.0507`; **rejected reward:** `2.9381`.

**Prompt:** Given a reference text about the Annapurna Post, tell me when it launched its online news portal and when it became successful.

**Chosen/Dolly:** Annapurna Post launched its online news portal in 2013. By April 2014, it became among the 10 most visited news portals in Nepal.

**Rejected/SFT sample:** Annapurna Post

## Conclusion

The reward model learned a measurable synthetic preference signal: it assigns higher reward to Dolly references than to sampled SFT generations in about `72.65%` of held-out test pairs. This is enough to support the next RLHF engineering step, such as PPO smoke tests and reward-scored rollouts.

The main limitation is the source of preference. Because `chosen` and `rejected` are synthetic rather than human-labeled, the model can learn generation artifacts, response length patterns, or Dolly-specific style. Before using this as a serious alignment reward, the `rm_schema` pool should be populated with audited human preference pairs or a stronger labeling process.
