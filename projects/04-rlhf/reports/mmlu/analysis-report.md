# GPT-2 MMLU Evaluation

## Protocol

- Dataset: `cais/mmlu`.
- Test split is scored; dev split supplies up to five few-shot examples.
- Prediction is the largest normalized next-token probability among `A`, `B`, `C`, and `D`.
- The local `gpt2` baseline is the small GPT-2 checkpoint, not the 1.5B GPT-2 model cited in the MMLU appendix.
- This `cais/mmlu` snapshot exposes 14042 test examples; the paper reports 14079.

## Overall Accuracy

| Model | Mode | Examples | Accuracy (%) | Avg confidence (%) | RMS cal. error (%) | Truncated | Mean shots |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | five_shot | 14042 | 26.60 | 44.58 | 20.78 | 0 | 4.58 |
| base | zero_shot | 14042 | 22.96 | 84.60 | 62.12 | 0 | 0.00 |
| sft | five_shot | 14042 | 25.80 | 43.92 | 21.10 | 0 | 4.58 |
| sft | zero_shot | 14042 | 22.92 | 85.29 | 63.12 | 0 | 0.00 |

## Supercategories

| Model | Mode | Supercategory | Examples | Accuracy (%) | Avg confidence (%) |
| --- | --- | --- | ---: | ---: | ---: |
| base | five_shot | Humanities | 4705 | 23.85 | 39.98 |
| base | five_shot | Other | 3107 | 26.59 | 41.97 |
| base | five_shot | STEM | 3153 | 26.48 | 50.99 |
| base | five_shot | Social Sciences | 3077 | 30.94 | 47.69 |
| base | zero_shot | Humanities | 4705 | 24.19 | 87.06 |
| base | zero_shot | Other | 3107 | 23.88 | 80.44 |
| base | zero_shot | STEM | 3153 | 21.38 | 84.49 |
| base | zero_shot | Social Sciences | 3077 | 21.77 | 85.16 |
| sft | five_shot | Humanities | 4705 | 25.06 | 41.27 |
| sft | five_shot | Other | 3107 | 27.16 | 42.36 |
| sft | five_shot | STEM | 3153 | 25.72 | 51.12 |
| sft | five_shot | Social Sciences | 3077 | 25.64 | 42.16 |
| sft | zero_shot | Humanities | 4705 | 24.17 | 88.12 |
| sft | zero_shot | Other | 3107 | 23.82 | 80.26 |
| sft | zero_shot | STEM | 3153 | 21.28 | 85.44 |
| sft | zero_shot | Social Sciences | 3077 | 21.81 | 85.89 |

## Lowest-Accuracy Subjects

### base / five_shot

| Subject | Supercategory | Examples | Accuracy (%) |
| --- | --- | ---: | ---: |
| global facts | Other | 100 | 15.00 |
| formal logic | Humanities | 126 | 15.08 |
| astronomy | STEM | 152 | 17.11 |
| marketing | Other | 234 | 17.52 |
| jurisprudence | Humanities | 108 | 17.59 |
| business ethics | Other | 100 | 18.00 |
| abstract algebra | STEM | 100 | 19.00 |
| international law | Humanities | 121 | 19.01 |
| electrical engineering | STEM | 145 | 19.31 |
| philosophy | Humanities | 311 | 19.61 |

### base / zero_shot

| Subject | Supercategory | Examples | Accuracy (%) |
| --- | --- | ---: | ---: |
| high school chemistry | STEM | 203 | 14.78 |
| high school statistics | STEM | 216 | 15.28 |
| management | Other | 103 | 17.48 |
| high school biology | STEM | 310 | 17.74 |
| astronomy | STEM | 152 | 17.76 |
| global facts | Other | 100 | 18.00 |
| high school geography | Social Sciences | 198 | 18.18 |
| philosophy | Humanities | 311 | 18.33 |
| security studies | Social Sciences | 245 | 18.78 |
| high school psychology | Social Sciences | 545 | 19.08 |

### sft / five_shot

| Subject | Supercategory | Examples | Accuracy (%) |
| --- | --- | ---: | ---: |
| public relations | Social Sciences | 110 | 15.45 |
| formal logic | Humanities | 126 | 15.87 |
| global facts | Other | 100 | 16.00 |
| astronomy | STEM | 152 | 17.76 |
| college chemistry | STEM | 100 | 18.00 |
| computer security | STEM | 100 | 19.00 |
| electrical engineering | STEM | 145 | 20.00 |
| college medicine | Other | 173 | 20.23 |
| world religions | Humanities | 171 | 21.05 |
| jurisprudence | Humanities | 108 | 21.30 |

### sft / zero_shot

| Subject | Supercategory | Examples | Accuracy (%) |
| --- | --- | ---: | ---: |
| high school chemistry | STEM | 203 | 15.27 |
| high school statistics | STEM | 216 | 15.28 |
| high school biology | STEM | 310 | 17.10 |
| management | Other | 103 | 17.48 |
| astronomy | STEM | 152 | 17.76 |
| global facts | Other | 100 | 18.00 |
| high school geography | Social Sciences | 198 | 18.18 |
| philosophy | Humanities | 311 | 18.65 |
| security studies | Social Sciences | 245 | 18.78 |
| college chemistry | STEM | 100 | 19.00 |
