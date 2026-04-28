# GPT-2 PPO Evaluation

## Summary

- PPO policy was initialized from `sft_gpt2/model` and optimized against `reward_gpt2/model`.
- The reward model was trained on synthetic proxy preferences, so this measures RM optimization rather than human alignment.
- SFT test loss: `2.6772`; perplexity: `14.54`.
- PPO test loss: `2.6985`; perplexity: `14.86`.
- PPO minus SFT loss delta: `0.0213`.
- PPO reward delta on qualitative generations: `0.1343`.
- PPO-vs-SFT reward win rate: `0.5000`.
- Approx KL on PPO generations vs SFT: `-0.0011`.
- Generation sample size: `16` examples.

## Dataset

- PPO prompt source: `rlhf_dolly_datasets/ppo`.
- Dolly/SFT reference source: `rlhf_dolly_datasets/sft`.
- PPO prompts are prompt-only; references are joined from the SFT dataset by `source_id` for evaluation.

| Split | PPO rows | SFT rows |
| --- | ---: | ---: |
| train | 12010 | 12010 |
| validation | 1502 | 1502 |
| test | 1499 | 1499 |

## Training Configuration

| Setting | Value |
| --- | --- |
| Policy initialization | `sft_gpt2/model` |
| Frozen reference model | `sft_gpt2/model` |
| Reward model | `reward_gpt2/model` |
| Output model | `ppo_gpt2/model` |
| PPO epochs | `1` |
| Max new tokens | `96` |
| Learning rate | `1e-06` |
| KL coefficient | `0.02` |
| Clip range | `0.2` |
| Per-device batch | `4` |
| GPUs/processes | `4` |
| Precision | `bf16 via accelerate launch` |

## Training Diagnostics

| Metric | Value |
| --- | ---: |
| steps | `751` |
| train samples | `12010` |
| reward mean | `2.3106` |
| last reward mean | `2.4022` |
| selected-token KL mean | `-1.1715` |
| selected-token abs KL mean | `1.2718` |
| clip fraction mean | `0.8234` |
| ratio mean | `96433026.1965` |
| loss mean | `3416139.8407` |

The training diagnostics show an aggressive PPO update: high clip fraction and very large probability ratios appeared late in training. The final model still loads and evaluates, but these signals argue for a smaller LR, stronger KL control, or fewer/shorter rollouts before treating this as a stable RLHF policy.

## Test Metrics

| Metric | SFT | PPO | Delta |
| --- | ---: | ---: | ---: |
| loss | `2.6772` | `2.6985` | `0.0213` |
| perplexity | `14.54` | `14.86` | `0.31` |

## Generation Metrics

| Model | Reward mean | ROUGE-L F1 mean | Prompt copy rate | Mean response words |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 SFT | 1.1068 | 0.2470 | 0.6130 | 43.6 |
| GPT-2 PPO | 1.2411 | 0.2474 | 0.6192 | 41.0 |

## Qualitative Notes

- PPO increased mean reward on the sampled generations, but the reward win rate is only tied with SFT at this sample size.
- Test loss/perplexity worsened slightly, so PPO did not improve next-token fit to Dolly references.
- ROUGE-L stayed nearly flat, while copy rate increased slightly; the qualitative examples still show repetition and shallow instruction following.
- This report is downstream of a synthetic reward model; human preference evaluation would be needed before treating PPO as alignment progress.

## Examples

### brainstorming / databricks/databricks-dolly-15k:train:8120

**Prompt:** Tell me all the different ways I could eat an orange.

**Reference:** You could slice it in circles. You could cut it into cubes and add it to a fruit salad. You could juice it. You could rip it apart with your hands and eat with your teeth while showering. You could peel it first and eat segment by segment.

**SFT reward:** `0.1482`; **SFT:** 1. I could eat a banana, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel

**PPO reward:** `1.2499`; **PPO:** 1. I could eat a banana. 2. I could eat a banana with a spoon. 3. I could eat a banana with a fork. 4. I could eat a banana with a fork with a spoon. 5. I could eat a banana with a fork with a spoon. 6. I could eat a banana with a fork with a spoon. 7. I could eat a banana with a fork with a spoon. 8.

### brainstorming / databricks/databricks-dolly-15k:train:4866

**Prompt:** Tell me what I should do in London, England.

**Reference:** If you're visiting London, it may feel daunting attempting to take advantage of everything this cosmopolitan city has to offer. But that's part of its magic — there's something for everyone! A few ideas: Enjoy the food. I highly recommend wandering through the vendor stands at Borough Market in Southwark — it's only open for lunch and you should bring cash because most vendors don't take cards! If you're looking for a high-end restaurant, try Chiltern Firehouse in Marylebone. Another popular option is Dishoom: it's an upscale Indian restaurant with multiple locations and worth the hype. And...

**SFT reward:** `0.8729`; **SFT:** I should go to London, England. I should go to the Royal Opera House, London, England. I should go to the Royal Opera House, London, England. I should go to the Royal Opera House, London, England. I should go to the Royal Opera House, London, England. I should go to the Royal Opera House, London, England. I should go to the Royal Opera House, London, England. I should

**PPO reward:** `0.6974`; **PPO:** I should go to London, England. I should go to the Royal Opera House, London. I should go to the Royal Opera House, London. I should go to the Royal Opera House, London. I should go to the Royal Opera House, London. I should go to the Royal Opera House, London. I should go to the Royal Opera House, London. I should go to the Royal Opera House, London. I should

### classification / databricks/databricks-dolly-15k:train:10690

**Prompt:** If we wanted to play sports that involve a ball, which of these would we play: soccer, football, swimming, snowboard, skiing, baseball.

**Reference:** Soccer, football and baseball are sports that require a ball whereas swimming, snowboarding and skiing do not.

**SFT reward:** `2.2964`; **SFT:** The soccer ball would be played on a flat surface, while the football would be played on a flat surface.

**PPO reward:** `2.3266`; **PPO:** The soccer ball would be played on a stick, while the snowboard would be played on a ball.

### classification / databricks/databricks-dolly-15k:train:5211

**Prompt:** Which of the following bodies of water are below sea level: Dead Sea, Lake Ontario, Sea of Galilee, the Hudson Bay, Lake Assal?

**Reference:** The Dead Sea, the Sea of Galilee, and Lake Assal are all below sea level.

**SFT reward:** `3.3927`; **SFT:** Lake Ontario, Sea of Galilee, Lake Assal

**PPO reward:** `3.3927`; **PPO:** Lake Ontario, Sea of Galilee, Lake Assal

### closed_qa / databricks/databricks-dolly-15k:train:3688

**Prompt:** What is a chemical structure? Context: A chemical structure determination includes a chemist's specifying the molecular geometry and, when feasible and necessary, the electronic structure of the target molecule or other solid. Molecular geometry refers to the spatial arrangement of atoms in a molecule and the chemical bonds that hold the atoms together and can be represented using structural formulae and by molecular models; complete electronic structure descriptions include specifying the occupation of a molecule's molecular orbitals. Structure determination can be applied to a range of ta...

**Reference:** A chemical structure determination includes a chemist's specifying the molecular geometry and, when feasible and necessary, the electronic structure of the target molecule or other solid.

**SFT reward:** `1.0618`; **SFT:** A chemical structure determination includes a chemist's specifying the molecular geometry and, when feasible and necessary, the electronic structure of the target molecule or other solid. Molecular geometry refers to the spatial arrangement of atoms in a molecule and the chemical bonds that hold the atoms together and can be represented using structural formulae and by molecular models; complete electronic structure descriptions include specifying the occupation of a molecule's molecular orbitals. Structure determination can be applied to a range of targets from very simple molecules

**PPO reward:** `1.0618`; **PPO:** A chemical structure determination includes a chemist's specifying the molecular geometry and, when feasible and necessary, the electronic structure of the target molecule or other solid. Molecular geometry refers to the spatial arrangement of atoms in a molecule and the chemical bonds that hold the atoms together and can be represented using structural formulae and by molecular models; complete electronic structure descriptions include specifying the occupation of a molecule's molecular orbitals. Structure determination can be applied to a range of targets from very simple molecules

### closed_qa / databricks/databricks-dolly-15k:train:6113

**Prompt:** Wat is goede vrijdag? Context: Goede Vrijdag is de vrijdag voor Pasen. Op deze dag herdenken christenen de kruisiging en dood van Jezus. Jezus werd volgens de Bijbel veroordeeld tot de kruisdood door de Romeinse stadhouder Pontius Pilatus, op aandrang van het sanhedrin. Deze straf werd voltrokken op de heuvel Golgotha nabij de stad Jeruzalem. Goede Vrijdag volgt op Witte Donderdag en gaat vooraf aan Stille Zaterdag. Daarop volgt Pasen.

**Reference:** De dag dat Jezus gekruisigd werd

**SFT reward:** `3.0696`; **SFT:** Goede Vrijdag is de vrijdag voor Pasen.

**PPO reward:** `3.0696`; **PPO:** Goede Vrijdag is de vrijdag voor Pasen.

### creative_writing / databricks/databricks-dolly-15k:train:12290

**Prompt:** You stumble upon a new animal, one the world has never seen. What is it and how did you come up with it's name?

**Reference:** I can't believe I found the worlds first ever Micophant. I thought for sure what I had stumbled upon was a very tiny, baby elephant but I know baby elephants can weigh up to 200lbs at birth. This little 3lb creature with all the features of a full grown elephant was truly one of a kind. The most notable item my Microphant had was a tiny set of functional wings. So not only could this small creature move by foot, it could also fly! I was so astonished by this discovery I contacted my local Zoo which helped me identify this animal and notify the appropriate scientific community of this incred...

**SFT reward:** `1.3660`; **SFT:** The name of the animal is a combination of the Greek word for "animal" and the Latin word for "animal" for "animal." The name of the animal is a combination of the Greek word for "animal" and the Latin word for "animal" for "animal."

**PPO reward:** `1.3660`; **PPO:** The name of the animal is a combination of the Greek word for "animal" and the Latin word for "animal" for "animal." The name of the animal is a combination of the Greek word for "animal" and the Latin word for "animal" for "animal."

### creative_writing / databricks/databricks-dolly-15k:train:8849

**Prompt:** I have wine tasting plans for the weekend , give me some tips so that I look like an expert

**Reference:** The most important part of Wine tasting is to enjoy it and have fun. There are five important steps to tasting wine which are described below. See the Color - Take a look at the color of the wine you are about to taste, it will give you a good indication on the age of the wine. Young white wines are clear and look like water, while older wines that have been aged in oak get a gold tint. Young red wines are vivid purplish red while older wines have brownish tones. Swirl the wine - Swirling the wine helps to expose the wine to more oxygen and release its aroma, this is called as aerating the ...

**SFT reward:** `-0.1986`; **SFT:** 1. Drink a glass of wine. 2. Drink a glass of wine. 3. Drink a glass of wine. 4. Drink a glass of wine. 5. Drink a glass of wine. 6. Drink a glass of wine. 7. Drink a glass of wine. 8. Drink a glass of wine. 9. Drink a glass of wine. 10. Drink a glass

**PPO reward:** `-0.1986`; **PPO:** 1. Drink a glass of wine. 2. Drink a glass of wine. 3. Drink a glass of wine. 4. Drink a glass of wine. 5. Drink a glass of wine. 6. Drink a glass of wine. 7. Drink a glass of wine. 8. Drink a glass of wine. 9. Drink a glass of wine. 10. Drink a glass

## Conclusion

The PPO run completed the full engineering loop: prompt-only dataset, SFT policy, frozen SFT reference, synthetic GPT-2 reward model, local PPO checkpoint, and comparative evaluation.

The measured outcome is mixed. PPO improved the sampled RM reward by `0.1343`, but only tied SFT on pairwise RM wins and slightly worsened test perplexity. Given the unstable training diagnostics, this checkpoint is best treated as a working PPO prototype, not a clearly better aligned policy.