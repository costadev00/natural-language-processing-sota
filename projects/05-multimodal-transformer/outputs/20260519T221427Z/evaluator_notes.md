# Evaluator Notes

Votes were redone as an anonymized pairwise evaluation: for each prompt, the two responses were treated as Response A and Response B in randomized order before assigning a side preference. Model identities were revealed only when writing the final CSV mapping.

- `p01_brainstorming_birds`: better instruction following and usable caption blocks; the other response contains malformed artifacts and incomplete image placeholders.
- `p02_how_to_tv`: more complete and safer procedure, with two valid reference captions; the other response is truncated and has broken Markdown.
- `p03_hypothetical_vehicle`: more detailed tradeoff analysis and a clearer concept; the other response is generic and repeats the prompt.
- `p04_story_octopus`: complete four-scene children story with captions; the other response drifts into unrelated discussion.
- `p05_advice_workspace`: complete practical advice and two valid visual captions; the other response stops before completing the caption section.
- `p06_explanation_architecture`: more accurate explanation of image tokens and shared transformer processing; the other response incorrectly describes CNN/RNN-style components.
- `p07_identification_statue`: correct landmark identification with image-grounded clues and a comparison caption; the other response has artifacts and weaker visual evidence.
- `p08_comparison_animals`: correctly compares cats and parrots; the other response hallucinates a chameleon in the first image.
- `p09_report_landmark`: concise non-repetitive report with a clear caption; the other response repeats itself and includes artifacts.
- `p10_reasoning_scene`: correctly reasons about two cats resting; the other response hallucinates a chameleon and an attack scene.
