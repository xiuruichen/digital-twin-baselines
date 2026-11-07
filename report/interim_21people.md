# Digital-twin evaluation report

Baseline on Twin-2K-500 holdout (wave 4) items.

- **bonsai2-9b** reaches 62.8% of human test-retest consistency (95% CI 56.6%-69.5%, 21 people); on held-out people, answers in the low-confidence half are 1.61x as likely to be wrong as high-confidence ones (within-question AUROC 0.53).

Config: bonsai2-9b: 100 people, persona 4000 chars, shuffle_seed=None

## Individual-level accuracy

| Model | People | Answers | Accuracy | Human retest acc. | Normalized acc. (95% CI) | Crowd-mode baseline | Chance | Valid-letter mass |
|---|---|---|---|---|---|---|---|---|
| bonsai2-9b | 21 | 735 | 37.4% | 59.6% | 62.8% (56.6%-69.5%) | 38.9% | 29.2% | 0.933 |

## Population-level distribution distance

| Model | Questions | TV soft (mean probs) | TV hard (argmax) | TV uniform guess | TV human retest (floor) |
|---|---|---|---|---|---|
| bonsai2-9b | 55 | 0.294 | 0.554 | 0.319 | 0.201 |

## Confidence: does it predict errors?

| Model | Confidence signal | AUROC pooled | AUROC within-question | AUROC pooled (test half) | Acc. low-conf (test) | Acc. high-conf (test) | Error ratio low/high | ECE (test) |
|---|---|---|---|---|---|---|---|---|
| bonsai2-9b | top-option prob. | 0.727 | 0.528 | 0.725 | 18.3% | 49.4% | 1.61x | 0.091 |
| bonsai2-9b | top-2 margin | 0.698 | 0.536 | 0.676 | 23.5% | 43.9% | 1.36x | 0.176 |

## Per-question population TV

| QID | bonsai2-9b: TV soft / human retest |
|---|---|
| QID157 | 0.32 / 0.31 (n=13) |
| QID158 | 0.43 / 0.50 (n=8) |
| QID159 | 0.33 / 0.33 (n=13) |
| QID160 | 0.47 / 0.29 (n=8) |
| QID161 | 0.27 / 0.17 (n=12) |
| QID162 | 0.36 / 0.33 (n=9) |
| QID163 | 0.31 / 0.18 (n=11) |
| QID165 | 0.02 / 0.30 (n=10) |
| QID167 | 0.42 / 0.00 (n=11) |
| QID169 | 0.08 / 0.20 (n=10) |
| QID171 | 0.76 / 0.50 (n=4) |
| QID172 | 0.29 / 0.25 (n=12) |
| QID173 | 0.42 / 0.20 (n=5) |
| QID174 | 0.48 / 0.25 (n=4) |
| QID175 | 0.32 / 0.17 (n=12) |
| QID176 | 0.41 / 0.20 (n=5) |
| QID177 | 0.28 / 0.50 (n=4) |
| QID178 | 0.24 / 0.08 (n=12) |
| QID179 | 0.66 / 0.60 (n=5) |
| QID183 | 0.11 / 0.08 (n=13) |
| QID184 | 0.54 / 0.25 (n=8) |
| QID192 | 0.07 / 0.11 (n=9) |
| QID193 | 0.16 / 0.00 (n=12) |
| QID194 | 0.48 / 0.09 (n=11) |
| QID195 | 0.38 / 0.40 (n=10) |
| QID196 | 0.01 / 0.10 (n=21) |
| QID198 | 0.13 / 0.12 (n=8) |
| QID203 | 0.15 / 0.08 (n=13) |
| QID287 | 0.21 / 0.15 (n=21) |
| QID288 | 0.39 / 0.23 (n=21) |
| QID289 | 0.44 / 0.18 (n=21) |
| QID291 | 0.30 / 0.10 (n=21) |
| QID9_1 | 0.09 / 0.10 (n=21) |
| QID9_13 | 0.11 / 0.05 (n=21) |
| QID9_22 | 0.03 / 0.05 (n=21) |
| QID9_28 | 0.13 / 0.10 (n=21) |

## Notes

- **Scope:** categorical holdout items only (multiple choice + Likert matrix rows) from the wave-4 heuristics-and-biases / behavioral-economics block; a fixed seeded subset of rows/pricing items is used to bound compute. Sliders and free-text numeric items are not scored in this phase.
- **Persona:** compact 'question: answer' rendering of wave 1-3 answers (demographics, personality, economic preferences) truncated to a character budget; the holdout items are never in the persona.
- **Normalized accuracy** = model accuracy / the same people's wave 1-3 vs wave 4 agreement, on the same items. Exact-match on ordinal scales is harsh for both, which is why the ratio is the headline.
- **Individual and population metrics are separate on purpose.** A model can match the crowd (low TV) yet be no better than the crowd-mode baseline for individuals.
- **Confidence** is evaluated out of sample: people are split in half by id hash; bucket edges and the low/high threshold are fit on one half and applied to the other. 'Within-question AUROC' removes the confound that different items have different option counts and difficulty.
- **Scoring** uses only recorded human answers; no LLM judge.
