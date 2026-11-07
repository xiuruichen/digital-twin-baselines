# Digital-twin evaluation report

Baseline on Twin-2K-500 holdout (wave 4) items.

- **bonsai2-9b** reaches 70.7% of human test-retest consistency (95% CI 61.1%-83.4%, 10 people); on held-out people, answers in the low-confidence half are 1.57x as likely to be wrong as high-confidence ones (within-question AUROC 0.59).
- **bonsai2-4b** reaches 42.0% of human test-retest consistency (95% CI 32.8%-53.5%, 10 people); on held-out people, answers in the low-confidence half are 1.50x as likely to be wrong as high-confidence ones (within-question AUROC 0.49).
- **qwen3.5-4b-q4** reaches 58.0% of human test-retest consistency (95% CI 49.2%-70.2%, 10 people); on held-out people, answers in the low-confidence half are 1.44x as likely to be wrong as high-confidence ones (within-question AUROC 0.53).

Config: bonsai2-9b: 10 people, persona 4000 chars, shuffle_seed=None; bonsai2-4b: 10 people, persona 4000 chars, shuffle_seed=None; qwen3.5-4b-q4: 10 people, persona 4000 chars, shuffle_seed=None

## Individual-level accuracy

| Model | People | Answers | Accuracy | Human retest acc. | Normalized acc. (95% CI) | Crowd-mode baseline | Chance | Log-lik. of real answer (uniform guess) | Valid-letter mass |
|---|---|---|---|---|---|---|---|---|---|
| bonsai2-9b | 10 | 350 | 36.6% | 51.7% | 70.7% (61.1%-83.4%) | 30.8% | 29.2% | -1.353 (-1.381) | 0.920 |
| bonsai2-4b | 10 | 350 | 21.7% | 51.7% | 42.0% (32.8%-53.5%) | 30.8% | 29.2% | -1.683 (-1.381) | 0.697 |
| qwen3.5-4b-q4 | 10 | 350 | 30.0% | 51.7% | 58.0% (49.2%-70.2%) | 30.8% | 29.2% | -1.457 (-1.381) | 0.876 |

## Population-level distribution distance

| Model | Questions | TV soft (mean probs) | TV hard (argmax) | TV uniform guess | TV human retest (floor) |
|---|---|---|---|---|---|
| bonsai2-9b | 55 | 0.384 | 0.578 | 0.384 | 0.304 |
| bonsai2-4b | 55 | 0.504 | 0.775 | 0.384 | 0.304 |
| qwen3.5-4b-q4 | 55 | 0.426 | 0.689 | 0.384 | 0.304 |

## Confidence: does it predict errors?

| Model | Confidence signal | AUROC pooled | AUROC within-question | AUROC pooled (test half) | Acc. low-conf (test) | Acc. high-conf (test) | Error ratio low/high | ECE (test) |
|---|---|---|---|---|---|---|---|---|
| bonsai2-9b | top-option prob. | 0.711 | 0.591 | 0.698 | 20.7% | 49.4% | 1.57x | 0.091 |
| bonsai2-9b | top-2 margin | 0.677 | 0.560 | 0.635 | 23.9% | 45.8% | 1.40x | 0.210 |
| bonsai2-4b | top-option prob. | 0.845 | 0.493 | 0.803 | 5.8% | 37.1% | 1.50x | 0.343 |
| bonsai2-4b | top-2 margin | 0.839 | 0.515 | 0.796 | 4.7% | 38.2% | 1.54x | 0.147 |
| qwen3.5-4b-q4 | top-option prob. | 0.784 | 0.534 | 0.725 | 15.3% | 41.1% | 1.44x | 0.148 |
| qwen3.5-4b-q4 | top-2 margin | 0.689 | 0.533 | 0.606 | 22.7% | 34.5% | 1.18x | 0.154 |

## Per-question population TV

| QID | bonsai2-9b: TV soft / human retest | bonsai2-4b: TV soft / human retest | qwen3.5-4b-q4: TV soft / human retest |
|---|---|---|---|
| QID157 | 0.37 / 0.33 (n=6) | 0.55 / 0.33 (n=6) | 0.44 / 0.33 (n=6) |
| QID158 | 0.68 / 0.75 (n=4) | 0.64 / 0.75 (n=4) | 0.67 / 0.75 (n=4) |
| QID159 | 0.61 / 0.58 (n=4) | 0.68 / 0.58 (n=4) | 0.61 / 0.58 (n=4) |
| QID160 | 0.42 / 0.28 (n=6) | 0.62 / 0.28 (n=6) | 0.46 / 0.28 (n=6) |
| QID161 | 0.36 / 0.14 (n=7) | 0.59 / 0.14 (n=7) | 0.43 / 0.14 (n=7) |
| QID162 | 0.55 / 0.33 (n=3) | 0.57 / 0.33 (n=3) | 0.55 / 0.33 (n=3) |
| QID163 | 0.18 / 0.20 (n=5) | 0.09 / 0.20 (n=5) | 0.06 / 0.20 (n=5) |
| QID165 | 0.23 / 0.20 (n=5) | 0.49 / 0.20 (n=5) | 0.34 / 0.20 (n=5) |
| QID167 | 0.31 / 0.00 (n=5) | 0.08 / 0.00 (n=5) | 0.28 / 0.00 (n=5) |
| QID169 | 0.08 / 0.20 (n=5) | 0.34 / 0.20 (n=5) | 0.15 / 0.20 (n=5) |
| QID171 | 0.94 / 0.00 (n=1) | 0.96 / 0.00 (n=1) | 0.87 / 0.00 (n=1) |
| QID172 | 0.26 / 0.38 (n=8) | 0.37 / 0.38 (n=8) | 0.35 / 0.38 (n=8) |
| QID173 | 0.66 / 1.00 (n=1) | 0.80 / 1.00 (n=1) | 0.85 / 1.00 (n=1) |
| QID174 | 0.62 / 1.00 (n=1) | 0.77 / 1.00 (n=1) | 0.80 / 1.00 (n=1) |
| QID175 | 0.41 / 0.00 (n=8) | 0.42 / 0.00 (n=8) | 0.45 / 0.00 (n=8) |
| QID176 | 0.78 / 0.00 (n=1) | 0.77 / 0.00 (n=1) | 0.80 / 0.00 (n=1) |
| QID177 | 0.61 / 1.00 (n=1) | 0.79 / 1.00 (n=1) | 0.83 / 1.00 (n=1) |
| QID178 | 0.27 / 0.12 (n=8) | 0.49 / 0.12 (n=8) | 0.49 / 0.12 (n=8) |
| QID179 | 0.86 / 0.00 (n=1) | 0.85 / 0.00 (n=1) | 0.90 / 0.00 (n=1) |
| QID183 | 0.08 / 0.29 (n=7) | 0.16 / 0.29 (n=7) | 0.05 / 0.29 (n=7) |
| QID184 | 0.68 / 0.67 (n=3) | 0.73 / 0.67 (n=3) | 0.52 / 0.67 (n=3) |
| QID192 | 0.17 / 0.33 (n=3) | 0.56 / 0.33 (n=3) | 0.26 / 0.33 (n=3) |
| QID193 | 0.37 / 0.29 (n=7) | 0.68 / 0.29 (n=7) | 0.43 / 0.29 (n=7) |
| QID194 | 0.85 / 0.50 (n=4) | 0.82 / 0.50 (n=4) | 0.75 / 0.50 (n=4) |
| QID195 | 0.45 / 0.33 (n=6) | 0.49 / 0.33 (n=6) | 0.41 / 0.33 (n=6) |
| QID196 | 0.15 / 0.10 (n=10) | 0.27 / 0.10 (n=10) | 0.24 / 0.10 (n=10) |
| QID198 | 0.29 / 0.25 (n=2) | 0.25 / 0.25 (n=2) | 0.24 / 0.25 (n=2) |
| QID203 | 0.17 / 0.09 (n=8) | 0.31 / 0.09 (n=8) | 0.31 / 0.09 (n=8) |
| QID287 | 0.28 / 0.25 (n=10) | 0.49 / 0.25 (n=10) | 0.29 / 0.25 (n=10) |
| QID288 | 0.43 / 0.38 (n=10) | 0.59 / 0.38 (n=10) | 0.48 / 0.38 (n=10) |
| QID289 | 0.45 / 0.43 (n=10) | 0.58 / 0.43 (n=10) | 0.53 / 0.43 (n=10) |
| QID291 | 0.21 / 0.10 (n=10) | 0.44 / 0.10 (n=10) | 0.23 / 0.10 (n=10) |
| QID9_1 | 0.11 / 0.10 (n=10) | 0.35 / 0.10 (n=10) | 0.18 / 0.10 (n=10) |
| QID9_13 | 0.13 / 0.00 (n=10) | 0.44 / 0.00 (n=10) | 0.25 / 0.00 (n=10) |
| QID9_22 | 0.06 / 0.10 (n=10) | 0.34 / 0.10 (n=10) | 0.16 / 0.10 (n=10) |
| QID9_28 | 0.11 / 0.10 (n=10) | 0.14 / 0.10 (n=10) | 0.01 / 0.10 (n=10) |

## Notes

- **Scope:** categorical holdout items only (multiple choice + Likert matrix rows) from the wave-4 heuristics-and-biases / behavioral-economics block; a fixed seeded subset of rows/pricing items is used to bound compute. Sliders and free-text numeric items are not scored in this phase.
- **Persona:** compact 'question: answer' rendering of wave 1-3 answers (demographics, personality, economic preferences) truncated to a character budget; the holdout items are never in the persona.
- **Normalized accuracy** = model accuracy / the same people's wave 1-3 vs wave 4 agreement, on the same items. Exact-match on ordinal scales is harsh for both, which is why the ratio is the headline.
- **Individual and population metrics are separate on purpose.** A model can match the crowd (low TV) yet be no better than the crowd-mode baseline for individuals.
- **Confidence** is evaluated out of sample: people are split in half by id hash; bucket edges and the low/high threshold are fit on one half and applied to the other. 'Within-question AUROC' removes the confound that different items have different option counts and difficulty.
- **Scoring** uses only recorded human answers; no LLM judge.
