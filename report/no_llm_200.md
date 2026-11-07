# No-LLM systems on 200 people / 7000 answers (people 0-199 of the seeded order)

Human test-retest accuracy 58.3%, chance 29.2%.

| System | Accuracy | vs crowd prior, pts (95% CI) | Log-lik. of real answer |
|---|---|---|---|
| Crowd prior (other people, mode) | 47.3% | +0.00 (+0.00 to +0.00) | -1.171 |
| Persona-feature ensemble (no LLM) | 50.0% | +2.70 (+1.73 to +3.69) | -1.113 |
| Blend: crowd + persona-feature | 50.0% | +2.70 (+1.69 to +3.73) | -1.113 |

Blend weights (crowd, persona-feature), mean over folds/groups: [0.05, 0.91]
