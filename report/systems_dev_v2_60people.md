# Systems comparison: bonsai2-9b

Target (accuracy >= 42.0% AND >= 4 pts over the crowd prior with a CI excluding 0): NOT met on 60 people / 2100 answers. Human test-retest 58.1%, chance 29.2%.

| System | Accuracy | vs crowd prior, pts (95% CI) | vs no-LLM blend, pts (95% CI) | Log-lik. of real answer | Normalized acc. | Chance-adjusted normalized |
|---|---|---|---|---|---|---|
| LLM (raw) | 39.0% | -8.9 (-11.8 to -6.0) | -12.0 (-15.3 to -8.7) | -1.349 | 67.1% | 33.8% |
| LLM (label-free calibrated) | 31.4% | -16.5 (-19.4 to -13.4) | -19.6 (-22.6 to -16.7) | -1.374 | 54.1% | 7.6% |
| Crowd prior (other people, mode) | 47.9% | +0.0 (+0.0 to +0.0) | -3.1 (-5.0 to -1.4) | -1.167 | 82.4% | 64.6% |
| Persona-feature model (no LLM) | 50.8% | +2.9 (+1.0 to +4.8) | -0.2 (-1.0 to +0.6) | -1.108 | 87.4% | 74.6% |
| Blend: crowd + persona-feature (no LLM) | 51.0% | +3.1 (+1.4 to +5.0) | +0.0 (+0.0 to +0.0) | -1.108 | 87.7% | 75.3% |
| Blend: crowd + LLM | 48.6% | +0.7 (-0.2 to +1.7) | -2.4 (-4.6 to -0.2) | -1.164 | 83.6% | 67.0% |
| Blend: crowd + persona-feature + LLM | 51.2% | +3.3 (+1.6 to +5.2) | +0.2 (-0.3 to +0.8) | -1.105 | 88.1% | 76.1% |

Weights are cross-fit by person; persona-feature models and the crowd prior are out-of-fold over all 2,058 people and use wave 1-3 answers only. The 'vs no-LLM blend' column is the test of whether the LLM adds individual signal. Fitted weights (crowd, persona-feature, LLM): crowd + persona-feature (no LLM) = [0.11, 0.95]; crowd + LLM = [0.88, 0.48]; crowd + persona-feature + LLM = [0.12, 0.94, 0.37]
