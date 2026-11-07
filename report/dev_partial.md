# Systems comparison: bonsai2-9b

Target (accuracy >= 42.0% AND >= 4 pts over the crowd prior with a CI excluding 0): NOT met on 54 people / 1890 answers. Human test-retest 58.5%, chance 29.2%.

| System | Accuracy | vs crowd prior, pts (95% CI) | vs no-LLM blend, pts (95% CI) | Log-lik. of real answer | Normalized acc. | Chance-adjusted normalized |
|---|---|---|---|---|---|---|
| LLM (raw) | 34.2% | -13.9 (-17.6 to -9.9) | -16.9 (-20.8 to -12.9) | -1.364 | 58.5% | 16.9% |
| LLM (label-free calibrated) | 34.1% | -13.9 (-17.1 to -10.8) | -17.0 (-20.6 to -13.5) | -1.373 | 58.4% | 16.7% |
| Crowd prior (other people, mode) | 48.0% | +0.0 (+0.0 to +0.0) | -3.1 (-5.0 to -1.2) | -1.164 | 82.2% | 64.3% |
| Persona-feature model (no LLM) | 50.9% | +2.9 (+0.8 to +4.9) | -0.2 (-1.1 to +0.7) | -1.102 | 87.1% | 74.1% |
| Blend: crowd + persona-feature (no LLM) | 51.1% | +3.1 (+1.2 to +5.0) | +0.0 (+0.0 to +0.0) | -1.103 | 87.4% | 74.8% |
| Blend: crowd + LLM | 48.9% | +0.8 (-0.4 to +2.0) | -2.2 (-4.6 to +0.1) | -1.159 | 83.6% | 67.2% |
| Blend: crowd + persona-feature + LLM | 51.3% | +3.2 (+1.3 to +5.2) | +0.2 (-0.6 to +0.9) | -1.097 | 87.7% | 75.4% |

Weights are cross-fit by person; persona-feature models and the crowd prior are out-of-fold over all 2,058 people and use wave 1-3 answers only. The 'vs no-LLM blend' column is the test of whether the LLM adds individual signal. Fitted weights (crowd, persona-feature, LLM): crowd + persona-feature (no LLM) = [0.11, 0.95]; crowd + LLM = [0.89, 0.5]; crowd + persona-feature + LLM = [0.13, 0.93, 0.41]
