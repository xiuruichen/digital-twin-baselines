# Systems comparison: bonsai2-9b

Target (accuracy >= 42.0% AND >= 4 pts over the crowd prior with a CI excluding 0): NOT met on 34 people / 1190 answers. Human test-retest 59.0%, chance 29.2%.

| System | Accuracy | vs crowd prior, pts (95% CI) | vs no-LLM blend, pts (95% CI) | Log-lik. of real answer | Normalized acc. | Chance-adjusted normalized |
|---|---|---|---|---|---|---|
| LLM (raw) | 37.8% | -8.7 (-12.1 to -5.0) | -11.6 (-15.5 to -7.5) | -1.368 | 64.1% | 28.8% |
| LLM (label-free calibrated) | 32.0% | -14.5 (-18.6 to -10.1) | -17.4 (-21.3 to -13.3) | -1.377 | 54.3% | 9.3% |
| Crowd prior (other people, mode) | 46.5% | +0.0 (+0.0 to +0.0) | -2.9 (-5.7 to -0.2) | -1.188 | 78.8% | 57.9% |
| Persona-feature model (no LLM) | 48.8% | +2.4 (-0.3 to +4.9) | -0.6 (-1.6 to +0.3) | -1.128 | 82.8% | 65.8% |
| Blend: crowd + persona-feature (no LLM) | 49.4% | +2.9 (+0.2 to +5.7) | +0.0 (+0.0 to +0.0) | -1.131 | 83.8% | 67.8% |
| Blend: crowd + LLM | 47.5% | +1.0 (-0.3 to +2.3) | -1.9 (-5.1 to +1.2) | -1.186 | 80.5% | 61.3% |
| Blend: crowd + persona-feature + LLM | 49.6% | +3.1 (+0.4 to +5.8) | +0.2 (-0.6 to +0.9) | -1.126 | 84.0% | 68.4% |

Weights are cross-fit by person; persona-feature models and the crowd prior are out-of-fold over all 2,058 people and use wave 1-3 answers only. The 'vs no-LLM blend' column is the test of whether the LLM adds individual signal. Fitted weights (crowd, persona-feature, LLM): crowd + persona-feature (no LLM) = [0.04, 1.03]; crowd + LLM = [0.84, 0.46]; crowd + persona-feature + LLM = [0.05, 1.02, 0.36]
