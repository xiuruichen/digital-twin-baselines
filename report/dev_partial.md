# Systems comparison: bonsai2-9b

Target (accuracy >= 42.0% AND >= 4 pts over the crowd prior with a CI excluding 0): NOT met on 42 people / 1470 answers. Human test-retest 58.7%, chance 29.2%.

| System | Accuracy | vs crowd prior, pts (95% CI) | vs no-LLM blend, pts (95% CI) | Log-lik. of real answer | Normalized acc. | Chance-adjusted normalized |
|---|---|---|---|---|---|---|
| LLM (raw) | 33.5% | -13.8 (-17.9 to -9.8) | -15.2 (-19.0 to -11.3) | -1.373 | 57.1% | 14.6% |
| LLM (label-free calibrated) | 34.3% | -13.1 (-16.5 to -9.8) | -14.4 (-17.7 to -11.2) | -1.371 | 58.4% | 17.1% |
| Crowd prior (other people, mode) | 47.3% | +0.0 (+0.0 to +0.0) | -1.4 (-3.6 to +0.8) | -1.173 | 80.6% | 61.4% |
| Persona-feature model (no LLM) | 49.2% | +1.8 (-0.5 to +4.4) | +0.5 (-0.4 to +1.4) | -1.123 | 83.8% | 67.7% |
| Blend: crowd + persona-feature (no LLM) | 48.7% | +1.4 (-0.8 to +3.6) | +0.0 (+0.0 to +0.0) | -1.128 | 83.0% | 66.1% |
| Blend: crowd + LLM | 47.7% | +0.3 (-1.0 to +1.6) | -1.0 (-3.4 to +1.4) | -1.170 | 81.2% | 62.6% |
| Blend: crowd + persona-feature + LLM | 49.4% | +2.0 (-0.1 to +4.1) | +0.7 (-0.1 to +1.5) | -1.124 | 84.1% | 68.4% |

Weights are cross-fit by person; persona-feature models and the crowd prior are out-of-fold over all 2,058 people and use wave 1-3 answers only. The 'vs no-LLM blend' column is the test of whether the LLM adds individual signal. Fitted weights (crowd, persona-feature, LLM): crowd + persona-feature (no LLM) = [0.12, 0.89]; crowd + LLM = [0.94, 0.54]; crowd + persona-feature + LLM = [0.14, 0.88, 0.4]
