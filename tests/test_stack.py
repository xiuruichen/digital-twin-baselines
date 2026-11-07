import numpy as np
import pandas as pd

from dteval import features, stack


def test_persona_features_ordinal_onehot_style_and_text():
    import json
    persona = json.dumps([{"BlockName": "b", "Questions": [
        {"QuestionID": "Q1", "QuestionType": "MC", "Options": ["a", "b", "c"], "Answers": {"SelectedByPosition": 3}},
        {"QuestionID": "Q2", "QuestionType": "Matrix", "Columns": ["x", "y", "z", "w", "v"], "Rows": ["r1", "r2"],
         "Answers": {"SelectedByPosition": [1, 5]}},
        {"QuestionID": "Q3", "QuestionType": "TE", "Answers": {"Text": " 5  Cents "}}]}])
    f = features.persona_features(persona)
    assert f["o:Q1"] == 1.0 and f["h:Q1:3"] == 1.0
    assert f["o:Q2#0"] == 0.0 and f["o:Q2#1"] == 1.0
    assert f["s:extreme"] == 1.0 and f["s:mid"] == 0.0
    assert f["t:Q3:5 cents"] == 1.0


def test_paired_diff_detects_real_gap_and_no_gap():
    pid = np.repeat(np.arange(50).astype(str), 10)
    a = pd.DataFrame({"pid": pid, "correct": np.tile([1.0] * 7 + [0.0] * 3, 50)})
    b = pd.DataFrame({"pid": pid, "correct": np.tile([1.0] * 5 + [0.0] * 5, 50)})
    d, (lo, hi) = stack.paired_diff(a, b)
    assert abs(d - 0.2) < 1e-9 and lo > 0.1
    d0, (lo0, hi0) = stack.paired_diff(a, a)
    assert d0 == 0 and lo0 == hi0 == 0


def test_crossfit_blend_puts_weight_on_informative_source():
    rng = np.random.default_rng(0)
    n, k = 1500, 3
    gold = rng.integers(0, k, n)
    informative = np.full((n, stack.K), -50.0)
    noise = np.full((n, stack.K), -50.0)
    informative[:, :k] = rng.normal(size=(n, k))
    informative[np.arange(n), gold] += 2.0       # carries the answer
    noise[:, :k] = rng.normal(size=(n, k))       # carries nothing
    mask = np.zeros((n, stack.K), bool)
    mask[:, :k] = True
    fold = np.arange(n) % 5
    P, w = stack.crossfit([informative, noise], mask, gold, fold, np.zeros(n, dtype=str))
    assert w[0] > 0.5 > abs(w[1])
    assert (P.argmax(1) == gold).mean() > 0.7
    assert np.allclose(P.sum(1), 1.0)
