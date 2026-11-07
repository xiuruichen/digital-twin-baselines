import math

import numpy as np
import pandas as pd

from dteval import metrics as M
from dteval.data import Item
from dteval.models import letter_probs
from dteval.prompts import build_prompt

ITEM = Item("Q1", "Q1", "b", "Pick one", None, ("red", "green", "blue"), 1, 1)


def test_letter_probs_merges_space_variants_and_renormalises():
    top = [(" A", math.log(0.2)), ("A", math.log(0.1)), (" B", math.log(0.3)), ("\n", math.log(0.4))]
    probs, valid = letter_probs(top, "ABC")
    assert abs(valid - 0.6) < 1e-9
    assert np.allclose(probs, [0.5, 0.5, 0.0])


def test_letter_probs_no_valid_mass_is_uniform():
    probs, valid = letter_probs([("x", -1.0)], "AB")
    assert valid == 0.0 and probs == [0.5, 0.5]


def test_shuffle_is_deterministic_and_changes_order():
    p0, o0 = build_prompt("persona", ITEM, "1")
    assert o0 == [0, 1, 2]
    orders = {tuple(build_prompt("persona", ITEM, str(i), shuffle_seed=1)[1]) for i in range(30)}
    assert len(orders) > 1
    assert build_prompt("persona", ITEM, "7", 1) == build_prompt("persona", ITEM, "7", 1)


def test_auroc_known_values():
    assert M.auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 1.0
    assert M.auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([1, 1, 0, 0])) == 0.0
    assert M.auroc(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5


def test_tv_distance():
    assert M.tv(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0
    assert M.tv(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0


def _fake_df(n_people=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_people):
        for i in range(6):
            gold, conf = int(rng.integers(0, 3)), rng.uniform(0.34, 1.0)
            right = rng.uniform() < conf  # confidence is informative by construction
            pick = gold if right else (gold + 1) % 3
            probs = np.full(3, (1 - conf) / 2)
            probs[pick] = conf
            rows.append({"pid": str(p), "item_id": f"Q{i}", "qid": f"Q{i}", "block": "b", "n_options": 3,
                         "gold": gold, "retest": gold if rng.uniform() < 0.7 else (gold + 1) % 3,
                         "probs": probs.tolist(), "valid_mass": 1.0})
    return pd.DataFrame(rows)


def test_pipeline_metrics_on_synthetic_predictions(tmp_path):
    f = tmp_path / "p.jsonl"
    _fake_df().to_json(f, orient="records", lines=True)
    df = M.load_predictions(f)
    ind = M.individual(df, n_boot=50)
    assert 0 < ind["normalized_accuracy"] < 2 and ind["n_people"] == 40
    pop, per_item = M.population(df)
    assert len(per_item) == 6 and 0 <= pop["tv_soft"] <= 1
    c = M.confidence(df)
    assert c["auroc_pooled_all"] > 0.6  # informative confidence is detected
    assert c["error_ratio_low_vs_high"] > 1
