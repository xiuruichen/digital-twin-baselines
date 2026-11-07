"""LLM-free baselines trained on the other participants: a crowd prior and a persona-feature model.

For every holdout item we learn, from people who are NOT the person being predicted, how wave 1-3
answers relate to the wave-4 answer. Predictions are out-of-fold: person i is predicted by models
that never saw person i (K folds over all 2,058 people, seeded). Inputs are wave 1-3 answers only;
the person's own wave 1-3 answer to the holdout item (the test-retest answer) is never used.

Outputs per (pid, item_id):
  p_crowd  smoothed answer frequencies among the training folds ("what do most people say")
  p_ml     ensemble (logistic regression + boosted trees + extra-trees) on persona features
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import data

CACHE = Path("cache")
EPS = 1e-3


def persona_features(persona_json: str) -> dict[str, float]:
    """Ordinal (position scaled to 0-1) for every MC/Matrix answer, plus one-hot for MC choices
    (needed for nominal questions and for right/wrong cognitive-test options)."""
    f: dict[str, float] = {}
    likert: list[float] = []
    for block in json.loads(persona_json):
        for q in block.get("Questions", []):
            pos = (q.get("Answers") or {}).get("SelectedByPosition")
            qid = q["QuestionID"]
            if q["QuestionType"] == "MC" and isinstance(pos, int):
                n = max(len(q["Options"]), 2)
                f[f"o:{qid}"] = (pos - 1) / (n - 1)
                f[f"h:{qid}:{pos}"] = 1.0
            elif q["QuestionType"] == "TE" and isinstance((q.get("Answers") or {}).get("Text"), str):
                txt = " ".join(q["Answers"]["Text"].lower().split())[:30]
                if txt:  # common free-text answers (e.g. CRT-style numbers) as one-hot flags
                    f[f"t:{qid}:{txt}"] = 1.0
            elif q["QuestionType"] == "Matrix" and isinstance(pos, list):
                n = max(len(q.get("Columns", [])), 2)
                for r, p in enumerate(pos):
                    if isinstance(p, int):
                        f[f"o:{qid}#{r}"] = (p - 1) / (n - 1)
                        likert.append((p - 1) / (n - 1))
    if likert:  # response style: how a person uses rating scales in general
        a = np.asarray(likert)
        f.update({"s:mean": a.mean(), "s:std": a.std(), "s:extreme": float(np.mean((a < 0.1) | (a > 0.9))),
                  "s:mid": float(np.mean(np.abs(a - 0.5) < 0.1))})
    return f


def feature_matrix(people: pd.DataFrame) -> np.ndarray:
    X = pd.DataFrame([persona_features(js) for js in people.wave1_3_persona_json])
    X = X.loc[:, X.notna().mean() > 0.05]
    onehot = [c for c in X.columns if c.startswith(("h:", "t:"))]
    X[onehot] = X[onehot].fillna(0.0)
    X = (X - X.mean()) / X.std().replace(0, 1)  # label-free standardisation
    return X.fillna(0.0).to_numpy(dtype=np.float32)


def _fit_predict(Xtr, ytr, Xte, k: int, C: float) -> np.ndarray:
    """Average of three differently-biased models: ridge logistic regression, shallow boosted trees,
    extra-trees. Each is probability-smoothed, then averaged."""
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    models = [LogisticRegression(C=C, max_iter=300),
              HistGradientBoostingClassifier(max_depth=2, max_iter=40, learning_rate=0.05, min_samples_leaf=40,
                                             l2_regularization=5.0, random_state=0),
              ExtraTreesClassifier(n_estimators=150, min_samples_leaf=8, max_features=0.1, random_state=0)]
    out = np.zeros((len(Xte), k))
    for m in models:
        m.fit(Xtr, ytr)
        P = np.full((len(Xte), k), EPS)
        P[:, m.classes_] = m.predict_proba(Xte)
        out += P / P.sum(1, keepdims=True) / len(models)
    return out


def _one_item(X, idx, gold, k, fold, folds, C):
    out = {}
    for f in range(folds):
        te, tr = fold[idx] == f, fold[idx] != f
        if not te.any():
            continue
        counts = np.bincount(gold[tr], minlength=k) + 1.0
        crowd = np.tile(counts / counts.sum(), (te.sum(), 1))
        ml = crowd
        if tr.sum() >= 40 and len(np.unique(gold[tr])) > 1:
            ml = _fit_predict(X[idx[tr]], gold[tr], X[idx[te]], k, C)
        for j, pi in enumerate(idx[te]):
            out[pi] = (crowd[j], ml[j])
    return out


def out_of_fold(people: pd.DataFrame, C: float = 0.0006, folds: int = 5, seed: int = 0,
                pricing_k: int = 4, matrix_rows: int | None = 4, item_seed: int = 0, n_jobs: int = 3):
    """Return {(pid, item_id): (p_crowd, p_ml)} for every holdout item of every person."""
    from joblib import Parallel, delayed
    key = hashlib.sha256(f"v2|{len(people)}|{C}|{folds}|{seed}|{pricing_k}|{matrix_rows}|{item_seed}".encode()).hexdigest()[:12]
    path = CACHE / f"oof_{key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    X = feature_matrix(people)
    pids = people.pid.astype(str).to_numpy()
    fold = np.random.default_rng(seed).permutation(len(people)) % folds
    rows: dict[str, list[tuple[int, int, int]]] = {}   # item_id -> [(person idx, gold, n_options)]
    for i, row in enumerate(people.itertuples(index=False)):
        for it in data.person_items(row, pricing_k, matrix_rows, item_seed):
            rows.setdefault(it.item_id, []).append((i, it.gold, len(it.options)))
    jobs = {iid: (np.array([r[0] for r in recs]), np.array([r[1] for r in recs]), recs[0][2]) for iid, recs in rows.items()}
    results = Parallel(n_jobs=n_jobs)(delayed(_one_item)(X, idx, g, k, fold, folds, C) for idx, g, k in jobs.values())
    out = {(pids[pi], iid): v for iid, res in zip(jobs, results) for pi, v in res.items()}
    CACHE.mkdir(exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out
