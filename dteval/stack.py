"""Combine the LLM's answer distribution with the crowd prior and the persona-feature model.

Final score per option k:   s_k = sum_j  w_j * log p_j,k      (then softmax)
sources j:  crowd prior, persona-feature model, and the LLM after label-free calibration
            (its log-probs minus the item's average log-prob over the evaluated people, i.e. what is
            *person-specific* about the LLM's answer).
The 2-3 weights are fit by minimising log-loss and are always CROSS-FIT: people are split into folds
by id hash, the weights used for a fold's predictions are fit on the other folds only.
All comparisons are paired bootstrap over people.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import features
from . import metrics as M
from .data import load_people
from .report import md_table, pct

K = 9  # max options


def _pad_log(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    L = np.full((len(rows), K), -50.0)
    mask = np.zeros((len(rows), K), bool)
    for i, p in enumerate(rows):
        L[i, : len(p)] = np.log(np.clip(p, 1e-4, 1))
        mask[i, : len(p)] = True
    return L, mask


def _softmax(S: np.ndarray, mask: np.ndarray) -> np.ndarray:
    S = np.where(mask, S, -1e9)
    S = S - S.max(1, keepdims=True)
    e = np.exp(S) * mask
    return e / e.sum(1, keepdims=True)


def fit_weights(Ls: list[np.ndarray], mask: np.ndarray, gold: np.ndarray, l2: float = 1e-2) -> np.ndarray:
    n = np.arange(len(gold))

    def nll(w):
        p = _softmax(sum(wj * L for wj, L in zip(w, Ls)), mask)
        return -np.log(np.clip(p[n, gold], 1e-9, 1)).mean() + l2 * (w ** 2).sum()

    return minimize(nll, np.ones(len(Ls)) * 0.5, method="L-BFGS-B").x


def crossfit(Ls: list[np.ndarray], mask: np.ndarray, gold: np.ndarray, fold: np.ndarray, group: np.ndarray):
    """Weights are fit separately per question group (binary / multi-option / Likert row), never on the
    fold being predicted."""
    out = np.zeros(mask.shape)
    ws = []
    for g in np.unique(group):
        for f in np.unique(fold):
            tr, te = (fold != f) & (group == g), (fold == f) & (group == g)
            if not te.any():
                continue
            w = fit_weights([L[tr] for L in Ls], mask[tr], gold[tr])
            out[te] = _softmax(sum(wj * L[te] for wj, L in zip(w, Ls)), mask[te])
            ws.append(w)
    return out, np.mean(ws, axis=0)


def build_systems(df: pd.DataFrame, oof: dict, n_folds: int = 5) -> tuple[dict, dict]:
    """Return ({system: probs (N, K)}, {system: mean fitted weights})."""
    keys = list(zip(df.pid, df.item_id))
    crowd, mlp = zip(*[oof[k] for k in keys])
    llm = [np.asarray(p) for p in df.probs]
    Lc, mask = _pad_log(list(crowd))
    Lm, _ = _pad_log(list(mlp))
    Ll, _ = _pad_log(llm)
    # label-free LLM calibration: subtract the item's mean log-prob over evaluated people
    item = df.item_id.to_numpy()
    Llc = Ll.copy()
    for it in np.unique(item):
        m = item == it
        Llc[m] = Ll[m] - np.log(np.clip(np.exp(Ll[m]).mean(0), 1e-4, 1))
    gold = df.gold.to_numpy()
    group = np.where(df.item_id.str.startswith("QID9_"), "pricing", np.where(df.item_id.str.contains("#"), "likert",
                     np.where(df.n_options == 2, "binary", "multi")))
    fold = df.pid.map(lambda s: int(hashlib.sha256(s.encode()).hexdigest(), 16) % n_folds).to_numpy()
    systems = {"LLM (raw)": np.exp(Ll) * mask, "LLM (label-free calibrated)": _softmax(Llc, mask),
               "Crowd prior (other people, mode)": np.exp(Lc) * mask, "Persona-feature model (no LLM)": np.exp(Lm) * mask}
    weights = {}
    for name, Ls in {"Blend: crowd + persona-feature (no LLM)": [Lc, Lm],
                     "Blend: crowd + LLM": [Lc, Llc],
                     "Blend: crowd + persona-feature + LLM": [Lc, Lm, Llc]}.items():
        systems[name], weights[name] = crossfit(Ls, mask, gold, fold, group)
    return systems, weights


def score(df: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    gold = df.gold.to_numpy()
    pred = P.argmax(1)
    return pd.DataFrame({"pid": df.pid.to_numpy(), "correct": (pred == gold).astype(float),
                         "logp": np.log(np.clip(P[np.arange(len(gold)), gold] / P.sum(1), 1e-4, 1))})


def paired_diff(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 0):
    """Mean accuracy difference a-b with a bootstrap-over-people 95% CI."""
    g = pd.DataFrame({"pid": a.pid, "d": a.correct.to_numpy() - b.correct.to_numpy()}).groupby("pid").d.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    diffs = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.percentile(diffs, [2.5, 97.5])


TARGET_ACC, TARGET_LIFT = 0.42, 0.04


def evaluate_run(run: Path, out: Path, C: float = 0.0006) -> dict:
    cfg = json.loads((run / "config.json").read_text())
    df = M.load_predictions(run / "predictions.jsonl")
    people = load_people()
    oof = features.out_of_fold(people, C=C, pricing_k=cfg["pricing_k"], matrix_rows=cfg["matrix_rows"],
                               item_seed=cfg["seed"])
    systems, weights = build_systems(df, oof)
    scored = {k: score(df, P) for k, P in systems.items()}
    base = scored["Crowd prior (other people, mode)"]
    ml_only = scored["Blend: crowd + persona-feature (no LLM)"]
    retest = df.retest_correct.mean()
    chance = float((1 / df.n_options).mean())
    rows, summary = [], {}
    for k, s in scored.items():
        acc = s.correct.mean()
        d, (lo, hi) = paired_diff(s, base)
        d2, (lo2, hi2) = paired_diff(s, ml_only)
        summary[k] = dict(acc=acc, lift_vs_crowd=d, lift_lo=lo, lift_hi=hi, logp=s.logp.mean(),
                          lift_vs_noLLM_blend=d2, lo2=lo2, hi2=hi2)
        rows.append([k, pct(acc), f"{100 * d:+.1f} ({100 * lo:+.1f} to {100 * hi:+.1f})",
                     f"{100 * d2:+.1f} ({100 * lo2:+.1f} to {100 * hi2:+.1f})", f"{s.logp.mean():.3f}",
                     pct(acc / retest), pct((acc - chance) / (retest - chance))])
    head = ["System", "Accuracy", "vs crowd prior, pts (95% CI)", "vs no-LLM blend, pts (95% CI)",
            "Log-lik. of real answer", "Normalized acc.", "Chance-adjusted normalized"]
    best = summary["Blend: crowd + persona-feature + LLM"]
    hit = best["acc"] >= TARGET_ACC and best["lift_lo"] > 0 and best["lift_vs_crowd"] >= TARGET_LIFT
    n_people = df.pid.nunique()
    verdict = (f"Target (accuracy >= {pct(TARGET_ACC)} AND >= {100 * TARGET_LIFT:.0f} pts over the crowd prior with a CI excluding 0): "
               f"{'MET' if hit else 'NOT met'} on {n_people} people / {len(df)} answers. "
               f"Human test-retest {pct(retest)}, chance {pct(chance)}.")
    notes = ("Weights are cross-fit by person; persona-feature models and the crowd prior are out-of-fold over all 2,058 people and use wave 1-3 "
             "answers only. The 'vs no-LLM blend' column is the test of whether the LLM adds individual signal. Fitted weights "
             "(crowd, persona-feature, LLM): " + "; ".join(f"{k.split(': ')[1]} = {np.round(w, 2).tolist()}" for k, w in weights.items()))
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}.md").write_text(f"# Systems comparison: {cfg['model']}\n\n{verdict}\n\n{md_table(head, rows)}\n\n{notes}\n", encoding="utf-8")
    Path(f"{out}.json").write_text(json.dumps({"verdict": verdict, "hit": bool(hit), "summary": summary,
                                              "n_people": n_people, "weights": {k: v.tolist() for k, v in weights.items()}}, indent=1))
    print(verdict); print(md_table(head, rows)); print(notes)
    return {"hit": hit, "summary": summary}


def evaluate_no_llm(n_people: int, offset: int, out: Path, C: float = 0.0006, pricing_k: int = 4,
                    matrix_rows: int | None = 4) -> dict:
    """Crowd prior vs persona-feature ensemble vs their cross-fit blend, on a slice of people, no LLM needed.

    Cheap, so it can use hundreds of people and answer 'how much lift does persona information give over the
    crowd?' with tight intervals. Use people the persona model was never tuned on (offset 0, n 200 = dev + test)."""
    from .data import person_items, select_people
    people = load_people()
    oof = features.out_of_fold(people, C=C, pricing_k=pricing_k, matrix_rows=matrix_rows)
    rows = []
    for r in select_people(people, n_people, 0, offset).itertuples(index=False):
        for it in person_items(r, pricing_k, matrix_rows, 0):
            rows.append({"pid": str(r.pid), "item_id": it.item_id, "gold": it.gold, "n_options": len(it.options),
                         "retest_correct": float(it.retest == it.gold) if it.retest is not None else np.nan})
    df = pd.DataFrame(rows)
    keys = list(zip(df.pid, df.item_id))
    crowd, mlp = zip(*[oof[k] for k in keys])
    Lc, mask = _pad_log(list(crowd))
    Lm, _ = _pad_log(list(mlp))
    gold = df.gold.to_numpy()
    fold = df.pid.map(lambda s: int(hashlib.sha256(s.encode()).hexdigest(), 16) % 5).to_numpy()
    group = np.where(df.item_id.str.startswith("QID9_"), "pricing", np.where(df.item_id.str.contains("#"), "likert",
                     np.where(df.n_options == 2, "binary", "multi")))
    blend, w = crossfit([Lc, Lm], mask, gold, fold, group)
    systems = {"Crowd prior (other people, mode)": np.exp(Lc) * mask, "Persona-feature ensemble (no LLM)": np.exp(Lm) * mask,
               "Blend: crowd + persona-feature": blend}
    scored = {k: score(df, P) for k, P in systems.items()}
    base = scored["Crowd prior (other people, mode)"]
    retest, chance = df.retest_correct.mean(), float((1 / df.n_options).mean())
    table = []
    for k, s in scored.items():
        d, (lo, hi) = paired_diff(s, base)
        table.append([k, pct(s.correct.mean()), f"{100 * d:+.2f} ({100 * lo:+.2f} to {100 * hi:+.2f})", f"{s.logp.mean():.3f}"])
    head = ["System", "Accuracy", "vs crowd prior, pts (95% CI)", "Log-lik. of real answer"]
    text = (f"# No-LLM systems on {df.pid.nunique()} people / {len(df)} answers (people {offset}-{offset + n_people - 1} of the seeded order)\n\n"
            f"Human test-retest accuracy {pct(retest)}, chance {pct(chance)}.\n\n{md_table(head, table)}\n\n"
            f"Blend weights (crowd, persona-feature), mean over folds/groups: {np.round(w, 2).tolist()}\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}.md").write_text(text, encoding="utf-8")
    print(text)
    return {"n_people": df.pid.nunique(), "n_answers": len(df)}
