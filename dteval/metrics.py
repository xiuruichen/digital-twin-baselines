"""Metrics. Individual, population and confidence results are computed separately and never blended.

Input: a DataFrame of predictions (one row per person x item) with columns
pid, item_id, qid, n_options, gold, retest (may be NaN), probs (list), valid_mass.
All scoring is against recorded human answers only.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def load_predictions(path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True, dtype={"pid": str})
    p = np.stack([np.pad(np.asarray(x), (0, 9 - len(x))) for x in df.probs])
    srt = np.sort(p, axis=1)[:, ::-1]
    df["pred"] = p.argmax(1)
    df["correct"] = (df.pred == df.gold).astype(float)
    df["conf_top"] = srt[:, 0]
    df["conf_margin"] = srt[:, 0] - srt[:, 1]
    gold_p = p[np.arange(len(df)), df.gold.to_numpy()]
    df["gold_logp"] = np.log(np.clip(gold_p, 1e-4, 1.0))  # log-likelihood of the real answer (less noisy than accuracy)
    df["retest_correct"] = np.where(df.retest.notna(), (df.retest == df.gold).astype(float), np.nan)
    df["half"] = df.pid.map(lambda s: "calib" if int(hashlib.sha256(s.encode()).hexdigest(), 16) % 2 == 0 else "test")
    return df


# ---------- individual ----------
def individual(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """Accuracy, human test-retest accuracy and their ratio on items that have a retest answer.

    Also a leave-one-person-out 'predict the crowd's modal answer' baseline: if the model does
    not beat it, the persona is adding nothing beyond consensus.
    """
    d = df[df.retest.notna()].copy()
    d["modal_correct"] = _modal_baseline(d)
    by_person = d.groupby("pid")[["correct", "retest_correct", "modal_correct"]].mean()
    rng = np.random.default_rng(seed)
    pids = by_person.index.to_numpy()
    ratios = []
    for _ in range(n_boot):
        s = by_person.loc[rng.choice(pids, len(pids))]
        ratios.append(s.correct.mean() / s.retest_correct.mean())
    m = by_person.mean()
    return {"n_people": len(by_person), "n_answers": len(d), "accuracy": m.correct,
            "retest_accuracy": m.retest_correct, "modal_baseline_accuracy": m.modal_correct,
            "chance_accuracy": float((1 / d.n_options).mean()),
            "normalized_accuracy": m.correct / m.retest_correct,
            "normalized_ci": tuple(np.percentile(ratios, [2.5, 97.5])),
            "mean_gold_logp": float(d.gold_logp.mean()),
            "uniform_gold_logp": float(-np.log(d.n_options).mean()),
            "valid_mass": float(df.valid_mass.mean())}


def _modal_baseline(d: pd.DataFrame) -> pd.Series:
    """Per item, leave one person out, predict the most common remaining human answer."""
    out = pd.Series(np.nan, index=d.index)
    for _, g in d.groupby("item_id"):
        vals = g.gold.to_numpy()
        for j, idx in enumerate(g.index):
            rest = np.delete(vals, j)
            out[idx] = float(np.bincount(rest).argmax() == vals[j]) if len(rest) else np.nan
    return out


# ---------- population ----------
def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def population(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Total-variation distance between simulated and real answer distributions, per item.

    sim_soft = mean of the model's probability vectors; sim_hard = distribution of its top picks.
    retest = TV between the same people's wave 1-3 and wave 4 answers (human noise floor).
    TV is biased upward when n is small; compare models on the same people, not across sample sizes.
    """
    rows = []
    for iid, g in df.groupby("item_id"):
        k = int(g.n_options.iloc[0])
        real = np.bincount(g.gold, minlength=k) / len(g)
        soft = np.stack([np.asarray(x) for x in g.probs]).mean(0)
        hard = np.bincount(g.pred, minlength=k) / len(g)
        r = g[g.retest.notna()]
        retest_tv = (tv(np.bincount(r.retest.astype(int), minlength=k) / len(r),
                        np.bincount(r.gold, minlength=k) / len(r)) if len(r) else np.nan)
        rows.append({"item_id": iid, "qid": g.qid.iloc[0], "n": len(g), "tv_soft": tv(real, soft),
                     "tv_hard": tv(real, hard), "tv_uniform": tv(real, np.full(k, 1 / k)),
                     "tv_retest": retest_tv})
    per_item = pd.DataFrame(rows)
    summary = per_item[["tv_soft", "tv_hard", "tv_uniform", "tv_retest"]].mean().to_dict()
    summary["n_items"] = len(per_item)
    return summary, per_item


# ---------- confidence ----------
def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """P(score of a random correct answer > score of a random wrong one); ties count half."""
    pos, neg = score[positive == 1], score[positive == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def within_item_auroc(df: pd.DataFrame, col: str) -> float:
    """Mean per-item AUROC. Pooled AUROC is confounded by item difficulty/option count; this is not."""
    vals = [auroc(g[col].to_numpy(), g.correct.to_numpy()) for _, g in df.groupby("item_id")]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def calibration_bins(conf: np.ndarray, correct: np.ndarray, edges: np.ndarray) -> pd.DataFrame:
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        rows.append({"bin": b, "n": int(m.sum()), "mean_conf": conf[m].mean() if m.any() else np.nan,
                     "accuracy": correct[m].mean() if m.any() else np.nan})
    return pd.DataFrame(rows)


def confidence(df: pd.DataFrame, col: str = "conf_top", n_bins: int = 5) -> dict:
    """Out-of-sample confidence evaluation.

    People are split into two halves by a hash of their id. Bucket edges (quantiles) and the
    low/high threshold (median) are fit on the 'calib' half and only APPLIED to the 'test' half.
    AUROC has no fitted parameters, so it is also reported on all data.
    """
    cal, test = df[df.half == "calib"], df[df.half == "test"]
    edges = np.quantile(cal[col], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    thr = float(np.median(cal[col]))
    low, high = test[test[col] <= thr], test[test[col] > thr]
    err_low, err_high = 1 - low.correct.mean(), 1 - high.correct.mean()
    bins = calibration_bins(test[col].to_numpy(), test.correct.to_numpy(), edges)
    ece = float((bins.n * (bins.mean_conf - bins.accuracy).abs()).sum() / bins.n.sum())
    return {"auroc_pooled_all": auroc(df[col].to_numpy(), df.correct.to_numpy()),
            "auroc_pooled_test": auroc(test[col].to_numpy(), test.correct.to_numpy()),
            "auroc_within_item_all": within_item_auroc(df, col),
            "threshold": thr, "n_test": len(test), "n_low": len(low), "n_high": len(high),
            "acc_low": 1 - err_low, "acc_high": 1 - err_high,
            "error_ratio_low_vs_high": err_low / err_high if err_high > 0 else float("nan"),
            "ece_test": ece, "bins": bins}
