"""Build report.html (self-contained, embedded chart) and report.md from one or more run dirs."""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import metrics as M  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def evaluate(run: Path, n_boot: int) -> dict:
    cfg = json.loads((run / "config.json").read_text())
    df = M.load_predictions(run / "predictions.jsonl")
    pop, per_item = M.population(df)
    return {"name": cfg["model"], "cfg": cfg, "df": df, "ind": M.individual(df, n_boot),
            "pop": pop, "per_item": per_item,
            "conf": {"conf_top": M.confidence(df, "conf_top"),
                     "conf_margin": M.confidence(df, "conf_margin")}}


def headline(r: dict) -> str:
    i, c = r["ind"], r["conf"]["conf_top"]
    lo, hi = i["normalized_ci"]
    return (f"**{r['name']}** reaches {pct(i['normalized_accuracy'])} of human test-retest consistency "
            f"(95% CI {pct(lo)}-{pct(hi)}, {i['n_people']} people); on held-out people, answers in the "
            f"low-confidence half are {c['error_ratio_low_vs_high']:.2f}x as likely to be wrong as "
            f"high-confidence ones (within-question AUROC {c['auroc_within_item_all']:.2f}).")


def tables(results: list[dict]) -> dict[str, tuple[list[str], list[list[str]]]]:
    ind = (["Model", "People", "Answers", "Accuracy", "Human retest acc.", "Normalized acc. (95% CI)",
            "Crowd-mode baseline", "Chance", "Log-lik. of real answer (uniform guess)", "Valid-letter mass"],
           [[r["name"], r["ind"]["n_people"], r["ind"]["n_answers"], pct(r["ind"]["accuracy"]),
             pct(r["ind"]["retest_accuracy"]),
             f"{pct(r['ind']['normalized_accuracy'])} ({pct(r['ind']['normalized_ci'][0])}-"
             f"{pct(r['ind']['normalized_ci'][1])})", pct(r["ind"]["modal_baseline_accuracy"]),
             pct(r["ind"]["chance_accuracy"]),
             f"{r['ind']['mean_gold_logp']:.3f} ({r['ind']['uniform_gold_logp']:.3f})", f"{r['ind']['valid_mass']:.3f}"] for r in results])
    pop = (["Model", "Questions", "TV soft (mean probs)", "TV hard (argmax)", "TV uniform guess",
            "TV human retest (floor)"],
           [[r["name"], r["pop"]["n_items"], f"{r['pop']['tv_soft']:.3f}", f"{r['pop']['tv_hard']:.3f}",
             f"{r['pop']['tv_uniform']:.3f}", f"{r['pop']['tv_retest']:.3f}"] for r in results])
    conf_rows = []
    for r in results:
        for col, label in [("conf_top", "top-option prob."), ("conf_margin", "top-2 margin")]:
            c = r["conf"][col]
            conf_rows.append([r["name"], label, f"{c['auroc_pooled_all']:.3f}", f"{c['auroc_within_item_all']:.3f}",
                              f"{c['auroc_pooled_test']:.3f}", pct(c["acc_low"]), pct(c["acc_high"]),
                              f"{c['error_ratio_low_vs_high']:.2f}x", f"{c['ece_test']:.3f}"])
    conf = (["Model", "Confidence signal", "AUROC pooled", "AUROC within-question", "AUROC pooled (test half)",
             "Acc. low-conf (test)", "Acc. high-conf (test)", "Error ratio low/high", "ECE (test)"], conf_rows)
    qids = sorted({q for r in results for q in r["per_item"].qid}, key=lambda s: (len(s), s))
    q_rows = []
    for q in qids:
        row = [q]
        for r in results:
            g = r["per_item"][r["per_item"].qid == q]
            row.append("-" if g.empty else f"{g.tv_soft.mean():.2f} / {g.tv_retest.mean():.2f} (n={int(g.n.min())})")
        q_rows.append(row)
    per_q = (["QID"] + [f"{r['name']}: TV soft / human retest" for r in results], q_rows)
    return {"Individual-level accuracy": ind, "Population-level distribution distance": pop,
            "Confidence: does it predict errors?": conf, "Per-question population TV": per_q}


def chart(results: list[dict]) -> str:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=SURFACE)
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
    a.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (4, 3)), label="perfect calibration")
    for k, r in enumerate(results):
        bins = r["conf"]["conf_top"]["bins"].dropna()
        a.plot(bins.mean_conf, bins.accuracy, "-o", color=SERIES[k % 4], lw=2, ms=7,
               markeredgecolor=SURFACE, markeredgewidth=1.5, label=r["name"])
        b.plot(range(1, len(bins) + 1), bins.accuracy, "-o", color=SERIES[k % 4], lw=2, ms=7,
               markeredgecolor=SURFACE, markeredgewidth=1.5, label=r["name"])
        b.axhline(r["ind"]["retest_accuracy"], color=SERIES[k % 4], lw=1, ls=":")
    a.set(xlabel="mean top-option probability (held-out people, equal-count bins)",
          ylabel="accuracy", xlim=(0, 1.02), ylim=(0, 1.02), title="Calibration (reliability) plot")
    b.set(xlabel="confidence bucket (1 = least confident); edges fit on other half of people",
          ylabel="accuracy", xticks=range(1, 6), ylim=(0, 1.02),
          title="Accuracy by confidence bucket")
    for ax in (a, b):
        ax.title.set_color(INK); ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
        ax.xaxis.label.set_size(9); ax.yaxis.label.set_size(9)
    a.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    b.text(0.99, 0.02, "dotted lines: each model's human test-retest accuracy", transform=b.transAxes,
           ha="right", fontsize=8, color=MUTED)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def md_table(head: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def html_table(head: list[str], rows: list[list]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class=scroll><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"


NOTES = """- **Scope:** categorical holdout items only (multiple choice + Likert matrix rows) from the wave-4 \
heuristics-and-biases / behavioral-economics block; a fixed seeded subset of rows/pricing items is used to bound compute. \
Sliders and free-text numeric items are not scored in this phase.
- **Persona:** compact 'question: answer' rendering of wave 1-3 answers (demographics, personality, economic preferences) \
truncated to a character budget; the holdout items are never in the persona.
- **Normalized accuracy** = model accuracy / the same people's wave 1-3 vs wave 4 agreement, on the same items. \
Exact-match on ordinal scales is harsh for both, which is why the ratio is the headline.
- **Individual and population metrics are separate on purpose.** A model can match the crowd (low TV) yet be no better than \
the crowd-mode baseline for individuals.
- **Confidence** is evaluated out of sample: people are split in half by id hash; bucket edges and the low/high threshold \
are fit on one half and applied to the other. 'Within-question AUROC' removes the confound that different items have \
different option counts and difficulty.
- **Scoring** uses only recorded human answers; no LLM judge."""

CSS = """:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e6e5e1}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#383835}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 48px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px}
p.sub{color:var(--muted);margin:0 0 16px}.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:8px 0}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;background:var(--surface)}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:600}
img{max-width:100%;height:auto;border-radius:8px;background:#fcfcfb}li{margin:4px 0}"""


def build_report(runs: list[Path], out: Path, n_boot: int = 1000) -> None:
    results = [evaluate(r, n_boot) for r in runs]
    if len(results) > 1:  # fair comparison: score every model on the (person, item) pairs all of them answered
        keys = [set(zip(r["df"].pid, r["df"].item_id)) for r in results]
        common = set.intersection(*keys)
        print(f"comparing on {len(common)} shared (person, item) pairs")
        for r in results:
            r["df"] = r["df"][[k in common for k in zip(r["df"].pid, r["df"].item_id)]].reset_index(drop=True)
            pop, per_item = M.population(r["df"])
            r.update(ind=M.individual(r["df"], n_boot), pop=pop, per_item=per_item,
                     conf={c: M.confidence(r["df"], c) for c in ("conf_top", "conf_margin")})
    tabs = tables(results)
    heads = [headline(r) for r in results]
    png = chart(results)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = "; ".join(f"{r['name']}: {r['cfg']['n_people']} people, persona {r['cfg']['persona_chars']} chars, "
                    f"shuffle_seed={r['cfg']['shuffle_seed']}" for r in results)

    md = ["# Digital-twin evaluation report", "", "Baseline on Twin-2K-500 holdout (wave 4) items.", ""]
    md += [f"- {h}" for h in heads] + ["", f"Config: {cfg}", ""]
    for title, (h, rows) in tabs.items():
        md += [f"## {title}", "", md_table(h, rows), ""]
    md += ["## Notes", "", NOTES, ""]
    Path(f"{out}.md").write_text("\n".join(md), encoding="utf-8")

    def inline(s: str) -> str:
        parts = html.escape(s).split("**")
        return "".join(f"<strong>{p}</strong>" if i % 2 else p for i, p in enumerate(parts))

    page = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'><title>Digital Twin Eval</title><style>{CSS}</style></head><main>",
            "<h1>Digital-twin evaluation report</h1><p class=sub>Baseline on Twin-2K-500 holdout (wave 4) items. "
            f"{html.escape(cfg)}</p>"]
    page += [f"<div class=card>{inline(h)}</div>" for h in heads]
    page.append(f"<h2>Calibration</h2><img alt='Calibration plot and accuracy by confidence bucket' "
                f"src='data:image/png;base64,{png}'>")
    for title, (h, rows) in tabs.items():
        page.append(f"<h2>{html.escape(title)}</h2>{html_table(h, rows)}")
    notes = "".join(f"<li>{inline(n[2:])}</li>" for n in NOTES.splitlines())
    page.append(f"<h2>Notes</h2><ul>{notes}</ul></main></html>")
    Path(f"{out}.html").write_text("".join(page), encoding="utf-8")
    print(f"wrote {out}.html and {out}.md")
