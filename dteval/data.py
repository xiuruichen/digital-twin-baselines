"""Load Twin-2K-500, build compact personas, and extract the holdout items.

Holdout = the 85 wave-4 question IDs (heuristics-and-biases / behavioral-economics
experiments). They are NOT in the wave 1-3 persona, so there is no leakage.
Test-retest = the same person's wave 1-3 answer to the same item.
Only categorical items are scored (multiple choice and Likert matrix rows);
sliders and free-text numeric entries are skipped in this phase.
"""
from __future__ import annotations

import glob
import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/twin2k")
REPO = "LLM-Digital-Twin/Twin-2K-500"
PERSONA_BLOCK_PRIORITY = ["Demographics", "Personality", "Economic preferences"]
PRICING_PREFIX = "QID9_"  # 40 yes/no purchase items; a fixed seeded subset is used


@dataclass(frozen=True)
class Item:
    item_id: str                  # e.g. "QID158" or "QID287#3" (matrix row 3)
    qid: str
    block: str
    stem: str                     # question text
    statement: str | None         # matrix row text, if any
    options: tuple[str, ...]
    gold: int                     # 0-based index of the wave-4 answer
    retest: int | None            # 0-based wave 1-3 answer to the same item, if any


def download(data_dir: Path = DATA_DIR) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, repo_type="dataset", local_dir=str(data_dir),
                      allow_patterns=["wave_split/**", "README.md"])


def load_people(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "wave_split/chunks/*.parquet")))
    if not files:
        raise SystemExit("Dataset missing. Run: python -m dteval download")
    return pd.concat([pd.read_parquet(f) for f in files]).reset_index(drop=True)


def select_people(df: pd.DataFrame, n: int | None, seed: int = 0) -> pd.DataFrame:
    """Seeded shuffle then take the first n, so a 20-person run is a subset of a 100-person run."""
    order = list(range(len(df)))
    random.Random(seed).shuffle(order)
    return df.iloc[order[: n or len(df)]].reset_index(drop=True)


def _answers(js: str) -> dict[str, tuple[int, str, dict, int | None]]:
    """item_id -> (0-based answer, block name, question dict, matrix row) for categorical items."""
    out = {}
    for block in json.loads(js):
        for q in block.get("Questions", []):
            a = q.get("Answers") or {}
            pos = a.get("SelectedByPosition")
            if q["QuestionType"] == "MC" and isinstance(pos, int):
                out[q["QuestionID"]] = (pos - 1, block["BlockName"].strip(), q, None)
            elif q["QuestionType"] == "Matrix" and isinstance(pos, list):
                for r, p in enumerate(pos):
                    if isinstance(p, int):
                        out[f'{q["QuestionID"]}#{r}'] = (p - 1, block["BlockName"].strip(), q, r)
    return out


def pricing_subset(k: int, seed: int = 0) -> set[str]:
    ids = [f"{PRICING_PREFIX}{i}" for i in range(1, 41)]
    random.Random(seed).shuffle(ids)
    return set(ids[:k])


def _keep_row(qid: str, r: int, n_rows: int, k: int | None, seed: int) -> bool:
    """Fixed (same for every person) seeded subset of k matrix rows per question."""
    if k is None or n_rows <= k:
        return True
    rows = list(range(n_rows))
    random.Random(f"{seed}|{qid}").shuffle(rows)
    return r in rows[:k]


def person_items(row: pd.Series, pricing_k: int = 4, matrix_rows: int | None = 4,
                 seed: int = 0) -> list[Item]:
    """Holdout items for one person. Subsets are global (same items for everyone), to limit compute."""
    truth, before = _answers(row.wave4_Q_wave4_A), _answers(row.wave4_Q_wave1_3_A)
    keep_pricing = pricing_subset(pricing_k, seed)
    items = []
    for iid, (gold, block, q, r) in truth.items():
        if iid.startswith(PRICING_PREFIX) and iid not in keep_pricing:
            continue
        if r is not None and not _keep_row(q["QuestionID"], r, len(q["Rows"]), matrix_rows, seed):
            continue
        options = tuple(q["Options"] if r is None else q["Columns"])
        if not 0 <= gold < len(options) or len(options) < 2 or len(options) > 9:
            continue
        retest = before[iid][0] if iid in before else None
        if retest is not None and not 0 <= retest < len(options):
            retest = None
        items.append(Item(iid, q["QuestionID"], block, q["QuestionText"].strip(),
                          None if r is None else q["Rows"][r].strip(), options, gold, retest))
    return items


def render_persona(persona_json: str, max_chars: int | None = 4000) -> str:
    """Compact 'question: answer' persona from wave 1-3 answers, in block-priority order.

    Option lists are dropped; whole entries are kept until the character budget is used.
    max_chars=None keeps every categorical answer (long: ~15k tokens).
    """
    blocks: dict[str, list[str]] = {}
    for block in json.loads(persona_json):
        name = block["BlockName"].strip()
        for q in block.get("Questions", []):
            a = q.get("Answers") or {}
            stem = " ".join(q["QuestionText"].split())[:200]
            if q["QuestionType"] == "MC" and isinstance(a.get("SelectedText"), str):
                blocks.setdefault(name, []).append(f"- {stem}: {a['SelectedText']}")
            elif q["QuestionType"] == "Matrix" and isinstance(a.get("SelectedText"), list):
                rows = [f"  - {r.strip()}: {t}" for r, t in zip(q["Rows"], a["SelectedText"])]
                blocks.setdefault(name, []).append(f"- {stem}\n" + "\n".join(rows))
    names = [b for b in PERSONA_BLOCK_PRIORITY if b in blocks]
    if max_chars is None:
        names += [b for b in blocks if b not in names]
    lines, used = [], 0
    for name in names:
        for entry in blocks[name]:
            if max_chars is not None and used + len(entry) + 1 > max_chars:
                continue  # skip entries that don't fit, keep trying smaller ones
            lines.append(entry)
            used += len(entry) + 1
    return "\n".join(lines)
