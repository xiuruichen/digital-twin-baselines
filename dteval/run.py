"""Simulate: for every person x holdout item, prompt the model and save the option distribution."""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import data, prompts
from .models import Cache, Model, letter_probs


def simulate(model: Model, out_dir: Path, n_people: int | None, persona_chars: int | None,
             shuffle_seed: int | None, pricing_k: int, matrix_rows: int | None,
             seed: int = 0, cache_dir: Path = Path("cache")) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(cache_dir / f"{model.name.replace('/', '_')}.jsonl")
    people = data.select_people(data.load_people(), n_people, seed)
    config = dict(model=model.name, n_people=len(people), persona_chars=persona_chars,
                  shuffle_seed=shuffle_seed, pricing_k=pricing_k, matrix_rows=matrix_rows, seed=seed)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    path = out_dir / "predictions.jsonl"
    t0 = time.time()
    with path.open("w", encoding="utf-8") as f:
        for n, row in enumerate(people.itertuples(index=False), 1):
            persona = data.render_persona(row.wave1_3_persona_json, persona_chars)
            done_prefix: set[str] = set()
            for item in data.person_items(row, pricing_k, matrix_rows, seed):
                prompt, order = prompts.build_prompt(persona, item, str(row.pid), shuffle_seed)
                if hasattr(model, "prefill") and not cache.has(model, prompt):
                    for pre in prompts.prefixes(prompt):  # checkpoint exactly where prompts diverge
                        if pre not in done_prefix:
                            model.prefill(pre)
                            done_prefix.add(pre)
                shown, valid = letter_probs(cache.get_or_compute(model, prompt),
                                            prompts.LABELS[: len(order)])
                probs = [0.0] * len(order)
                for k, i in enumerate(order):  # map displayed letter back to original option
                    probs[i] = shown[k]
                f.write(json.dumps({"pid": str(row.pid), "item_id": item.item_id, "qid": item.qid,
                                    "block": item.block, "n_options": len(order), "gold": item.gold,
                                    "retest": item.retest, "probs": probs,
                                    "valid_mass": valid}) + "\n")
            f.flush()
            print(f"[{n}/{len(people)}] pid={row.pid}  {time.time() - t0:.0f}s elapsed", flush=True)
    return path
