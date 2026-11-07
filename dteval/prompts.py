"""Fixed prompt templates. Raw text completion (no chat template) so base and tuned models are comparable.

Layout is persona -> question stem -> options -> (matrix row statement) -> answer cue, so the rows
of one matrix question share a long cached prefix. Options are labelled A, B, C... and the model's
next-token distribution over those letters is the answer distribution. `shuffle_seed` permutes the
displayed option order (order-sensitivity test; one order per question, shared by its matrix rows);
probabilities are mapped back to the original option order.

Two styles are kept:
  v1  third-person profile ("Below is a profile of a real survey respondent ...")
  v2  first-person ("About me: ...", "My answer is (") - default; far more persona-sensitive.
"""
from __future__ import annotations

import hashlib
import random

from .data import Item

LABELS = "ABCDEFGHI"
DIGITS = "123456789"
STATEMENT_MARK = "\n\nStatement to rate:"

STYLES = {
    "v1": dict(
        head="Below is a profile of a real survey respondent, based on their earlier survey answers. "
             "After it is a new survey question. Answer exactly as this specific person would, not as an average person."
             "\n\n## Respondent profile\n",
        split="\n\n## New survey question\n", cue="\n\nThe respondent chooses option ("),
    "v2": dict(head="", split="\n\nI am filling out a survey. Here is the next question.\n",
               cue="\n\nMy answer is ("),
}


def build_prompt(persona: str, item: Item, pid: str, shuffle_seed: int | None = None,
                 labels: str = LABELS, style: str = "v2"):
    """Return (prompt, order); order[k] is the original option index shown at letter k."""
    st = STYLES[style]
    order = list(range(len(item.options)))
    if shuffle_seed is not None:
        key = hashlib.sha256(f"{shuffle_seed}|{pid}|{item.qid}".encode()).hexdigest()
        random.Random(key).shuffle(order)
    options = "\n".join(f"{labels[k]}. {item.options[i]}" for k, i in enumerate(order))
    statement = f"{STATEMENT_MARK} {item.statement}" if item.statement else ""
    prompt = (f"{st['head']}{persona}{st['split']}{item.stem}\nOptions:\n{options}{statement}{st['cue']}")
    return prompt, order


def prefixes(prompt: str, style: str = "v2") -> list[str]:
    """Shared prefixes worth prefilling: the persona, then persona + stem + options."""
    st = STYLES[style]
    i1 = prompt.index(st["split"])
    end = prompt.rfind(STATEMENT_MARK)
    return [prompt[:i1], prompt[: end if end != -1 else prompt.rfind(st["cue"])]]
