"""Fixed prompt template. Raw text completion (no chat template) so base and tuned models are comparable.

Layout is persona -> question stem -> options -> (matrix row statement) -> answer cue, so the rows
of one matrix question share a long cached prefix. Options are labelled A, B, C... and the model's
next-token distribution over those letters is the answer distribution. `shuffle_seed` permutes the
displayed option order (order-sensitivity test; one order per question, shared by its matrix rows);
probabilities are mapped back to the original option order.
"""
from __future__ import annotations

import hashlib
import random

from .data import Item

LABELS = "ABCDEFGHI"
STATEMENT_MARK = "\n\nStatement to rate:"
CUE = "\n\nThe respondent chooses option ("

TEMPLATE = """Below is a profile of a real survey respondent, based on their earlier survey answers. \
After it is a new survey question. Answer exactly as this specific person would, not as an average person.

## Respondent profile
{persona}

## New survey question
{stem}
Options:
{options}{statement}""" + CUE


def build_prompt(persona: str, item: Item, pid: str, shuffle_seed: int | None = None):
    """Return (prompt, order); order[k] is the original option index shown at letter k."""
    order = list(range(len(item.options)))
    if shuffle_seed is not None:
        key = hashlib.sha256(f"{shuffle_seed}|{pid}|{item.qid}".encode()).hexdigest()
        random.Random(key).shuffle(order)
    options = "\n".join(f"{LABELS[k]}. {item.options[i]}" for k, i in enumerate(order))
    statement = f"{STATEMENT_MARK} {item.statement}" if item.statement else ""
    prompt = TEMPLATE.format(persona=persona, stem=item.stem, statement=statement, options=options)
    return prompt, order


def prefixes(prompt: str) -> list[str]:
    """Shared prefixes worth prefilling: the persona, then persona + stem + options."""
    i1 = prompt.index("## New survey question")
    end = prompt.rfind(STATEMENT_MARK)
    return [prompt[:i1], prompt[: end if end != -1 else prompt.rfind(CUE)]]
