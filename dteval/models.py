"""Pluggable model backends.

A model is anything with `.name` and `.top_logprobs(prompt) -> list[(token, logprob)]`: the
top next-token candidates after the prompt. Everything else (letter extraction, caching,
renormalisation) is shared, so a new backend is ~20 lines.

Backends:
  LlamaServer  llama.cpp `llama-server` (local, default). Raw /completion, no chat template.
  OpenAIChat   any OpenAI-compatible /chat/completions endpoint that returns logprobs
               (OpenAI, vLLM, llama-server itself, ...). Anthropic's API has no logprobs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Protocol

import requests

TopLogprobs = list[tuple[str, float]]


class Model(Protocol):
    name: str

    def top_logprobs(self, prompt: str) -> TopLogprobs: ...


def _post(url: str, payload: dict, headers: dict | None = None, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=900)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


class LlamaServer:
    def __init__(self, name: str, url: str = "http://127.0.0.1:8080", n_probs: int = 20):
        self.name, self.url, self.n_probs = name, url.rstrip("/"), n_probs

    def top_logprobs(self, prompt: str) -> TopLogprobs:
        out = _post(f"{self.url}/completion", {
            "prompt": prompt, "n_predict": 1, "temperature": 0.0, "seed": 0,
            "n_probs": self.n_probs, "cache_prompt": True})
        return [(t["token"], t["logprob"]) for t in out["completion_probabilities"][0]["top_logprobs"]]


    def prefill(self, prefix: str) -> None:
        """Process a prompt prefix so the server stores a recurrent-state checkpoint exactly at its end.

        Hybrid (Qwen3.5-style) models can only resume from checkpoints; without this, each request
        re-evaluates tokens back to the previous checkpoint boundary.
        """
        _post(f"{self.url}/completion", {"prompt": prefix, "n_predict": 0, "cache_prompt": True})


class OpenAIChat:
    def __init__(self, name: str, api_model: str, base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY"):
        self.name, self.api_model, self.base_url = name, api_model, base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key and "api.openai.com" in base_url:
            raise SystemExit(f"Set {api_key_env} to use the API model.")

    def top_logprobs(self, prompt: str) -> TopLogprobs:
        out = _post(f"{self.base_url}/chat/completions", {
            "model": self.api_model, "max_tokens": 1, "temperature": 0, "seed": 0,
            "logprobs": True, "top_logprobs": 20,
            "messages": [{"role": "user",
                          "content": prompt + " (reply with the single letter only)"}]},
            headers={"Authorization": f"Bearer {self.api_key}"})
        top = out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        return [(t["token"], t["logprob"]) for t in top]


class Cache:
    """Append-only JSONL cache of raw model outputs, keyed by (model, prompt)."""

    def __init__(self, path: Path):
        self.path, self.data = path, {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.data[rec["k"]] = rec["v"]
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}\0{prompt}".encode()).hexdigest()

    def has(self, model: Model, prompt: str) -> bool:
        return self.key(model.name, prompt) in self.data

    def get_or_compute(self, model: Model, prompt: str) -> TopLogprobs:
        k = self.key(model.name, prompt)
        if k not in self.data:
            self.data[k] = model.top_logprobs(prompt)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"k": k, "v": self.data[k]}) + "\n")
        return [tuple(x) for x in self.data[k]]


def letter_probs(top: TopLogprobs, labels: str) -> tuple[list[float], float]:
    """Probability of each option letter (tokens 'A' or ' A'), renormalised over valid letters.

    Returns (probs, valid_mass); valid_mass is the raw probability the model put on valid letters.
    """
    mass = {c: 0.0 for c in labels}
    for tok, lp in top:
        t = tok.strip()
        if t in mass:
            mass[t] += math.exp(lp)
    total = sum(mass.values())
    if total <= 0:
        return [1 / len(labels)] * len(labels), 0.0
    return [mass[c] / total for c in labels], total
