# dteval: a baseline evaluation harness for open digital-twin models

How well do open LLMs simulate **specific real people**, and can a cheap confidence signal
(next-token log-probabilities) tell us **when the simulation is wrong**?

This repo is a baseline and scoring harness for an open human-behavior model, not a Simile clone.
It runs on [Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500)
(2,058 US participants, 4 survey waves) and reports three things separately:

| Level | Question | Metric |
|---|---|---|
| **Individual** | Does the twin pick the same option the person picked? | accuracy, and accuracy / that person's own test-retest accuracy ("normalized accuracy") |
| **Population** | Does the twin reproduce the crowd's answer distribution? | total-variation distance per question |
| **Confidence** | Does the model's confidence predict its own errors, out of sample? | AUROC, accuracy by confidence bucket, error ratio low vs high confidence, ECE, calibration plot |

Individual and population numbers are never blended into one score. Everything is scored against
recorded human answers only (no LLM judge).

## Results

**One-sentence version.** On 2,100 held-out answers from 60 people, the best system (crowd prior + persona-feature
ensemble + a 9B ternary LLM, blend weights cross-fit) reaches **51.2% accuracy, 88% of human test-retest consistency**, but
it is only **+3.3 points (95% CI +1.6 to +5.2) above the strongest no-LLM baseline, "predict what the other 2,057 people
said"**, and the LLM itself contributes **+0.2 points (CI -0.3 to +0.8)** on top of persona features.

Systems, scored on the same 60 people (`report/systems_dev_v2_60people.md`; human test-retest 58.1%, chance 29.2%):

| System | Accuracy | vs crowd prior (95% CI) | Log-lik. of real answer |
|---|---|---|---|
| LLM alone (Ternary-Bonsai-2 9B, v2 prompt) | 39.0% | -8.9 pts | -1.349 |
| **Crowd prior** (modal answer of the other people) | 47.9% | baseline | -1.167 |
| Persona-feature ensemble (no LLM) | 50.8% | +2.9 (+1.0 to +4.8) | -1.108 |
| Blend: crowd + persona-feature (no LLM) | 51.0% | +3.1 (+1.4 to +5.0) | -1.108 |
| Blend: crowd + persona-feature + LLM | **51.2%** | **+3.3 (+1.6 to +5.2)** | **-1.105** |

A better-powered check that needs no LLM (`python -m dteval systems-no-llm`), on 200 people never used for tuning
(7,000 answers): persona-feature ensemble **50.0% vs crowd prior 47.3% = +2.70 points (95% CI +1.7 to +3.7)**.

What this means:

- **A crowd lookup is a strong baseline.** The modal answer of the *other 2,057 people* already scores ~47-48%. The harness's
  earlier "crowd-mode baseline" was computed from the evaluated sample only (31-39%), which understated it. Any claim that a twin
  works should be stated relative to the crowd prior, not to chance. Raw accuracy alone is nearly meaningless here.
- **Individual signal is real but small.** Wave 1-3 answers (demographics, personality, response style, cognitive-test answers) lift
  accuracy about 2.7-3.3 points above the crowd, against a headroom of ~10 points to the human test-retest ceiling.
- **The LLM does not add much on top of the persona features** (+0.2 pts, not significant), even after fixing the prompt (below).
  Used alone, its answer distributions are almost identical across very different people with the original prompt format.
- **Target check.** Goal set for this iteration: accuracy >= 42% *and* >= 4 points above the crowd prior with a CI excluding zero.
  Accuracy >= 42% is met (a crowd lookup alone clears it); the +4 lift over the crowd prior is **not met** (+3.3 on 60 people,
  +2.7 on 200). The 200-person interval's upper end (+3.7) is below 4, so more tuning on the same inputs is unlikely to get there.
  The next real gains need a stronger LLM, or fine-tuning on Twin-2K.

### What was tried (so it is not repeated)

| Idea | Outcome |
|---|---|
| Persona-feature models (LR, boosted trees, extra-trees, average of the three), tuned only on people 200+ | crowd 48.0% -> 50.5% on tuning people; all tree settings within 0.5 pts |
| Response-style features (extreme / midpoint use, mean agreement) | +0.6 pts |
| Free-text cognitive-test answers as one-hot flags | +0.1 pts |
| PCA latent traits, pooled-by-question models, ordinal two-stage model, stage-2 stacking across items | no gain (stage-2: +0.4 with leakage, dropped) |
| LLM prompt v2 (first-person "About me" persona, "My answer is (") vs v1 | LLM log-lik. -1.363 -> -1.321 on the same 420 answers; raw accuracy 34% -> 39% |
| Digit labels instead of letters for Likert items | accuracy up, log-lik. unchanged: argmax flips, no better probabilities |
| Qwen3.5-9B at Q4 vs Ternary-Bonsai-2 9B (12 people, Likert) | no better: quantization is not the bottleneck |
| Label-free LLM calibration (subtract the item's mean log-prob) | hurts the LLM alone; the blend uses it only as a feature |

Model comparison from the first iteration (v1 prompt, same 10 people / 350 answers, `report/model_comparison_10people.html`):
Bonsai-2 9B 36.6% (log-lik. -1.353), Qwen3.5-4B Q4_K_M 30.0% (-1.457), Bonsai-2 4B 21.7% (-1.683); the 9B is the default local model.

## Quick start (Windows, no GPU needed)

```powershell
# 1. Python env + dependencies
uv venv --python 3.12 .venv ; uv pip install -e ".[dev]" --python .venv\Scripts\python.exe
.venv\Scripts\python -m dteval download            # Twin-2K-500 wave_split (~200 MB)

# 2. A local model server. llama.cpp release b12180 (pinned), Vulkan build runs on integrated GPUs;
#    the CPU build also works (-Backend cpu).
mkdir tools\llama-vulkan
curl -L -o tools\v.zip https://github.com/ggml-org/llama.cpp/releases/download/b12180/llama-b12180-bin-win-vulkan-x64.zip
tar -xf tools\v.zip -C tools\llama-vulkan
.venv\Scripts\python -c "from huggingface_hub import hf_hub_download as d; d('prism-ml/Ternary-Bonsai-2-9B-gguf','Ternary-Bonsai-2-9B-Q2_0.gguf',local_dir='models')"
scripts\serve.ps1 models\Ternary-Bonsai-2-9B-Q2_0.gguf          # leave running in another terminal

# 3. Simulate (cached + resumable), then report
.venv\Scripts\python -m dteval run --name bonsai2-9b --n-people 20     # pilot subset
.venv\Scripts\python -m dteval run --name bonsai2-9b --n-people 100    # extends the pilot; cache is reused
.venv\Scripts\python -m dteval report runs\bonsai2-9b --out report\report

# 4. Does the LLM beat "ask the crowd"? Crowd prior, persona-feature models, cross-fit blends, paired bootstrap.
.venv\Scripts\python -m dteval systems runs\bonsai2-9b          # first call fits ~10 min of tree models, then cached
.venv\Scripts\python -m dteval systems-no-llm --n-people 200    # no LLM needed; cheap, tight intervals
```

On macOS/Linux, build or download llama.cpp for your platform, start
`llama-server -m <model.gguf> -c 4096 -np 1 -cms 0 -ub 64 --port 8080`, and use the same `dteval` commands.
Any server exposing llama.cpp's `/completion` endpoint works; vLLM / OpenAI-compatible servers work via `--backend openai`.

### Comparing models

```powershell
# another local checkpoint: restart serve.ps1 with a different GGUF, then
python -m dteval run --name qwen3.5-4b-q4 --n-people 100
# API comparison point (needs logprobs, so OpenAI-compatible; Anthropic's API does not expose them)
$env:OPENAI_API_KEY = "..." ; python -m dteval run --backend openai --name gpt-4.1-mini --n-people 100 --persona-chars 0
# order-sensitivity test: same model, shuffled option order
python -m dteval run --name bonsai2-9b --n-people 100 --shuffle-seed 1 --out runs\bonsai2-9b-shuffled
# one report for everything
python -m dteval report runs\bonsai2-9b runs\qwen3.5-4b-q4 runs\gpt-4.1-mini --out report\report
```

`--persona-chars 0` sends the full categorical persona (~15k tokens) which is fine for an API model
but far too slow for a laptop CPU/iGPU.

## Approach

1. **Data split.** Holdout = the 85 wave-4 question IDs (heuristics-and-biases and behavioral-economics
   experiments). The persona comes only from the dataset's `wave1_3_persona_json`, which does not contain them.
   The same person's wave 1-3 answer to the same item gives the **test-retest ceiling**.
2. **Persona.** The full wave 1-3 text is ~95k characters (~25k tokens), too long for local inference. We render a compact
   `question: answer` list (option lists dropped), in priority order demographics, personality, economic preferences,
   truncated to a character budget (default 4000, ~1k tokens). The budget is a CLI flag, and the long-persona
   effect is an obvious ablation.
3. **Prompt.** One fixed raw-completion template (`dteval/prompts.py`), no chat template, so base and tuned checkpoints are
   compared like for like. Options are lettered; the prompt ends with `The respondent chooses option (` and we read the
   next-token distribution over the letters. (A first draft ending in `...option letter:` put 90% of the probability on
   non-letter tokens; the letters must carry most of the mass; `valid_mass` is reported for every model so this cannot hide.)
   Probabilities are renormalised over valid letters. `--shuffle-seed` permutes the displayed option order to test
   position/label artifacts.
4. **Items.** Categorical items only: single-choice questions and Likert-matrix rows. Sliders and free-text
   numerics are skipped in this phase. To bound compute, matrix questions are capped at 4 rows and the 40 yes/no
   pricing items at 4, using fixed seeded subsets (the same items for everyone, so population metrics are well defined).
   Both are CLI flags (`--matrix-rows 0 --pricing-items 40` = everything).
5. **Individual metrics.** Accuracy; test-retest accuracy on the same items; normalized accuracy
   (bootstrap CI over people); leave-one-out "predict the crowd's mode" baseline; chance. If the twin does not beat the
   crowd-mode baseline, the persona is not adding individual signal.
6. **Population metrics.** Per-question total variation between real answers and (soft) mean model probabilities,
   (hard) argmax picks, plus uniform-guess and human-retest reference rows. TV is biased upward for small n, so compare models on the same people.
7. **Confidence.** Signals: top-option probability and top-2 margin. People are split in two halves by a hash of their
   id; bucket edges and the low/high threshold are fit on one half and applied to the other. AUROC is reported pooled and
   **within question** (pooled AUROC is inflated by the fact that items differ in option count and difficulty).
8. **Caching and determinism.** Every model call is stored in `cache/<model>.jsonl` keyed by `(model, prompt)`;
   re-running costs nothing and interrupted runs resume. People are chosen by a seeded shuffle (a 20-person run is a subset of a
   100-person run). Temperature 0, seed 0.

## Model choice on this machine

Developed on a Ryzen 5 PRO 5650U laptop: 14.8 GB RAM, no discrete GPU. Raw `llama-bench` throughput (b12180, prompt-processing tokens/s; end-to-end speed differs, see below):

| Model | File | Backend | Prefill tok/s |
|---|---|---|---|
| Qwen3.5-4B Q4_K_M | 2.5 GiB | CPU, 12 threads | 49 |
| Qwen3.5-4B Q4_K_M | 2.5 GiB | Vulkan iGPU | 55 |
| Ternary-Bonsai-2-4B Q2_0 | 1.1 GiB | CPU | 20 |
| Ternary-Bonsai-2-4B Q2_0 | 1.1 GiB | Vulkan iGPU | 62 |
| **Ternary-Bonsai-2-9B Q2_0** | 2.4 GiB | Vulkan iGPU | 35 |

[Ternary Bonsai 2](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf) (PrismML, built on Qwen3.5) is a ~1.7-bit
ternary quantization: a 9B model in 2.4 GiB. It runs on upstream llama.cpp via the plain `Q2_0` file; the Vulkan backend is ~2-3x
faster than CPU for it here. Default local model: **Bonsai-2 9B**. This is a quantized, assistant-style model, which is a
known weakness for simulating people (see pitfalls), so treat it as a baseline, not a ceiling.

Compute reality: about 70 seconds per person on the 9B model (35 items), so 100 people is ~2 hours; run it in the background, it resumes from cache.
Measured, not assumed: the 4B models were *not* faster end to end (about 70 s per person for Bonsai-4B, ~100 s for Qwen3.5-4B Q4_K_M), because
per-request overhead dominates, and they were clearly worse. What did help (2.2x, 160 s to 73 s per person):
(1) the prompt is ordered persona -> stem -> options -> matrix-row statement, so the rows of one matrix share a cached prefix;
(2) before each new item the client sends a short `n_predict=0` prefill request, so llama-server stores its recurrent-state checkpoint exactly where
prompts diverge (hybrid Qwen3.5 models can only resume from checkpoints, otherwise ~64 extra tokens are re-read per item).
`scripts/compare.sh <name> <gguf> <n-people>` starts a server, runs and stops it, for model sweeps.

## Known pitfalls and how they are handled

- *Chat/assistant models collapse toward consensus.* Models are swappable; the report includes the crowd-mode baseline and TV against the real distribution so collapse is visible.
- *Group-level fit is not individual fit.* Reported separately.
- *Prompt artifacts.* Fixed, documented template; `--shuffle-seed` test.
- *Confidence only counts out of sample.* Split-half protocol above.
- *No LLM judge.* Scoring is exact match to recorded answers.

## Limitations

- Exact-match scoring on 5-7 point ordinal scales is strict; both the model and the humans' own retest are penalized, hence the normalization.
- The compact persona drops most of the survey (cognitive tests, most personality scales); full-persona ablations are cheap with an API model.
- Wave-4 conditions assigned between subjects mean each item has ~500-1000 respondents in the full dataset; with 100 people expect 25-60 per item, so per-question TV is noisy.
- Only one API-style comparison is wired (OpenAI-compatible). Untested here without a key; the backend is covered by the same code path as the local one.

## Layout

```
dteval/data.py      load data, compact persona, holdout items + test-retest answers
dteval/prompts.py   fixed prompt templates (v1 third-person, v2 first-person) and option shuffling
dteval/models.py    pluggable backends (LlamaServer, OpenAIChat), JSONL cache, letter extraction
dteval/run.py       person x item simulation loop
dteval/metrics.py   individual / population / confidence metrics
dteval/report.py    HTML + markdown report with calibration chart
dteval/features.py  LLM-free baselines: crowd prior + persona-feature ensemble, out-of-fold over all people
dteval/stack.py     cross-fit blends of crowd / persona-feature / LLM; paired-bootstrap system comparison
scripts/serve.ps1   start llama-server with the flags this harness needs
tests/              unit tests (python -m pytest)
```

Not in this phase (but not designed out): fine-tuning an open 8B model to emit answer distributions, a hidden-state probe as a
stronger confidence model, more benchmarks (SimBench, SocSci210), collecting a new consented panel. Adding a model = one small class
with `top_logprobs(prompt)`; adding a benchmark = a new loader that yields `Item`s.

## License

MIT (code). Twin-2K-500 is CC BY 4.0; Bonsai weights are Apache 2.0.
