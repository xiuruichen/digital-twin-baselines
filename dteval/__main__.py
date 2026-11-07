"""CLI:  python -m dteval {download | run | report} ..."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(prog="dteval", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="fetch Twin-2K-500 wave_split from Hugging Face")

    r = sub.add_parser("run", help="simulate holdout answers with one model and cache outputs")
    r.add_argument("--name", required=True, help="model label; also the cache/run name")
    r.add_argument("--backend", choices=["llama", "openai"], default="llama")
    r.add_argument("--url", default="http://127.0.0.1:8080", help="llama-server URL or OpenAI-style base URL")
    r.add_argument("--api-model", help="model id sent to the API (openai backend); defaults to --name")
    r.add_argument("--n-people", type=int, default=100, help="0 = all 2058")
    r.add_argument("--persona-chars", type=int, default=4000, help="persona budget; 0 = full (long!)")
    r.add_argument("--shuffle-seed", type=int, default=None, help="shuffle option order (sensitivity test)")
    r.add_argument("--pricing-items", type=int, default=4, help="how many of the 40 pricing items")
    r.add_argument("--matrix-rows", type=int, default=4, help="max rows per matrix question; 0 = all")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--offset", type=int, default=0, help="skip the first N people (disjoint test set)")
    r.add_argument("--out", type=Path, help="default runs/<name>")

    p = sub.add_parser("report", help="build report.html + report.md from run directories")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("report"))
    p.add_argument("--bootstrap", type=int, default=1000)

    s = sub.add_parser("systems", help="LLM vs crowd prior vs persona-feature model vs cross-fit blends")
    s.add_argument("run", type=Path)
    s.add_argument("--out", type=Path, help="default report/systems_<run name>")
    s.add_argument("--C", type=float, default=0.0006, help="regularisation of the persona-feature model")

    a = ap.parse_args()
    parser_default = "http://127.0.0.1:8080"
    if a.cmd == "download":
        from .data import download
        download()
    elif a.cmd == "run":
        from .models import LlamaServer, OpenAIChat
        from .run import simulate
        model = (LlamaServer(a.name, a.url) if a.backend == "llama" else
                 OpenAIChat(a.name, a.api_model or a.name, a.url if a.url != parser_default else
                            "https://api.openai.com/v1"))
        simulate(model, a.out or Path("runs") / a.name, a.n_people or None, a.persona_chars or None,
                 a.shuffle_seed, a.pricing_items, a.matrix_rows or None, a.seed, a.offset)
    elif a.cmd == "systems":
        from .stack import evaluate_run
        evaluate_run(a.run, a.out or Path("report") / f"systems_{a.run.name}", a.C)
    else:
        from .report import build_report
        build_report(a.runs, a.out, a.bootstrap)


main()
