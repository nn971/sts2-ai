"""Generate a strict paired λ=0 versus λ=0.7 combat training/evaluation report."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sts2_ai.evaluation.ablation_analysis import (
    compare_ablation,
    render_markdown,
)


def _atomic(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-train", type=Path, required=True)
    parser.add_argument("--priority-train", type=Path, required=True)
    parser.add_argument("--target-eval", type=Path, required=True)
    parser.add_argument("--priority-eval", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    def load(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    result = compare_ablation(
        target_train=load(args.target_train),
        priority_train=load(args.priority_train),
        target_eval=load(args.target_eval),
        priority_eval=load(args.priority_eval),
        manifest=load(args.manifest) if args.manifest else None,
        samples=args.bootstrap_samples, seed=args.seed,
    )
    _atomic(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    markdown = args.output.with_suffix(".md")
    _atomic(markdown, render_markdown(result))
    delta = result["evaluation"]["priority_vs_target"]
    print(
        f"[v15q ablation] entries={result['scenarios']} "
        f"sources={result['source_runs']} "
        f"delta_survival={delta['delta_win_rate']:+.4f} "
        f"delta_hp={delta['delta_exit_hp']:+.3f} "
        f"report={args.output} markdown={markdown}",
        flush=True,
    )


if __name__ == "__main__":
    main()
