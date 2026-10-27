"""Analyze a sequence of paired v14/v15 combat benchmark JSON reports.

Example:
  python -u tools/analyze_combat_checkpoints_v15p.py \
    --evaluation 20=results/.../evaluation-0020.json \
    --evaluation 40=results/.../evaluation-0040.json \
    --evaluation 200=results/.../evaluation-0200.json \
    --output results/.../v15p-diagnostic.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sts2_ai.evaluation.combat_diagnostics import (
    diagnose_reports,
    render_markdown,
)


def _atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation", action="append", required=True,
        metavar="ROUND=FILE",
        help="Pair evaluation round with its full benchmark JSON report",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--min-enemy-count", type=int, default=10)
    args = parser.parse_args()
    reports = {}
    for item in args.evaluation:
        raw_round, sep, filename = item.partition("=")
        if not sep or not raw_round.isdecimal() or not filename:
            parser.error(f"Expected ROUND=FILE, got {item!r}")
        stage = int(raw_round)
        if stage in reports:
            parser.error(f"Repeated round: {stage}")
        reports[stage] = json.loads(Path(filename).read_text(encoding="utf-8"))
    result = diagnose_reports(
        reports, bootstrap_samples=args.bootstrap_samples,
        seed=args.seed, min_enemy_count=args.min_enemy_count,
    )
    _atomic(args.output, json.dumps(result, sort_keys=True, indent=2) + "\n")
    markdown = args.output.with_suffix(".md")
    _atomic(markdown, render_markdown(result))
    print(
        f"[v15p] rounds={sorted(reports)} source_runs={result['source_runs']} "
        f"entries={result['scenarios']} report={args.output} markdown={markdown}",
        flush=True,
    )


if __name__ == "__main__":
    main()
