#!/usr/bin/env python3
"""Build a provenance-pinned H10 tactical curriculum from paired native replays."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sts2_ai.evaluation.combat_snapshots import load_recipes
from sts2_ai.training.h10_tactical_priorities import (
    build_opportunities,
    corpus_sha256,
    encode_priority_file,
)
from sts2_ai.training.snapshot_importance import (
    build_importance_distribution,
    snapshot_identity,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-corpus", type=Path, required=True)
    p.add_argument("--comparison", type=Path, action="append", required=True,
                   help="Native paired report: baseline=current, candidate=reference")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target-balance", type=float, default=0.5)
    p.add_argument("--priority-mix", type=float, default=0.7)
    args = p.parse_args()
    if args.output.exists():
        p.error("Output already exists; preserve research provenance")
    recipes = load_recipes(args.train_corpus)
    if not recipes:
        p.error("Empty native training snapshot corpus")
    revisions = {r.emulator_revision for r in recipes}
    if len(revisions) != 1:
        p.error("Multiple emulator revisions in snapshot corpus")
    reports = [json.loads(path.read_text(encoding="utf-8"))
               for path in args.comparison]
    priorities, evidence = build_opportunities(
        recipes, reports, expected_revision=next(iter(revisions)),
    )
    distribution = build_importance_distribution(
        recipes, target_balance=args.target_balance,
        priority_mix=args.priority_mix, priority_scores=priorities,
    )
    output = encode_priority_file(
        recipes, corpus_digest=corpus_sha256(args.train_corpus),
        emulator_revision=next(iter(revisions)),
        current_model=reports[0]["baseline_model"],
        comparisons=[str(path) for path in args.comparison],
        details=evidence,
    )
    output["proposal"] = {
        "target_balance": args.target_balance,
        "priority_mix": args.priority_mix,
        "maximum_importance_weight": max(
            p / q for p, q in zip(
                distribution.target_probabilities,
                distribution.proposal_probabilities, strict=True
            )
        ),
    }
    output["summary"] = {
        "entries": len(priorities),
        "evidence_counts": dict(sorted(Counter(
            item["evidence"] for item in evidence.values()
        ).items())),
        "prioritized_entries": sum(x > 1 for x in priorities.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        **output["summary"],
        "maximum_importance_weight": output["proposal"]["maximum_importance_weight"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
