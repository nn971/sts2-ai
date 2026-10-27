#!/usr/bin/env python3
"""Frozen greedy H7 combat evaluation on never-trained source world namespaces."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from train_h7_reference_combat import collect, summarize

from sts2_ai.models.h7_public_policy import H7PlayerSpec
from sts2_ai.nature.h5_curriculum import source_bank as h5_source_bank
from sts2_ai.nature.research_combat_fixtures import source_bank as h8_source_bank
from sts2_ai.nature.protocol import canonical_sha256

SCHEMA = "sts2-h7-independent-reference-combat-evaluation-v1"


def bootstrap_interval(deltas: list[float], *, seed: int, iterations: int = 2000
                       ) -> tuple[float, float]:
    if not deltas:
        raise ValueError("No paired completed H7 reference combats")
    rng = random.Random(seed)
    averages = sorted(
        statistics.mean(rng.choices(deltas, k=len(deltas)))
        for _ in range(iterations)
    )
    return averages[int(0.025 * iterations)], averages[int(0.975 * iterations)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--player", action="append", required=True, help="label=/path/to/H7.pt")
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--emulator-bindings", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--max-boundaries", type=int, default=1200)
    p.add_argument("--split", choices=("confirmation", "zero-shot"), default="confirmation")
    p.add_argument("--runs", type=int, default=160)
    p.add_argument("--study-seed", type=int, default=73127)
    p.add_argument("--curriculum", choices=("h5", "h8"), default="h5")
    p.add_argument("--curriculum-seed", type=int, default=20261027)
    p.add_argument("--reference-seed", type=int, default=23387)
    p.add_argument("--actor-seed", type=int, default=31337)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    entries: dict[str, Path] = {}
    for item in args.player:
        label, separator, path = item.partition("=")
        if not separator or not label or label in entries:
            p.error("--player must use unique label=path")
        entries[label] = Path(path)
    divisor = 1 if args.curriculum == "h8" else (
        3 if args.split == "zero-shot" else 16
    )
    if args.runs < divisor or args.runs % divisor or args.workers < 1:
        p.error("H7 evaluation must balance the selected source curriculum")
    seeds = (
        h8_source_bank(args.split, args.runs, study_seed=args.study_seed)
        if args.curriculum == "h8"
        else h5_source_bank(args.split, args.runs, study_seed=args.study_seed)
    )
    config = {
        "schema": SCHEMA, "split": args.split,
        "curriculum": args.curriculum,
        "curriculum_seed": args.curriculum_seed if args.curriculum == "h8" else None,
        "source_digest": canonical_sha256(seeds),
        "emulator_revision": args.emulator_revision,
        "models": {name: H7PlayerSpec.from_file(
            path, seed=args.actor_seed, greedy=True,
        ).model_sha256 for name, path in entries.items()},
        "runs": args.runs,
    }
    print(json.dumps(config, indent=2), flush=True)
    if not args.execute:
        return
    arms = {}
    scores: dict[str, list[float]] = {}
    for label, path in entries.items():
        spec = H7PlayerSpec.from_file(path, seed=args.actor_seed, greedy=True)
        result = collect(
            seeds, spec=spec, args=args,
            reference_seed=args.reference_seed + 900000007,
            capture=False,
        )
        arms[label] = summarize(result["episodes"], curriculum=args.curriculum,
                                curriculum_seed=args.curriculum_seed)
        scores[label] = [ep.score for ep in result["episodes"]]
        print(
            f"[h7-eval] {label}: wins={arms[label]['wins']}/{args.runs} "
            f"score={arms[label]['mean_score']:.4f}", flush=True,
        )
    base = next(iter(scores))
    comparisons = {}
    for label in list(scores)[1:]:
        deltas = [new - old for old, new in zip(scores[base], scores[label], strict=True)]
        comparisons[f"{label}-minus-{base}"] = {
            "paired_mean_score_delta": statistics.mean(deltas),
            "paired_bootstrap_95": bootstrap_interval(deltas, seed=73127),
            "improved": sum(d > 0 for d in deltas),
            "worsened": sum(d < 0 for d in deltas),
            "ties": sum(d == 0 for d in deltas),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "config": config, "arms": arms, "comparisons": comparisons,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
