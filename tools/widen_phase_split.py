#!/usr/bin/env python3
"""Create a function-preserving v8 width-64 warm start from the width-32 champion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.models.width_expansion import widen_phase_split


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hidden", type=int, default=64)
    args = p.parse_args()
    if args.source.resolve() == args.output.resolve():
        p.error("Refusing to overwrite the source champion")
    source = PhaseSplitNeuralModel.load(args.source)
    if source.strategy.hidden != 32 or source.combat.hidden != 32:
        p.error("This controlled experiment requires the 32-unit v8 champion")
    widened = widen_phase_split(source, args.hidden)
    if args.output.exists():
        previous = PhaseSplitNeuralModel.load(args.output)
        if previous.to_dict() != widened.to_dict():
            p.error("Existing widened model differs from the deterministic expansion")
        print(f"[capacity] reused and verified widened warm start: {args.output}", flush=True)
    else:
        widened.save(args.output)
        print(f"[capacity] wrote widened warm start: {args.output}", flush=True)
    print(json.dumps({
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "widened_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "source_width": source.strategy.hidden,
        "target_width": widened.strategy.hidden,
        "combat_objective": widened.combat_value_objective,
        "note": "Functional expansion only; PPO optimizer state is NOT transferred",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
