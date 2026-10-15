#!/usr/bin/env python3
"""Equal-simulator-budget toy combat A/B for intent/draw Nature.

Demonstrates the learning pipeline on a SYNTHETIC game with declared
toy chance laws, before the still-pending real semantic game/JSONL bridge.
Do not use synthetic numbers as STS2 win-rate evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts2_ai.nature.corpus import atomic_write
from sts2_ai.training.two_site_combat_toy import run_paired_toy_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--episodes-per-batch", type=int, default=32)
    parser.add_argument("--evaluation-episodes", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--nature-mix", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_paired_toy_pilot(
        rounds=args.rounds, episodes_per_batch=args.episodes_per_batch,
        seed=args.seed, evaluation_episodes=args.evaluation_episodes,
        nature_mix=args.nature_mix,
    )
    for checkpoint in result["progress"]:
        print(
            "[nature-two-site-toy] "
            f"round={checkpoint['round']} "
            f"native_return={checkpoint['native']['return']:.4f} "
            f"nature_trained_return={checkpoint['nature_train']['return']:.4f} "
            f"native_eval_delta={checkpoint['delta_native_eval_return']:+.4f}",
            flush=True,
        )
    atomic_write(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[nature-two-site-toy] report={args.output}", flush=True)


if __name__ == "__main__":
    main()
