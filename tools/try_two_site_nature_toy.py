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
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="Independent training seeds; overrides --seed")
    parser.add_argument("--nature-mix", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = args.seeds or [args.seed]
    if len(seeds) != len(set(seeds)):
        parser.error("Training seeds must be unique")
    runs = []
    for seed in seeds:
        result = run_paired_toy_pilot(
            rounds=args.rounds, episodes_per_batch=args.episodes_per_batch,
            seed=seed, evaluation_episodes=args.evaluation_episodes,
            nature_mix=args.nature_mix,
        )
        runs.append(result)
        for checkpoint in result["progress"]:
            print(
                "[nature-two-site-toy] "
                f"seed={seed} round={checkpoint['round']} "
                f"native_return={checkpoint['native']['return']:.4f} "
                f"nature_trained_return={checkpoint['nature_train']['return']:.4f} "
                f"native_eval_delta={checkpoint['delta_native_eval_return']:+.4f} "
                f"frozen_nature_difficulty="
                f"{checkpoint['frozen_player_difficulty']['nature_minus_native']:+.4f} "
                f"learned_vs_untrained_nature="
                f"{checkpoint['frozen_player_difficulty']['learned_minus_untrained_nature']:+.4f}",
                flush=True,
            )
    if len(runs) == 1:
        report = runs[0]
    else:
        import statistics

        deltas = [
            run["progress"][-1]["delta_native_eval_return"] for run in runs
        ]
        difficulties = [
            run["progress"][-1]["frozen_player_difficulty"]["nature_minus_native"]
            for run in runs
        ]
        learned_deltas = [
            run["progress"][-1]["frozen_player_difficulty"][
                "learned_minus_untrained_nature"
            ] for run in runs
        ]
        report = {
            "schema": "sts2-ai-two-site-toy-multiseed-v1",
            "synthetic_not_sts2": True,
            "seeds": seeds,
            "runs": runs,
            "summary": {
                "mean_native_eval_delta": statistics.mean(deltas),
                "stdev_native_eval_delta": statistics.stdev(deltas),
                "mean_frozen_nature_difficulty": statistics.mean(difficulties),
                "mean_learned_minus_untrained_nature": statistics.mean(learned_deltas),
                "nature_better_native_eval_count": sum(x > 0 for x in deltas),
                "seed_count": len(seeds),
            },
        }
        print(
            "[nature-two-site-toy] "
            f"multi_seed_mean_delta={report['summary']['mean_native_eval_delta']:+.4f} "
            f"mean_difficulty={report['summary']['mean_frozen_nature_difficulty']:+.4f} "
            f"learned_vs_untrained="
            f"{report['summary']['mean_learned_minus_untrained_nature']:+.4f} "
            f"better_native_eval={report['summary']['nature_better_native_eval_count']}"
            f"/{len(seeds)}",
            flush=True,
        )
    atomic_write(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[nature-two-site-toy] report={args.output}", flush=True)


if __name__ == "__main__":
    main()
