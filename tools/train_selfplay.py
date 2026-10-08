#!/usr/bin/env python3
"""Train and evaluate a first teacher-free neural Silent agent.

The actor sees only public observations and legal menus. The emulator
privately handles game RNG, and no exact-state search or posterior
calculation is used. Early decision caps produce censored runs and
NEVER fake defeat/terminal training labels.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from sts2_ai.agents import NeuralGreedyAgent, RandomAgent
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.evaluation import play_run
from sts2_ai.training.selfplay import SELFPLAY_VERSION, train_selfplay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--auxiliary-weight", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--evaluate-seeds", type=int, default=3)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.evaluate_seeds < 1:
        parser.error("--evaluate-seeds must be positive")
    with JsonlEmulatorBackend(build=args.build) as backend:
        trained = train_selfplay(
            backend,
            rounds=args.rounds, episodes_per_round=args.episodes,
            max_decisions=args.max_decisions,
            dimension=args.dimension, hidden=args.hidden,
            learning_rate=args.learning_rate,
            auxiliary_weight=args.auxiliary_weight,
            seed=args.seed,
        )
        trained.model.save(args.output)
        contenders = {
            "random": RandomAgent(seed=args.seed + 999),
            "neural_greedy": NeuralGreedyAgent(
                trained.model, content_hash=trained.model.model_id
            ),
        }
        evaluations: dict[str, list[dict[str, object]]] = {}
        for name, agent in contenders.items():
            evaluations[name] = []
            for index in range(args.evaluate_seeds):
                episode = play_run(
                    backend, agent, seed=f"selfplay-heldout-{index}",
                    policy=InformationPolicy(FAIR_POLICY_ID),
                    max_decisions=args.max_decisions,
                )
                evaluations[name].append({
                    "seed": episode.seed,
                    "outcome": episode.outcome,
                    "censored": episode.censored,
                    "decisions": episode.decisions,
                    "act": episode.terminal_act,
                    "floor": episode.terminal_floor,
                    "frontier_progress": episode.frontier_progress,
                })
        report = {
            "schema": "sts2-onpolicy-neural-training-smoke-v1",
            "training_version": SELFPLAY_VERSION,
            "game_prior": "ordinary-prototype-emulator-reset",
            "policy_information": FAIR_POLICY_ID,
            "emulator_revision": backend.emulator_revision,
            "model_id": trained.model.model_id,
            "rounds": [asdict(row) for row in trained.rounds],
            "train_completed": sum(row.completed for row in trained.rounds),
            "train_censored": sum(row.censored for row in trained.rounds),
            "gradient_updates": sum(row.update_steps for row in trained.rounds),
            "heldout_evaluation": evaluations,
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
