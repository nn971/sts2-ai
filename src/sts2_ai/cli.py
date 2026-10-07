from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from sts2_ai.agents import HeuristicAgent, OracleMctsAgent, RandomAgent
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.evaluation import collect_experiment_manifest, play_run
from sts2_ai.search import UctMcts


def main() -> None:
    parser = argparse.ArgumentParser(description="sts2-ai research utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="write a minimal experiment manifest")
    manifest.add_argument("output", type=Path)
    manifest.add_argument("--repo-root", type=Path, default=Path.cwd())
    manifest.add_argument("--experiment-id", required=True)
    manifest.add_argument("--game-build", default="unknown")
    manifest.add_argument("--schema-version", default="unknown")
    manifest.add_argument("--binding-version", default="unknown")
    manifest.add_argument("--information-policy", default="fair-v1")

    evaluate = sub.add_parser("evaluate", help="run whole-run emulator baselines")
    evaluate.add_argument("--agent", choices=("random", "heuristic", "mcts"), required=True)
    evaluate.add_argument("--budget", type=int, default=32, help="MCTS simulations per decision")
    evaluate.add_argument("--seeds", type=int, default=10, help="number of deterministic run seeds")
    evaluate.add_argument("--seed-prefix", default="eval")
    evaluate.add_argument("--agent-seed", type=int, default=0)
    evaluate.add_argument("--ascension", type=int, default=0)
    evaluate.add_argument("--max-decisions", type=int)
    evaluate.add_argument("--repo-root", type=Path, default=Path.cwd())
    evaluate.add_argument("--no-build", action="store_true")
    evaluate.add_argument("--json-output", type=Path)

    args = parser.parse_args()
    if args.command == "manifest":
        result = collect_experiment_manifest(
            repo_root=args.repo_root,
            experiment_id=args.experiment_id,
            game_build=args.game_build,
            emulator_schema_version=args.schema_version,
            binding_version=args.binding_version,
            information_policy=args.information_policy,
            config={},
            seeds={},
        )
        result.write_json(args.output)
        print(args.output)
        return

    if args.command == "evaluate":
        _evaluate(args)


def _evaluate(args: argparse.Namespace) -> None:
    if args.seeds <= 0:
        raise SystemExit("--seeds must be positive")
    if args.budget < 0:
        raise SystemExit("--budget must be non-negative")

    policy = InformationPolicy(FAIR_POLICY_ID)
    summaries = []
    with JsonlEmulatorBackend(
        repo_root=args.repo_root,
        build=not args.no_build,
    ) as backend:
        for index in range(args.seeds):
            if args.agent == "random":
                agent = RandomAgent(seed=args.agent_seed + index)
            elif args.agent == "heuristic":
                agent = HeuristicAgent()
            else:
                search = UctMcts(
                    backend,
                    policy=policy,
                    rollout_policy=HeuristicAgent(),
                    seed=args.agent_seed + index,
                )
                agent = OracleMctsAgent(search, simulations=args.budget)

            summaries.append(
                play_run(
                    backend,
                    agent,
                    seed=f"{args.seed_prefix}-{index}",
                    policy=policy,
                    ascension=args.ascension,
                    max_decisions=args.max_decisions,
                )
            )

    wins = sum(summary.won for summary in summaries)
    avg_progress = statistics.fmean(summary.terminal_progress for summary in summaries)
    avg_decisions = statistics.fmean(summary.decisions for summary in summaries)
    total_decisions = sum(summary.decisions for summary in summaries)
    total_transitions = sum(summary.emulator_transitions for summary in summaries)
    transitions_per_decision = (
        total_transitions / total_decisions if total_decisions else 0.0
    )
    avg_time = statistics.fmean(summary.wall_seconds for summary in summaries)
    label = args.agent if args.agent != "mcts" else f"MCTS-{args.budget}"

    print(
        "| Agent | Runs | Win % | Avg terminal progress | "
        "Emulator transitions/decision | Time/run |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    print(
        f"| {label} | {len(summaries)} | {100.0 * wins / len(summaries):.1f} | "
        f"{avg_progress:.2f} | {transitions_per_decision:.1f} | {avg_time:.3f}s |"
    )
    print(f"Average decisions/run: {avg_decisions:.1f}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent": label,
            "budget": args.budget if args.agent == "mcts" else None,
            "runs": [asdict(summary) for summary in summaries],
        }
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
