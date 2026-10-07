from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from sts2_ai.agents import (
    Agent,
    ExactStateAgent,
    HeuristicAgent,
    OracleMctsAgent,
    RandomAgent,
)
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.evaluation import collect_experiment_manifest, play_run
from sts2_ai.search import SearchResult, UctMcts
from sts2_ai.strategy_db import SQLiteStrategyStore, record_search_result


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
    evaluate.add_argument(
        "--strategy-db",
        type=Path,
        default=Path("results/strategy.sqlite"),
        help="SQLite evidence store used by MCTS evaluations",
    )
    evaluate.add_argument("--game-build", default="unknown")
    evaluate.add_argument(
        "--profile",
        action="store_true",
        help="print JSONL bridge operation timing for the actual workload",
    )

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
    bridge_profile = {}
    strategy_store = (
        SQLiteStrategyStore(args.strategy_db)
        if args.agent == "mcts" and args.budget > 0
        else None
    )

    try:
        with JsonlEmulatorBackend(
            repo_root=args.repo_root,
            build=not args.no_build,
        ) as backend:
            backend.reset_operation_profile()
            for index in range(args.seeds):
                agent: Agent | ExactStateAgent
                if args.agent == "random":
                    agent = RandomAgent(seed=args.agent_seed + index)
                elif args.agent == "heuristic" or args.budget == 0:
                    agent = HeuristicAgent()
                else:
                    search = UctMcts(
                        backend,
                        policy=policy,
                        rollout_policy=HeuristicAgent(),
                        seed=args.agent_seed + index,
                    )

                    def sink(
                        result: SearchResult,
                        chosen_action: object,
                    ) -> None:
                        if strategy_store is None:
                            return
                        from sts2_ai.emulator import LegalAction

                        if not isinstance(chosen_action, LegalAction):
                            raise TypeError("Search result sink requires a LegalAction")
                        record_search_result(
                            strategy_store,
                            result,
                            chosen_action,
                            information_policy=policy.policy_id,
                            search_regime="oracle-exact",
                            search_budget=args.budget,
                            emulator_revision=backend.emulator_revision,
                            game_build=args.game_build,
                        )

                    agent = OracleMctsAgent(
                        search,
                        simulations=args.budget,
                        result_sink=sink,
                    )

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
            bridge_profile = dict(backend.operation_profile())
    finally:
        if strategy_store is not None:
            strategy_store.close()

    wins = sum(summary.won for summary in summaries)
    avg_progress = statistics.fmean(summary.terminal_progress for summary in summaries)
    avg_decisions = statistics.fmean(summary.decisions for summary in summaries)
    total_decisions = sum(summary.decisions for summary in summaries)
    total_transitions = sum(summary.emulator_transitions for summary in summaries)
    transitions_per_decision = (
        total_transitions / total_decisions if total_decisions else 0.0
    )
    avg_time = statistics.fmean(summary.wall_seconds for summary in summaries)
    avg_agent_time = statistics.fmean(summary.agent_compute_seconds for summary in summaries)
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
    print(f"Average agent compute/run: {avg_agent_time:.3f}s")

    if args.profile:
        _print_bridge_profile(bridge_profile)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent": label,
            "budget": args.budget if args.agent == "mcts" else None,
            "runs": [asdict(summary) for summary in summaries],
            "bridge_profile": {
                operation: asdict(stats)
                for operation, stats in bridge_profile.items()
            },
        }
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _print_bridge_profile(profile: dict[str, object]) -> None:
    from sts2_ai.emulator import BridgeOperationStats

    print()
    print("| Bridge op | Calls | Total time | Mean time |")
    print("| --- | ---: | ---: | ---: |")
    for operation, raw_stats in profile.items():
        if not isinstance(raw_stats, BridgeOperationStats):
            continue
        print(
            f"| {operation} | {raw_stats.calls} | "
            f"{raw_stats.total_seconds:.3f}s | {1000.0 * raw_stats.mean_seconds:.3f}ms |"
        )


if __name__ == "__main__":
    main()
