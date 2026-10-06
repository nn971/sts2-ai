from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from sts2_ai.emulator import PrototypeJsonlBackend
from sts2_ai.evaluation import (
    PrototypeRandomRunEvaluator,
    PrototypeSearchDecision,
    PrototypeSearchRunEvaluator,
    collect_experiment_manifest,
)
from sts2_ai.scenarios import (
    mine_close_search_scenarios,
    replay_scenario,
    write_replay_scenario_archive,
)
from sts2_ai.search import PrototypeFlatRolloutSearch, SearchBudget
from sts2_ai.strategy_db import SQLiteStrategyStore


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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

    search_smoke = sub.add_parser(
        "prototype-search-smoke",
        help="run the first flat-rollout workload against the pinned prototype emulator",
    )
    search_smoke.add_argument("--repo-root", type=Path, default=Path.cwd())
    search_smoke.add_argument("--seed", default="search-smoke")
    search_smoke.add_argument("--nodes", type=int, default=256)
    search_smoke.add_argument("--depth", type=int, default=32)
    search_smoke.add_argument("--search-seed", type=int, default=0)
    search_smoke.add_argument("--manifest-output", type=Path)
    search_smoke.add_argument(
        "--experiment-id",
        default="prototype-search-smoke",
    )

    compare = sub.add_parser(
        "prototype-compare",
        help="compare rollout search with a deterministic random baseline on identical seeds",
    )
    compare.add_argument("--repo-root", type=Path, default=Path.cwd())
    compare.add_argument("--seed-prefix", default="comparison")
    compare.add_argument("--runs", type=int, default=20)
    compare.add_argument("--nodes-per-decision", type=int, default=128)
    compare.add_argument("--depth", type=int, default=32)
    compare.add_argument("--batch-size", type=int, default=16)
    compare.add_argument("--search-seed", type=int, default=0)
    compare.add_argument("--random-seed", type=int, default=0)
    compare.add_argument("--max-decisions", type=int, default=5000)

    evaluate = sub.add_parser(
        "prototype-evaluate",
        help="evaluate search-guided complete prototype runs over deterministic seeds",
    )
    evaluate.add_argument("--repo-root", type=Path, default=Path.cwd())
    evaluate.add_argument("--seed-prefix", default="evaluation")
    evaluate.add_argument("--runs", type=int, default=10)
    evaluate.add_argument("--nodes-per-decision", type=int, default=128)
    evaluate.add_argument("--depth", type=int, default=32)
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--search-seed", type=int, default=0)
    evaluate.add_argument("--max-decisions", type=int, default=5000)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--manifest-output", type=Path)
    evaluate.add_argument(
        "--scenario-output",
        type=Path,
        help="optional replayable archive of close/high-uncertainty searched decisions",
    )
    evaluate.add_argument("--scenario-limit", type=int, default=20)
    evaluate.add_argument(
        "--strategy-db",
        type=Path,
        help="optional SQLite exact-state search cache",
    )
    evaluate.add_argument(
        "--experiment-id",
        default="prototype-search-evaluation",
    )

    args = parser.parse_args()
    if args.command == "manifest":
        manifest_result = collect_experiment_manifest(
            repo_root=args.repo_root,
            experiment_id=args.experiment_id,
            game_build=args.game_build,
            emulator_schema_version=args.schema_version,
            binding_version=args.binding_version,
            information_policy=args.information_policy,
            config={},
            seeds={},
        )
        manifest_result.write_json(args.output)
        print(args.output)
    elif args.command == "prototype-search-smoke":
        if args.nodes <= 0:
            parser.error("--nodes must be positive")
        if args.depth <= 0:
            parser.error("--depth must be positive")

        with PrototypeJsonlBackend.from_repo(args.repo_root) as backend:
            state = backend.reset(args.seed)
            start_actions = backend.legal_actions(state)
            if len(start_actions) != 1 or start_actions[0].kind != "start_run":
                raise RuntimeError("Prototype reset did not expose exactly one start_run action")
            state = backend.step(state, start_actions[0]).child

            search = PrototypeFlatRolloutSearch(
                backend,
                backend.fair_policy,
                seed=args.search_seed,
                rollout_depth=args.depth,
            )
            started = time.perf_counter()
            search_result = search.search(state, SearchBudget(max_nodes=args.nodes))
            elapsed = time.perf_counter() - started

            print(f"Emulator revision: {backend.emulator_revision}")
            print(f"Binding: {backend.binding_version}")
            print(f"Ruleset: {backend.ruleset_id}")
            print(f"Search: {search_result.search_version}")
            print(f"Expanded nodes: {search_result.expanded_nodes}")
            print(f"Elapsed: {elapsed:.3f}s")
            if elapsed > 0:
                print(f"Nodes/sec: {search_result.expanded_nodes / elapsed:.1f}")

            print("Root actions:")
            for evaluation in sorted(
                search_result.evaluations,
                key=lambda item: item.value,
                reverse=True,
            ):
                uncertainty = (
                    "n/a"
                    if evaluation.uncertainty is None
                    else f"{evaluation.uncertainty:.3f}"
                )
                print(
                    f"  {evaluation.action.kind} "
                    f"value={evaluation.value:.3f} "
                    f"visits={evaluation.visits} "
                    f"stderr={uncertainty} "
                    f"id={evaluation.action.action_id}"
                )

            if args.manifest_output is not None:
                search_manifest = collect_experiment_manifest(
                    repo_root=args.repo_root,
                    experiment_id=args.experiment_id,
                    game_build="prototype-unbound",
                    emulator_schema_version=PrototypeJsonlBackend.EXPECTED_AI_SCHEMA,
                    binding_version=backend.binding_version,
                    information_policy=backend.fair_policy.policy_id,
                    config={
                        "search_version": search_result.search_version,
                        "max_nodes": args.nodes,
                        "rollout_depth": args.depth,
                        "run_seed": args.seed,
                    },
                    seeds={"search_seed": args.search_seed},
                )
                search_manifest.write_json(args.manifest_output)
                print(f"Manifest: {args.manifest_output}")
    elif args.command == "prototype-compare":
        if args.runs <= 0:
            parser.error("--runs must be positive")
        if args.nodes_per_decision <= 0:
            parser.error("--nodes-per-decision must be positive")
        if args.depth <= 0:
            parser.error("--depth must be positive")
        if args.batch_size <= 0:
            parser.error("--batch-size must be positive")
        if args.max_decisions <= 0:
            parser.error("--max-decisions must be positive")

        seeds = tuple(f"{args.seed_prefix}-{index}" for index in range(args.runs))

        with PrototypeJsonlBackend.from_repo(args.repo_root) as backend:
            random_summary = PrototypeRandomRunEvaluator(
                backend,
                backend.fair_policy,
                random_seed=args.random_seed,
                max_decisions=args.max_decisions,
            ).evaluate(seeds)

            search_summary = PrototypeSearchRunEvaluator(
                backend,
                backend.fair_policy,
                nodes_per_decision=args.nodes_per_decision,
                rollout_depth=args.depth,
                rollout_batch_size=args.batch_size,
                search_seed=args.search_seed,
                max_decisions=args.max_decisions,
            ).evaluate(seeds)

            delta = search_summary.victory_rate - random_summary.victory_rate
            print(f"Emulator revision: {backend.emulator_revision}")
            print(f"Runs: {len(seeds)}")
            print(
                f"Random: victory_rate={random_summary.victory_rate:.3f} "
                f"decisions={random_summary.total_decisions} "
                f"elapsed={random_summary.elapsed_seconds:.3f}s"
            )
            print(
                f"Search: victory_rate={search_summary.victory_rate:.3f} "
                f"decisions={search_summary.total_decisions} "
                f"search_decisions={search_summary.total_search_decisions} "
                f"expanded_nodes={search_summary.total_expanded_nodes} "
                f"elapsed={search_summary.elapsed_seconds:.3f}s"
            )
            print(f"Victory-rate delta (search-random): {delta:+.3f}")
            print(
                "Exact decision-state recurrence: "
                f"{search_summary.repeated_search_states}/"
                f"{search_summary.unique_search_states + search_summary.repeated_search_states} "
                f"({search_summary.exact_state_recurrence_rate:.3f})"
            )
            print(f"Search nodes/sec: {search_summary.nodes_per_second:.1f}")
    elif args.command == "prototype-evaluate":
        if args.runs <= 0:
            parser.error("--runs must be positive")
        if args.nodes_per_decision <= 0:
            parser.error("--nodes-per-decision must be positive")
        if args.depth <= 0:
            parser.error("--depth must be positive")
        if args.batch_size <= 0:
            parser.error("--batch-size must be positive")
        if args.max_decisions <= 0:
            parser.error("--max-decisions must be positive")
        if args.scenario_limit < 0:
            parser.error("--scenario-limit cannot be negative")

        seeds = tuple(f"{args.seed_prefix}-{index}" for index in range(args.runs))
        search_decisions: list[PrototypeSearchDecision] = []

        with PrototypeJsonlBackend.from_repo(args.repo_root) as backend:
            strategy_store = (
                SQLiteStrategyStore(args.strategy_db)
                if args.strategy_db is not None
                else None
            )
            try:
                evaluator = PrototypeSearchRunEvaluator(
                    backend,
                    backend.fair_policy,
                    nodes_per_decision=args.nodes_per_decision,
                    rollout_depth=args.depth,
                    rollout_batch_size=args.batch_size,
                    search_seed=args.search_seed,
                    max_decisions=args.max_decisions,
                    strategy_store=strategy_store,
                    on_search_decision=(
                        search_decisions.append
                        if args.scenario_output is not None
                        else None
                    ),
                )
                summary = evaluator.evaluate(seeds)
            finally:
                if strategy_store is not None:
                    strategy_store.close()

            print(f"Emulator revision: {backend.emulator_revision}")
            print(f"Binding: {backend.binding_version}")
            print(f"Ruleset: {backend.ruleset_id}")
            print(f"Search: {summary.search_version}")
            print(f"Runs: {len(summary.runs)}")
            print(
                f"Outcomes: {summary.victories} victory / "
                f"{summary.defeats} defeat / "
                f"{summary.unknown_terminal_outcomes} unknown"
            )
            print(f"Victory rate: {summary.victory_rate:.3f}")
            print(f"Decisions: {summary.total_decisions}")
            print(f"Search decisions: {summary.total_search_decisions}")
            print(f"Expanded nodes: {summary.total_expanded_nodes}")
            print(f"Search cache hits: {summary.total_cache_hits}")
            print(f"Search cache hit rate: {summary.cache_hit_rate:.3f}")
            print(
                "Exact decision-state recurrence: "
                f"{summary.repeated_search_states}/"
                f"{summary.unique_search_states + summary.repeated_search_states} "
                f"({summary.exact_state_recurrence_rate:.3f})"
            )
            print(f"Elapsed: {summary.elapsed_seconds:.3f}s")
            print(f"Decisions/sec: {summary.decisions_per_second:.1f}")
            print(f"Search nodes/sec: {summary.nodes_per_second:.1f}")

            if args.output is not None:
                summary.write_json(args.output)
                print(f"Results: {args.output}")

            if args.scenario_output is not None:
                scenarios = mine_close_search_scenarios(
                    search_decisions,
                    limit=args.scenario_limit,
                    emulator_revision=backend.emulator_revision,
                    information_policy=backend.fair_policy.policy_id,
                )
                for scenario in scenarios:
                    replayed = replay_scenario(backend, scenario)
                    backend.release_many((replayed,))
                write_replay_scenario_archive(args.scenario_output, scenarios)
                print(
                    f"Scenarios: {args.scenario_output} "
                    f"({len(scenarios)} verified)"
                )

            if args.manifest_output is not None:
                evaluation_manifest = collect_experiment_manifest(
                    repo_root=args.repo_root,
                    experiment_id=args.experiment_id,
                    game_build="prototype-unbound",
                    emulator_schema_version=PrototypeJsonlBackend.EXPECTED_AI_SCHEMA,
                    binding_version=backend.binding_version,
                    information_policy=backend.fair_policy.policy_id,
                    config={
                        "evaluation": "prototype-search-runs-v1",
                        "search_version": summary.search_version,
                        "runs": args.runs,
                        "seed_prefix": args.seed_prefix,
                        "nodes_per_decision": args.nodes_per_decision,
                        "rollout_depth": args.depth,
                        "rollout_batch_size": args.batch_size,
                        "max_decisions": args.max_decisions,
                        "strategy_db_path": (
                            str(args.strategy_db)
                            if args.strategy_db is not None
                            else None
                        ),
                        "scenario_output": (
                            str(args.scenario_output)
                            if args.scenario_output is not None
                            else None
                        ),
                        "scenario_limit": args.scenario_limit,
                    },
                    seeds={"search_seed": args.search_seed},
                    strategy_db_snapshot=(
                        f"sha256:{_sha256_file(args.strategy_db)}"
                        if args.strategy_db is not None
                        else None
                    ),
                )
                evaluation_manifest.write_json(args.manifest_output)
                print(f"Manifest: {args.manifest_output}")


if __name__ == "__main__":
    main()
