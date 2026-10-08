from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from sts2_ai.agents import (
    Agent,
    ExactStateAgent,
    HeuristicAgent,
    OracleMctsAgent,
    RandomAgent,
    RoutePlanningAgent,
)
from sts2_ai.emulator import (
    FAIR_POLICY_ID,
    BridgeOperationStats,
    InformationPolicy,
    JsonlEmulatorBackend,
    LegalAction,
)
from sts2_ai.evaluation import (
    RunSummary,
    collect_experiment_manifest,
    compare_paired_runs,
    markdown_table,
    paired_markdown_table,
    play_run,
    summarize_runs,
)
from sts2_ai.search import SearchResult, UctMcts
from sts2_ai.strategy_db import (
    SQLiteStrategyStore,
    diagnose_budget_disagreements,
    record_search_result,
)
from sts2_ai.training import (
    build_training_examples,
    evaluate_hashed_linear,
    load_training_jsonl,
    split_training_examples,
    train_hashed_linear,
    write_training_jsonl,
)


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
    evaluate.add_argument(
        "--agent", choices=("random", "heuristic", "route", "mcts"), required=True
    )
    evaluate.add_argument("--budget", type=int, default=32, help="MCTS simulations per decision")
    evaluate.add_argument(
        "--rollout-depth",
        type=int,
        default=64,
        help="maximum heuristic rollout decisions after each selected MCTS leaf",
    )
    evaluate.add_argument(
        "--rollout-batch-size",
        type=int,
        default=8,
        help="number of pending UCT simulations whose heuristic rollouts are advanced together",
    )
    evaluate.add_argument(
        "--rollout-mode",
        choices=("fixed", "combat-exit"),
        default="fixed",
        help=(
            "fixed uses the decision cap uniformly; combat-exit stops rollouts "
            "that start in combat as soon as they leave combat"
        ),
    )
    evaluate.add_argument(
        "--virtual-loss",
        type=float,
        default=None,
        help=(
            "fixed temporary value for pending UCT paths; "
            "omit for mean-preserving reservations"
        ),
    )
    evaluate.add_argument(
        "--rollout-policy", choices=("heuristic", "route"), default="heuristic",
        help="MCTS rollout agent; route is an experimental visible-map baseline",
    )
    evaluate.add_argument("--route-horizon", type=int, default=6)
    evaluate.add_argument("--route-discount", type=float, default=0.8)
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

    benchmark = sub.add_parser(
        "benchmark",
        help="compare random, heuristic, and a common-seed MCTS budget ladder",
    )
    benchmark.add_argument(
        "--budgets",
        type=int,
        nargs="*",
        default=[32, 128, 512, 2048],
        help="MCTS simulations per decision (pass --budgets alone for no MCTS)",
    )
    benchmark.add_argument("--seeds", type=int, default=3)
    benchmark.add_argument("--seed-prefix", default="benchmark")
    benchmark.add_argument("--agent-seed", type=int, default=0)
    benchmark.add_argument("--ascension", type=int, default=0)
    benchmark.add_argument("--max-decisions", type=int)
    benchmark.add_argument("--rollout-depth", type=int, default=8)
    benchmark.add_argument("--rollout-batch-size", type=int, default=8)
    benchmark.add_argument(
        "--rollout-mode",
        choices=("fixed", "combat-exit"),
        default="fixed",
    )
    benchmark.add_argument("--virtual-loss", type=float)
    benchmark.add_argument(
        "--rollout-policy", choices=("heuristic", "route"), default="heuristic",
    )
    benchmark.add_argument("--route-horizon", type=int, default=6)
    benchmark.add_argument("--route-discount", type=float, default=0.8)
    benchmark.add_argument(
        "--include-route", action=argparse.BooleanOptionalAction, default=True,
        help="include visible-route planner as an independent fair baseline",
    )
    benchmark.add_argument("--repo-root", type=Path, default=Path.cwd())
    benchmark.add_argument("--no-build", action="store_true")
    benchmark.add_argument(
        "--strategy-db",
        type=Path,
        default=Path("results/strategy.sqlite"),
    )
    benchmark.add_argument("--game-build", default="unknown")
    benchmark.add_argument("--json-output", type=Path)
    benchmark.add_argument(
        "--include-random",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    report = sub.add_parser(
        "strategy-report",
        help="show exact roots where search budgets selected different actions",
    )
    report.add_argument("database", type=Path)
    report.add_argument("--limit", type=int, default=20)
    report.add_argument("--information-policy", default=FAIR_POLICY_ID)
    report.add_argument("--search-regime", default="oracle-exact")
    report.add_argument(
        "--search-version",
        help="exact persisted search configuration to report",
    )
    report.add_argument("--summary-only", action="store_true")
    report.add_argument(
        "--sort-by",
        choices=("phase", "pair-shift", "regret"),
        default="phase",
        help="ordering for detailed disagreement rows",
    )
    report.add_argument(
        "--meaningful-only",
        action="store_true",
        help="hide disagreements that collapse to the same semantic action",
    )
    report.add_argument("--json-output", type=Path)

    training_export = sub.add_parser(
        "export-training",
        help="distill persisted search evidence into policy/value JSONL targets",
    )
    training_export.add_argument("database", type=Path)
    training_export.add_argument("output", type=Path)
    training_export.add_argument("--information-policy", default=FAIR_POLICY_ID)
    training_export.add_argument("--search-regime", default="oracle-exact")
    training_export.add_argument("--search-version")
    training_export.add_argument("--min-budget", type=int, default=0)
    training_export.add_argument(
        "--all-budgets",
        action="store_true",
        help="emit every matching budget instead of the strongest root per exact state",
    )

    train_linear = sub.add_parser(
        "train-linear",
        help="train the dependency-free hashed linear policy/value baseline",
    )
    train_linear.add_argument("dataset", type=Path)
    train_linear.add_argument("output", type=Path)
    train_linear.add_argument("--dimension", type=int, default=4096)
    train_linear.add_argument("--epochs", type=int, default=8)
    train_linear.add_argument("--learning-rate", type=float, default=0.03)
    train_linear.add_argument("--l2", type=float, default=1e-6)
    train_linear.add_argument("--seed", type=int, default=0)

    validate_linear = sub.add_parser(
        "validate-linear",
        help="train and evaluate on a deterministic exact-state grouped holdout",
    )
    validate_linear.add_argument("dataset", type=Path)
    validate_linear.add_argument("output", type=Path)
    validate_linear.add_argument("--validation-fraction", type=float, default=0.2)
    validate_linear.add_argument("--dimension", type=int, default=4096)
    validate_linear.add_argument("--epochs", type=int, default=8)
    validate_linear.add_argument("--learning-rate", type=float, default=0.03)
    validate_linear.add_argument("--l2", type=float, default=1e-6)
    validate_linear.add_argument("--seed", type=int, default=0)

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
        return

    if args.command == "benchmark":
        _benchmark(args)
        return

    if args.command == "strategy-report":
        _strategy_report(args)
        return

    if args.command == "export-training":
        _export_training(args)
        return

    if args.command == "train-linear":
        _train_linear(args)
        return

    if args.command == "validate-linear":
        examples = load_training_jsonl(args.dataset)
        train, validation = split_training_examples(
            examples,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        model, training_metrics = train_hashed_linear(
            train,
            dimension=args.dimension,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            seed=args.seed,
        )
        model.save(args.output)
        validation_report = {
            "train": asdict(training_metrics),
            "validation": asdict(evaluate_hashed_linear(model, validation)),
            "train_states": len({e.source_state_hash or e.observation_hash for e in train}),
            "validation_states": len(
                {e.source_state_hash or e.observation_hash for e in validation}
            ),
            "model_path": str(args.output),
            "seed": args.seed,
        }
        print(json.dumps(validation_report, sort_keys=True, indent=2))



def _route_agent(args: argparse.Namespace) -> RoutePlanningAgent:
    return RoutePlanningAgent(horizon=args.route_horizon, discount=args.route_discount)


def _rollout_agent(args: argparse.Namespace) -> HeuristicAgent | RoutePlanningAgent:
    return _route_agent(args) if args.rollout_policy == "route" else HeuristicAgent()


def _check_route_args(args: argparse.Namespace) -> None:
    if args.route_horizon <= 0:
        raise SystemExit("--route-horizon must be positive")
    if not 0.0 < args.route_discount <= 1.0:
        raise SystemExit("--route-discount must lie in (0, 1]")


def _evaluate(args: argparse.Namespace) -> None:
    _check_route_args(args)
    if args.seeds <= 0:
        raise SystemExit("--seeds must be positive")
    if args.budget < 0:
        raise SystemExit("--budget must be non-negative")
    if args.rollout_depth < 0:
        raise SystemExit("--rollout-depth must be non-negative")
    if args.rollout_batch_size <= 0:
        raise SystemExit("--rollout-batch-size must be positive")
    if args.virtual_loss is not None and not -1.0 <= args.virtual_loss <= 1.0:
        raise SystemExit("--virtual-loss must lie in [-1, 1]")

    policy = InformationPolicy(FAIR_POLICY_ID)
    summaries = []
    bridge_profile: Mapping[str, BridgeOperationStats] = {}
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
                elif args.agent == "route":
                    agent = _route_agent(args)
                elif args.agent == "heuristic" or args.budget == 0:
                    agent = HeuristicAgent()
                else:
                    search = UctMcts(
                        backend,
                        policy=policy,
                        rollout_policy=_rollout_agent(args),
                        rollout_depth=args.rollout_depth,
                        rollout_batch_size=args.rollout_batch_size,
                        rollout_mode=args.rollout_mode,
                        virtual_loss=args.virtual_loss,
                        seed=args.agent_seed + index,
                    )

                    def sink(
                        result: SearchResult,
                        chosen_action: LegalAction,
                    ) -> None:
                        if strategy_store is None:
                            return
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
            "rollout_depth": args.rollout_depth if args.agent == "mcts" else None,
            "rollout_batch_size": (
                args.rollout_batch_size if args.agent == "mcts" else None
            ),
            "rollout_mode": args.rollout_mode if args.agent == "mcts" else None,
            "rollout_policy": args.rollout_policy if args.agent == "mcts" else None,
            "route_horizon": args.route_horizon,
            "route_discount": args.route_discount,
            "virtual_loss": args.virtual_loss if args.agent == "mcts" else None,
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


def _print_bridge_profile(profile: Mapping[str, BridgeOperationStats]) -> None:
    print()
    print("| Bridge op | Calls | Total time | Mean time |")
    print("| --- | ---: | ---: | ---: |")
    for operation, stats in profile.items():
        print(
            f"| {operation} | {stats.calls} | "
            f"{stats.total_seconds:.3f}s | {1000.0 * stats.mean_seconds:.3f}ms |"
        )


def _benchmark(args: argparse.Namespace) -> None:
    _check_route_args(args)
    if args.seeds <= 0:
        raise SystemExit("--seeds must be positive")
    if any(budget <= 0 for budget in args.budgets):
        raise SystemExit("--budgets entries must be positive")
    if len(set(args.budgets)) != len(args.budgets):
        raise SystemExit("--budgets entries must be unique")
    if args.rollout_depth < 0:
        raise SystemExit("--rollout-depth must be non-negative")
    if args.rollout_batch_size <= 0:
        raise SystemExit("--rollout-batch-size must be positive")
    if args.virtual_loss is not None and not -1.0 <= args.virtual_loss <= 1.0:
        raise SystemExit("--virtual-loss must lie in [-1, 1]")

    budgets = sorted(args.budgets)
    policy = InformationPolicy(FAIR_POLICY_ID)
    rows = []
    summaries_by_label: dict[str, list[RunSummary]] = {}
    serialized_runs: dict[str, list[dict[str, object]]] = {}

    with SQLiteStrategyStore(args.strategy_db) as strategy_store:
        with JsonlEmulatorBackend(
            repo_root=args.repo_root,
            build=not args.no_build,
        ) as backend:
            labels: list[tuple[str, int | None]] = []
            if args.include_random:
                labels.append(("random", None))
            labels.append(("heuristic", 0))
            if args.include_route:
                labels.append(("route", None))
            labels.extend((f"MCTS-{budget}", budget) for budget in budgets)

            for label, budget in labels:
                summaries = []
                for index in range(args.seeds):
                    agent: Agent | ExactStateAgent
                    if label == "random":
                        agent = RandomAgent(seed=args.agent_seed + index)
                    elif label == "route":
                        agent = _route_agent(args)
                    elif budget == 0:
                        agent = HeuristicAgent()
                    else:
                        assert budget is not None
                        search = UctMcts(
                            backend,
                            policy=policy,
                            rollout_policy=_rollout_agent(args),
                            rollout_depth=args.rollout_depth,
                            rollout_batch_size=args.rollout_batch_size,
                            rollout_mode=args.rollout_mode,
                            virtual_loss=args.virtual_loss,
                            seed=args.agent_seed + index,
                        )

                        def sink(
                            result: SearchResult,
                            chosen_action: LegalAction,
                            *,
                            search_budget: int = budget,
                        ) -> None:
                            record_search_result(
                                strategy_store,
                                result,
                                chosen_action,
                                information_policy=policy.policy_id,
                                search_regime="oracle-exact",
                                search_budget=search_budget,
                                emulator_revision=backend.emulator_revision,
                                game_build=args.game_build,
                            )

                        agent = OracleMctsAgent(
                            search,
                            simulations=budget,
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

                rows.append(summarize_runs(label, summaries))
                summaries_by_label[label] = summaries
                serialized_runs[label] = [
                    asdict(summary) for summary in summaries
                ]

    paired = []
    if args.include_route:
        paired.append(
            compare_paired_runs(
                "heuristic",
                summaries_by_label["heuristic"],
                "route",
                summaries_by_label["route"],
            )
        )
    paired.extend([
        compare_paired_runs(
            "heuristic",
            summaries_by_label["heuristic"],
            f"MCTS-{budget}",
            summaries_by_label[f"MCTS-{budget}"],
        )
        for budget in budgets
    ])

    print(markdown_table(rows))
    print()
    for row in rows:
        print(
            f"{row.label}: avg decisions/run={row.average_decisions:.1f}, "
            f"agent compute/run={row.average_agent_compute_seconds:.3f}s"
        )

    print()
    print("Paired common-seed comparison against heuristic:")
    print(paired_markdown_table(paired))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "budgets": budgets,
            "seeds": args.seeds,
            "seed_prefix": args.seed_prefix,
            "rollout_depth": args.rollout_depth,
            "rollout_batch_size": args.rollout_batch_size,
            "rollout_mode": args.rollout_mode,
            "rollout_policy": args.rollout_policy,
            "route_horizon": args.route_horizon,
            "route_discount": args.route_discount,
            "include_route": args.include_route,
            "virtual_loss": args.virtual_loss,
            "rows": [asdict(row) for row in rows],
            "paired_vs_heuristic": [asdict(row) for row in paired],
            "runs": serialized_runs,
        }
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _strategy_report(args: argparse.Namespace) -> None:
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")

    with SQLiteStrategyStore(args.database) as store:
        versions = store.search_versions(
            args.information_policy,
            search_regime=args.search_regime,
        )
        selected_version = args.search_version
        if selected_version is None:
            if len(versions) > 1:
                available = "\n".join(f"  {version}" for version in versions)
                raise SystemExit(
                    "The database contains multiple search configurations. "
                    "Choose one with --search-version:\n"
                    f"{available}"
                )
            selected_version = versions[0] if versions else None
        elif selected_version not in versions:
            raise SystemExit(
                f"Search configuration {selected_version!r} is absent from the database."
            )

        selection_health = store.root_selection_visit_counts(
            args.information_policy,
            search_regime=args.search_regime,
            search_version=selected_version,
        )
        rollout_horizon = store.rollout_horizon_counts(
            args.information_policy,
            search_regime=args.search_regime,
            search_version=selected_version,
        )
        report = diagnose_budget_disagreements(
            store,
            args.information_policy,
            search_regime=args.search_regime,
            search_version=selected_version,
        )

    if selected_version is not None:
        print(f"Search configuration: {selected_version}")
        print()

    if selection_health:
        print("| Budget | Searched roots | Chosen with zero visits | Fraction |")
        print("| ---: | ---: | ---: | ---: |")
        for budget, roots, unvisited in selection_health:
            fraction = unvisited / roots if roots else 0.0
            print(
                f"| {budget} | {roots} | {unvisited} | {100.0 * fraction:.1f}% |"
            )
        print()

    if rollout_horizon and any(row[1] > 0 for row in rollout_horizon):
        print(
            "| Budget | Rollouts | Terminal | Combat exit | Cutoff | "
            "Resolved % | Avg rollout steps |"
        )
        print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for (
            budget,
            rollouts,
            terminals,
            boundaries,
            cutoffs,
            steps,
        ) in rollout_horizon:
            resolved_fraction = (
                (terminals + boundaries) / rollouts if rollouts else 0.0
            )
            average_steps = steps / rollouts if rollouts else 0.0
            print(
                f"| {budget} | {rollouts} | {terminals} | {boundaries} | "
                f"{cutoffs} | {100.0 * resolved_fraction:.1f}% | "
                f"{average_steps:.2f} |"
            )
        print()

    if report.total_roots == 0:
        print("No multi-budget action disagreements found.")
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return

    print(f"Multi-budget disagreement roots: {report.total_roots}")
    print()
    print(
        "| Phase | Roots | Unvisited selection | Semantic equivalent | "
        "Value-ranking flip | Visit-selection | Mixed | Other | "
        "Mean max selection regret |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for summary in report.by_phase:
        print(
            f"| {summary.phase} | {summary.roots} | "
            f"{summary.unvisited_selection} | {summary.semantic_equivalent} | "
            f"{summary.value_ranking_flips} | {summary.visit_selection} | "
            f"{summary.mixed} | {summary.other} | "
            f"{summary.mean_max_selection_regret:.4f} |"
        )

    if not args.summary_only:
        detail_roots = [
            root
            for root in report.roots
            if not args.meaningful_only or root.meaningful
        ]
        if args.sort_by == "pair-shift":
            detail_roots.sort(
                key=lambda root: (
                    root.pair_delta_shift,
                    root.max_selection_regret,
                    root.state_hash,
                ),
                reverse=True,
            )
        elif args.sort_by == "regret":
            detail_roots.sort(
                key=lambda root: (
                    root.max_selection_regret,
                    root.pair_delta_shift,
                    root.state_hash,
                ),
                reverse=True,
            )

        print()
        if args.meaningful_only:
            print(
                f"Displaying {min(args.limit, len(detail_roots))} of "
                f"{len(detail_roots)} semantically meaningful disagreements."
            )
            print()
        print(
            "| State | Phase | Act/Floor | HP | Class | Pair shift | Budget | "
            "Chosen action | Chosen mean | Best-mean action | Best mean | Regret | Visits |"
        )
        print(
            "| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | --- | "
            "---: | ---: | ---: |"
        )
        for root in detail_roots[: args.limit]:
            location = f"{root.act}/{root.floor}"
            hp = (
                f"{root.hp}/{root.max_hp}"
                if root.hp is not None and root.max_hp is not None
                else "?"
            )
            for decision in root.decisions:
                print(
                    f"| {root.state_hash[:10]} | {root.phase} | {location} | {hp} | "
                    f"{root.classification} | {root.pair_delta_shift:.4f} | "
                    f"{decision.budget} | "
                    f"{decision.chosen_semantic_signature} | "
                    f"{decision.chosen_value:.4f} | "
                    f"{decision.best_mean_semantic_signature} | "
                    f"{decision.best_mean_value:.4f} | "
                    f"{decision.selection_regret:.4f} | "
                    f"{decision.chosen_visits} |"
                )
            low_delta = (
                "?"
                if root.low_budget_pair_delta is None
                else f"{root.low_budget_pair_delta:+.4f}"
            )
            high_delta = (
                "?"
                if root.high_budget_pair_delta is None
                else f"{root.high_budget_pair_delta:+.4f}"
            )
            print(
                f"<!-- pair-delta low={low_delta} high={high_delta}; "
                f"low={_short_action_id(root.low_budget_action)}; "
                f"high={_short_action_id(root.high_budget_action)} -->"
            )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _export_training(args: argparse.Namespace) -> None:
    if args.min_budget < 0:
        raise SystemExit("--min-budget must be non-negative")

    with SQLiteStrategyStore(args.database) as store:
        versions = store.search_versions(
            args.information_policy,
            search_regime=args.search_regime,
        )
        selected_version = args.search_version
        if selected_version is None:
            if len(versions) > 1:
                available = "\n".join(f"  {version}" for version in versions)
                raise SystemExit(
                    "The database contains multiple search configurations. "
                    "Choose one with --search-version:\n"
                    f"{available}"
                )
            selected_version = versions[0] if versions else None
        elif selected_version not in versions:
            raise SystemExit(
                f"Search configuration {selected_version!r} is absent from the database."
            )

        examples = build_training_examples(
            store,
            args.information_policy,
            search_regime=args.search_regime,
            search_version=selected_version,
            min_budget=args.min_budget,
            highest_budget_only=not args.all_budgets,
        )

    count = write_training_jsonl(examples, args.output)
    print(f"Wrote {count} training examples to {args.output}")
    if selected_version is not None:
        print(f"Search configuration: {selected_version}")


def _train_linear(args: argparse.Namespace) -> None:
    examples = load_training_jsonl(args.dataset)
    if not examples:
        raise SystemExit("Training dataset is empty")

    model, metrics = train_hashed_linear(
        examples,
        dimension=args.dimension,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    model.save(args.output)

    print(f"Wrote model to {args.output}")
    print(f"Model id: {model.model_id}")
    print(f"Training examples: {metrics.examples}")
    print(f"Training policy cross-entropy: {metrics.policy_cross_entropy:.6f}")
    print(f"Training policy top-1 accuracy: {100.0 * metrics.policy_top1_accuracy:.1f}%")
    print(f"Training value RMSE: {metrics.value_rmse:.6f}")


def _short_action_id(action_id: str) -> str:
    kind, separator, digest = action_id.partition(":")
    if not separator:
        return action_id
    return f"{kind}:{digest[-8:]}"


if __name__ == "__main__":
    main()
