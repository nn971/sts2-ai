from __future__ import annotations

import argparse
import time
from pathlib import Path

from sts2_ai.emulator import PrototypeJsonlBackend
from sts2_ai.evaluation import collect_experiment_manifest
from sts2_ai.search import PrototypeFlatRolloutSearch, SearchBudget


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


if __name__ == "__main__":
    main()
