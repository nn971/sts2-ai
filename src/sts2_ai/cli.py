from __future__ import annotations

import argparse
from pathlib import Path

from sts2_ai.emulator import PrototypeJsonlBackend
from sts2_ai.evaluation import collect_experiment_manifest
from sts2_ai.search import MonteCarloRolloutSearch, SearchBudget


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

    search = sub.add_parser(
        "prototype-search",
        help="run the first rollout-search baseline against the pinned prototype emulator",
    )
    search.add_argument("seed")
    search.add_argument("--nodes", type=int, default=1_000)
    search.add_argument("--search-seed", type=int, default=0)
    search.add_argument("--emulator-root", type=Path, default=Path("emulator"))

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

    if args.command == "prototype-search":
        if args.nodes <= 0:
            parser.error("--nodes must be positive")

        with PrototypeJsonlBackend(args.emulator_root) as backend:
            state = backend.reset(args.seed)

            # Skip mechanically forced setup transitions so the search root is a real choice.
            while not backend.is_terminal(state):
                actions = tuple(backend.legal_actions(state))
                if len(actions) != 1:
                    break
                transition = backend.step(state, actions[0])
                backend.release_many([state])
                state = transition.child

            actions = tuple(backend.legal_actions(state))
            if not actions:
                raise RuntimeError("Prototype search root has no legal actions")

            result = MonteCarloRolloutSearch(
                backend,
                seed=args.search_seed,
            ).search(
                state,
                SearchBudget(max_nodes=args.nodes),
            )

            print(f"emulator_revision: {backend.emulator_revision}")
            print(f"binding_version: {backend.binding_version}")
            print(f"root_hash: {result.root_state_hash}")
            print(f"expanded_nodes: {result.expanded_nodes}")
            for evaluation in sorted(
                result.evaluations,
                key=lambda item: (-item.value, -item.visits, item.action.action_id),
            ):
                print(
                    f"{evaluation.action.kind}\t"
                    f"value={evaluation.value:.6f}\t"
                    f"visits={evaluation.visits}\t"
                    f"{evaluation.action.action_id}"
                )

            backend.release_many([state])
        return

    raise AssertionError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":
    main()
