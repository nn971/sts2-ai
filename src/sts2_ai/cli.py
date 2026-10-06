from __future__ import annotations

import argparse
from pathlib import Path

from sts2_ai.emulator import PrototypeJsonlBackend
from sts2_ai.evaluation import PrototypeHeuristicEvaluator, collect_experiment_manifest
from sts2_ai.search import BestFirstSearch, SearchBudget


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

    oracle = sub.add_parser(
        "oracle-search",
        help="rank the first branching prototype decision with exact-state lookahead",
    )
    oracle.add_argument("--repo-root", type=Path, default=Path.cwd())
    oracle.add_argument("--seed", default="oracle-search-demo")
    oracle.add_argument("--nodes", type=int, default=64)
    oracle.add_argument("--depth", type=int, default=3)

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

    if args.command == "oracle-search":
        _run_oracle_search(
            repo_root=args.repo_root,
            seed=args.seed,
            nodes=args.nodes,
            depth=args.depth,
        )


def _run_oracle_search(
    *,
    repo_root: Path,
    seed: str,
    nodes: int,
    depth: int,
) -> None:
    with PrototypeJsonlBackend.from_repo(repo_root) as backend:
        state = backend.reset(seed)

        forced_steps = 0
        while not backend.is_terminal(state):
            legal = tuple(backend.legal_actions(state))
            if len(legal) != 1:
                break
            state = backend.step(state, legal[0]).child
            forced_steps += 1
            if forced_steps >= 100:
                raise RuntimeError("Exceeded 100 forced actions before reaching a branching state")

        if backend.is_terminal(state):
            print("Run reached terminal state before a branching decision.")
            return

        legal = tuple(backend.legal_actions(state))
        if len(legal) < 2:
            raise RuntimeError("Expected a branching state after advancing forced actions")

        search = BestFirstSearch(
            backend=backend,
            evaluator=PrototypeHeuristicEvaluator(),
            information_policy=backend.fair_policy,
        )
        result = search.search(
            state,
            SearchBudget(
                max_nodes=nodes,
                max_depth=depth,
            ),
        )

        print("Mode: oracle-exact-state-search-v0")
        print(
            "Warning: branch transitions condition on the emulator's exact hidden state; "
            "this is not a fair-agent result."
        )
        print(f"Seed: {seed}")
        print(f"Emulator revision: {backend.emulator_revision}")
        print(f"Binding: {backend.binding_version}")
        print(f"Root exact hash: {result.root_state_hash}")
        print(f"Expanded nodes: {result.expanded_nodes}")
        print("Ranked actions:")

        ranked = sorted(
            result.evaluations,
            key=lambda item: (-item.value, item.action.action_id),
        )
        for rank, evaluation in enumerate(ranked, start=1):
            print(
                f"  {rank}. value={evaluation.value:.3f} "
                f"visits={evaluation.visits} kind={evaluation.action.kind} "
                f"payload={evaluation.action.payload_json}"
            )


if __name__ == "__main__":
    main()
