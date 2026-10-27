"""Stochastic paired snapshot evaluator for testing sampling-vs-greedy mismatch.

Use identical initial emulator RNG and matched action-selection uniforms per
(snapshot, replicate) for two policies. Model trajectories may diverge; the
pairing reduces variance but is NOT an exact matched-random-outcome coupling.
Results remain grouped by originating source seed for interval estimation.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from sts2_ai.emulator import JsonlEmulatorBackend
from sts2_ai.evaluation.combat_snapshots import (
    CombatSnapshotRecipe,
    build_report,
    evaluate_model_suite,
    load_recipes,
    pair_model_suite,
)
from sts2_ai.evaluation.stochastic_combat import (
    StochasticCombatAgent,
    snapshot_actor_seed,
)
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from snapshot_eval_runtime import source_groups

_WORKER_BACKEND = None
_WORKER_BASE = None
_WORKER_CANDIDATE = None


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _eval_group(
    backend, recipes: list[CombatSnapshotRecipe], base, candidate,
    *, temperature: float, repeats: int, max_decisions: int,
):
    models = {"baseline": base, "candidate": candidate}
    # evaluate_model_suite calls factory per snapshot, so model state/RNG never
    # leaks from one entry or one stochastic replicate into another.
    agents = {
        name: StochasticCombatAgent(model, seed=0, temperature=temperature)
        for name, model in models.items()
    }
    results = []
    for trial in range(repeats):
        suite = evaluate_model_suite(
            backend, recipes, agents, max_combat_decisions=max_decisions,
            agent_factory=lambda name, recipe: StochasticCombatAgent(
                models[name], seed=snapshot_actor_seed(recipe, trial),
                temperature=temperature,
            ),
        )
        results.extend(pair_model_suite(suite, "baseline", "candidate"))
    return results


def _worker_init(repo_root, emulator_root, revision, base_path, candidate_path):
    global _WORKER_BACKEND, _WORKER_BASE, _WORKER_CANDIDATE
    _WORKER_BACKEND = JsonlEmulatorBackend(
        repo_root=Path(repo_root), emulator_root=Path(emulator_root),
        expected_emulator_revision=revision, build=False,
    )
    atexit.register(_WORKER_BACKEND.close)
    _WORKER_BASE = PhaseSplitNeuralModel.load(Path(base_path))
    _WORKER_CANDIDATE = PhaseSplitNeuralModel.load(Path(candidate_path))


def _worker_group(payload):
    key, recipes, temperature, repeats, max_decisions = payload
    if _WORKER_BACKEND is None or _WORKER_BASE is None or _WORKER_CANDIDATE is None:
        raise RuntimeError("Stochastic snapshot worker not initialized")
    return key, _eval_group(
        _WORKER_BACKEND, recipes, _WORKER_BASE, _WORKER_CANDIDATE,
        temperature=temperature, repeats=repeats, max_decisions=max_decisions,
    )


def _atomic(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def evaluate_stochastic(
    *, corpus: Path, baseline_model: Path, candidate_model: Path,
    report_path: Path, repeats: int = 3, temperature: float = 0.7,
    workers: int = 1, max_decisions: int = 256,
    source_limit: int | None = None, build: bool = False,
):
    if repeats <= 0 or workers <= 0 or max_decisions <= 0:
        raise ValueError("Stochastic evaluation sizes must be positive")
    if source_limit is not None and source_limit <= 0:
        raise ValueError("source_limit must be positive")
    if report_path.exists():
        raise FileExistsError(report_path)
    recipes = load_recipes(corpus)
    if not recipes:
        raise ValueError("No snapshots to evaluate")
    groups = source_groups(recipes)
    if source_limit is not None:
        groups = dict(list(groups.items())[:source_limit])
    revisions = {r.emulator_revision for items in groups.values() for r in items}
    if len(revisions) != 1:
        raise ValueError("Mixed emulator revisions")
    revision = next(iter(revisions))
    done = {}
    with JsonlEmulatorBackend(build=build) as backend:
        if backend.emulator_revision != revision:
            raise ValueError("Stochastic snapshot emulator mismatch")
        if workers == 1:
            base = PhaseSplitNeuralModel.load(baseline_model)
            candidate = PhaseSplitNeuralModel.load(candidate_model)
            for key, items in groups.items():
                done[key] = _eval_group(
                    backend, items, base, candidate,
                    temperature=temperature, repeats=repeats,
                    max_decisions=max_decisions,
                )
                _log(f"[v15p stochastic] source={key[0]} done={len(done)}/{len(groups)} "
                     f"entries={len(items)} replicates={repeats}")
        else:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn"),
                initializer=_worker_init,
                initargs=(
                    str(backend._repo_root), str(backend._emulator_root),
                    revision, str(baseline_model.resolve()),
                    str(candidate_model.resolve()),
                ),
            ) as pool:
                future_to_key = {
                    pool.submit(_worker_group, (
                        key, items, temperature, repeats, max_decisions,
                    )): key for key, items in groups.items()
                }
                for future in as_completed(future_to_key):
                    key, result = future.result()
                    if key in done:
                        raise ValueError("Repeated stochastic source group")
                    done[key] = result
                    _log(
                        f"[v15p stochastic] source={key[0]} "
                        f"done={len(done)}/{len(groups)} "
                        f"replicates={repeats}"
                    )
    rows = [r for key in groups for r in done[key]]
    report = build_report(
        rows, emulator_revision=revision, corpus=str(corpus),
        baseline_model=str(baseline_model),
        candidate_model=str(candidate_model),
    )
    report["evaluation"] = {
        "mode": "stochastic", "temperature": temperature,
        "replicates": repeats, "workers": workers,
        "source_runs": len(groups), "subset_only": source_limit is not None,
        "max_combat_decisions": max_decisions,
        "identity": {
            "corpus_sha256": _sha(corpus),
            "baseline_sha256": _sha(baseline_model),
            "candidate_sha256": _sha(candidate_model),
            "emulator_revision": revision,
            "temperature": temperature, "replicates": repeats,
            "source_limit": source_limit,
        },
    }
    _atomic(report_path, report)
    _log(
        f"[v15p stochastic] report={report_path} pairs={len(rows)} "
        f"delta_win={report['summary']['delta_win_rate']:+.4f} "
        f"delta_hp={report['summary']['delta_exit_hp_all']:+.3f}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-combat-decisions", type=int, default=256)
    parser.add_argument("--limit-source-runs", type=int)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    evaluate_stochastic(
        corpus=args.corpus, baseline_model=args.baseline_model,
        candidate_model=args.candidate_model, report_path=args.report,
        repeats=args.replicates, temperature=args.temperature,
        workers=args.workers, max_decisions=args.max_combat_decisions,
        source_limit=args.limit_source_runs, build=args.build,
    )


if __name__ == "__main__":
    main()
