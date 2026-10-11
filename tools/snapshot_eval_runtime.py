"""Resumable, progress-reporting native paired combat-snapshot evaluation."""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from sts2_ai.agents.phase_split_agent import PhaseSplitGreedyAgent
from sts2_ai.emulator import JsonlEmulatorBackend
from sts2_ai.evaluation.combat_snapshots import (
    CombatResult,
    CombatSnapshotRecipe,
    PairedCombatResult,
    build_report,
    evaluate_recipes,
    load_recipes,
)

JOURNAL_SCHEMA = "sts2-snapshot-paired-progress-v1"
SourceKey = tuple[str, int, str]
_WORKER_BACKEND: JsonlEmulatorBackend | None = None
_WORKER_BASELINE: PhaseSplitGreedyAgent | None = None
_WORKER_CANDIDATE: PhaseSplitGreedyAgent | None = None


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def source_groups(
    recipes: Sequence[CombatSnapshotRecipe],
) -> dict[SourceKey, list[CombatSnapshotRecipe]]:
    groups: dict[SourceKey, list[CombatSnapshotRecipe]] = defaultdict(list)
    for recipe in recipes:
        recipe.validate()
        groups[(recipe.seed, recipe.ascension, recipe.source_policy)].append(recipe)
    return {
        key: sorted(rows, key=lambda x: len(x.action_ids))
        for key, rows in sorted(groups.items())
    }


def _pair(raw: dict[str, Any]) -> PairedCombatResult:
    def combat(value: dict[str, Any]) -> CombatResult:
        return CombatResult(
            value["outcome"], value["hp_remaining"],
            tuple(value["potions_remaining"]), value["decisions"],
            value["enemy_hp_remaining"],
        )

    return PairedCombatResult(
        seed=raw["seed"], floor=raw["floor"], tier=raw["tier"],
        progress=raw["progress"], enemy_ids=tuple(raw["enemy_ids"]),
        baseline=combat(raw["baseline"]),
        candidate=combat(raw["candidate"]),
    )


def verify_source(
    key: SourceKey, recipes: Sequence[CombatSnapshotRecipe],
    results: Sequence[PairedCombatResult],
) -> None:
    if len(results) != len(recipes):
        raise ValueError("Incomplete source-run snapshot results")
    for row, recipe in zip(results, recipes, strict=True):
        if (row.seed != recipe.seed or row.floor != recipe.floor
                or row.tier != recipe.tier or row.progress != recipe.progress
                or row.enemy_ids != recipe.enemy_ids):
            raise ValueError(f"Snapshot journal recipe mismatch: {key}")
        for outcome in (row.baseline, row.candidate):
            if (outcome.outcome not in ("victory", "defeat", "censored")
                    or type(outcome.decisions) is not int
                    or outcome.decisions < 0):
                raise ValueError("Invalid snapshot journal result")


def read_journal(
    path: Path, identity: dict[str, Any],
    groups: dict[SourceKey, list[CombatSnapshotRecipe]],
) -> dict[SourceKey, list[PairedCombatResult]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        first = stream.readline()
        if not first or json.loads(first) != {
            "schema": JOURNAL_SCHEMA, "identity": identity,
        }:
            raise ValueError("Snapshot evaluation journal fingerprint mismatch")
        done: dict[SourceKey, list[PairedCombatResult]] = {}
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            source = item["source"]
            if not isinstance(source, list) or len(source) != 3:
                raise ValueError("Malformed source-run journal entry")
            key = (source[0], source[1], source[2])
            if key not in groups or key in done:
                raise ValueError("Unknown or duplicated source run in journal")
            results = [_pair(value) for value in item["rows"]]
            verify_source(key, groups[key], results)
            done[key] = results
    return done


def init_journal(path: Path, identity: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "schema": JOURNAL_SCHEMA, "identity": identity,
        }, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_source(
    path: Path, key: SourceKey, results: Sequence[PairedCombatResult],
) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "source": list(key),
            "rows": [asdict(row) for row in results],
        }, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".incomplete")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def evaluate_source(
    backend: JsonlEmulatorBackend,
    baseline: PhaseSplitGreedyAgent, candidate: PhaseSplitGreedyAgent,
    recipes: Sequence[CombatSnapshotRecipe], *,
    max_combat_decisions: int, progress_every: int,
) -> list[PairedCombatResult]:
    def started(seed: str, total: int) -> None:
        log(f"[snapshot pid={os.getpid()}] source={seed} starting "
            f"combat_entries={total}")

    def snapshot(seed: str, finished: int, total: int) -> None:
        log(f"[snapshot pid={os.getpid()}] source={seed} "
            f"entries={finished}/{total} finished")

    def active(seed: str, floor: int, model: str, decision: int) -> None:
        log(f"[snapshot pid={os.getpid()}] source={seed} floor={floor} "
            f"model={model} combat_decisions={decision}/{max_combat_decisions}")

    def replay(seed: str, floor: int, count: int, total: int) -> None:
        log(f"[snapshot pid={os.getpid()}] source={seed} floor={floor} "
            f"prefix_actions={count}/{total}")

    return evaluate_recipes(
        backend, recipes, baseline, candidate,
        max_combat_decisions=max_combat_decisions,
        on_run_start=started, on_snapshot_complete=snapshot,
        on_combat_progress=active, on_replay_progress=replay,
        combat_progress_every=progress_every,
    )


def _worker_init(
    repo_root: str, emulator_root: str, revision: str,
    baseline_path: str, candidate_path: str,
) -> None:
    global _WORKER_BACKEND, _WORKER_BASELINE, _WORKER_CANDIDATE
    _WORKER_BACKEND = JsonlEmulatorBackend(
        repo_root=Path(repo_root), emulator_root=Path(emulator_root),
        expected_emulator_revision=revision, build=False,
    )
    atexit.register(_WORKER_BACKEND.close)
    _WORKER_BASELINE = PhaseSplitGreedyAgent.load(Path(baseline_path))
    _WORKER_CANDIDATE = PhaseSplitGreedyAgent.load(Path(candidate_path))


def _worker_run(
    request: tuple[SourceKey, list[CombatSnapshotRecipe], int, int],
) -> tuple[SourceKey, list[PairedCombatResult]]:
    key, recipes, max_decisions, progress_every = request
    if (_WORKER_BACKEND is None or _WORKER_BASELINE is None
            or _WORKER_CANDIDATE is None):
        raise RuntimeError("Snapshot worker not initialized")
    return key, evaluate_source(
        _WORKER_BACKEND, _WORKER_BASELINE, _WORKER_CANDIDATE, recipes,
        max_combat_decisions=max_decisions, progress_every=progress_every,
    )


def evaluate_paired_snapshots(
    *, corpus: Path, baseline_model: Path, candidate_model: Path,
    report_path: Path, workers: int = 1, max_combat_decisions: int = 256,
    progress_every: int = 32, limit_source_runs: int | None = None,
    build: bool = False,
) -> dict[str, Any]:
    """Safe group-level checkpointing; different models can NEVER share journals."""
    if (min(workers, max_combat_decisions, progress_every) <= 0
            or limit_source_runs is not None and limit_source_runs <= 0):
        raise ValueError("Invalid worker, decision or progress parameters")
    if report_path.exists():
        raise FileExistsError(f"Completed report already exists: {report_path}")
    if not baseline_model.is_file() or not candidate_model.is_file():
        raise FileNotFoundError("Snapshot baseline/candidate model does not exist")
    recipes = load_recipes(corpus)
    if not recipes:
        raise ValueError("Empty snapshot corpus")
    groups = source_groups(recipes)
    if limit_source_runs is not None:
        groups = dict(list(groups.items())[:limit_source_runs])
    if not groups:
        raise ValueError("No source runs selected")
    revisions = {recipe.emulator_revision for group in groups.values()
                 for recipe in group}
    if len(revisions) != 1:
        raise ValueError("Mixed snapshot emulator revisions")
    revision = next(iter(revisions))
    identity = {
        "corpus_sha256": _sha(corpus),
        "baseline_sha256": _sha(baseline_model),
        "candidate_sha256": _sha(candidate_model),
        "emulator_revision": revision,
        "max_combat_decisions": max_combat_decisions,
        "sources": [list(key) for key in groups],
    }
    journal_path = Path(str(report_path) + ".partial.jsonl")
    done = read_journal(journal_path, identity, groups)
    if not journal_path.exists():
        init_journal(journal_path, identity)
    remaining = [(key, rows) for key, rows in groups.items() if key not in done]
    total_entries = sum(len(x) for x in groups.values())
    completed_entries = sum(len(rows) for rows in done.values())
    start = time.monotonic()
    log(f"[snapshot] runs={len(groups)} entries={total_entries} "
        f"recovered_runs={len(done)} recovered_entries={completed_entries} "
        f"workers={workers} journal={journal_path}")

    def completed(key: SourceKey, rows: Sequence[PairedCombatResult]) -> None:
        nonlocal completed_entries
        if key in done or key not in groups:
            raise ValueError("Duplicate source-run result")
        verify_source(key, groups[key], rows)
        append_source(journal_path, key, rows)
        done[key] = list(rows)
        completed_entries += len(rows)
        log(f"[snapshot] completed_runs={len(done)}/{len(groups)} "
            f"entries={completed_entries}/{total_entries} "
            f"elapsed_s={time.monotonic()-start:.1f} source={key[0]}")

    with JsonlEmulatorBackend(build=build) as backend:
        if backend.emulator_revision != revision:
            raise ValueError("Snapshot corpus/emulator revision mismatch")
        if workers == 1:
            baseline = PhaseSplitGreedyAgent.load(baseline_model)
            candidate = PhaseSplitGreedyAgent.load(candidate_model)
            for key, rows in remaining:
                completed(key, evaluate_source(
                    backend, baseline, candidate, rows,
                    max_combat_decisions=max_combat_decisions,
                    progress_every=progress_every,
                ))
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
                tasks = [
                    pool.submit(
                        _worker_run, (key, rows, max_combat_decisions, progress_every),
                    )
                    for key, rows in remaining
                ]
                for future in as_completed(tasks):
                    key, rows = future.result()
                    completed(key, rows)

    ordered = [row for key in groups for row in done[key]]
    report = build_report(
        ordered, emulator_revision=revision, corpus=str(corpus),
        baseline_model=str(baseline_model), candidate_model=str(candidate_model),
    )
    report["evaluation"] = {
        "source_runs": len(groups), "workers": workers,
        "subset_only": limit_source_runs is not None,
        "source_run_limit": limit_source_runs,
        "journal": str(journal_path), "identity": identity,
    }
    _atomic_report(report_path, report)
    log(f"[snapshot] DONE pairs={len(ordered)} elapsed_s={time.monotonic()-start:.1f} "
        f"delta_win_rate={report['summary']['delta_win_rate']} "
        f"report={report_path}")
    return report
