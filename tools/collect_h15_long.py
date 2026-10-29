"""Resumable configuration-wide H15 H8 combat data collection.

Uses the existing H9 21-encounter rich generator and its 1024/256 disjoint
structural train/zero-shot index namespaces. Every worker opens one reusable
H8 JSONL subprocess and one CPU H9 teacher; GPU is reserved for fitting.

This samples the *fixed-fixture Reference Nature model*, not the native
full-run conditional distribution. Refusals/exhaustion are recorded, never
silently transformed into training labels. Each per-episode target is written
atomically with immutable teacher/dll/configuration provenance.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import multiprocessing
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sts2_ai.emulator.chance import PublicHistoryStep
from sts2_ai.emulator.h15_h8_reference import (
    H8_REFERENCE_MODEL,
    H8FixtureEntry,
    H8ReferenceCombatBackend,
    H8ReferenceNaturePosterior,
)
from sts2_ai.evaluation.combat_snapshots import POLICY
from sts2_ai.models.h15_policy_value import H15CombatEvaluator
from sts2_ai.nature.research_combat_fixtures import (
    RICH_DISTRIBUTION,
    RICH_ENCOUNTERS,
    rich_fixture_for_index,
    structural_key,
)
from sts2_ai.training.h15_native_search_collector import collect_live_combat_search

CORPUS_SCHEMA = "sts2-h15-rich-reference-corpus-v1"
_WORKER_SESSION: Any = None
_WORKER_TEACHER: H15CombatEvaluator | None = None
_WORKER_SETTINGS: dict[str, Any] = {}


def canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(data: Any) -> str:
    return hashlib.sha256(canonical(data).encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for piece in iter(lambda: source.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def source_seed(split: str, index: int, replica: int, study_seed: int) -> str:
    # Source seed is used to initialize mechanical H8 fixture randomness.
    return f"h15-rich-{split}-v1-{study_seed}-{index}-{replica}"


def nature_seed(split: str, index: int, replica: int, study_seed: int) -> int:
    material = f"h15-nature-v1:{split}:{index}:{replica}:{study_seed}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def episode_id(split: str, index: int, replica: int) -> str:
    return f"{split}-{index:05d}-{replica:03d}"


def tiers(spec: dict[str, Any]) -> str:
    encounter = (spec["encounter_id"], spec["native_encounter_entry"])
    if encounter not in {tuple(group[:2]) for group in RICH_ENCOUNTERS}:
        raise ValueError("Unknown rich combat encounter")
    index = next(
        k for k, item in enumerate(RICH_ENCOUNTERS)
        if tuple(item[:2]) == encounter
    )
    return "weak" if index < 4 else "normal" if index < 15 else (
        "elite" if index < 18 else "boss"
    )


def build_plan(*, train_configs: int, heldout_configs: int, replicas: int,
               study_seed: int, curriculum_seed: int) -> tuple[dict[str, Any], ...]:
    """Distinct structural prefixes, independent per-replica Nature labels."""
    if (not 1 <= train_configs <= 1024 or
            not 0 <= heldout_configs <= 256 or
            not 1 <= replicas <= 1000):
        raise ValueError("H15 rich plan exceeds reserved 1024/256 spec banks")
    tasks: list[dict[str, Any]] = []
    for split, indices in (
        ("train", range(train_configs)),
        ("zero-shot", range(1024, 1024 + heldout_configs)),
    ):
        for index in indices:
            spec = rich_fixture_for_index(index, curriculum_seed=curriculum_seed)
            if spec.get("distribution_id") != RICH_DISTRIBUTION:
                raise ValueError("Unexpected rich research distribution")
            for replica in range(replicas):
                tasks.append({
                    "id": episode_id(split, index, replica),
                    "split": split, "index": index, "replica": replica,
                    "fixture_spec": spec, "fixture_hash": structural_key(spec),
                    "tier": tiers(spec),
                    "seed": source_seed(split, index, replica, study_seed),
                    "nature_seed": nature_seed(split, index, replica, study_seed),
                })
    # Breadth-first: one replica on every structural configuration before
    # returning to any configuration. Reproducible split-disjoint ordering.
    return tuple(sorted(tasks, key=lambda x: (
        x["replica"], x["split"] != "train", x["index"],
    )))


def worker_init(settings: dict[str, Any]) -> None:
    global _WORKER_SESSION, _WORKER_TEACHER, _WORKER_SETTINGS
    import torch

    torch.set_num_threads(1)
    sys.path.insert(0, settings["emulator_bindings"])
    from sts2_emulator import Session  # type: ignore[import-not-found]

    _WORKER_SESSION = Session([
        "dotnet", settings["emulator_dll"], "prototype-ai-jsonl",
    ])
    _WORKER_SESSION.__enter__()
    atexit.register(_WORKER_SESSION.__exit__, None, None, None)
    _WORKER_TEACHER = H15CombatEvaluator.load(
        Path(settings["teacher"]), device="cpu",
    )
    if _WORKER_TEACHER.checkpoint_sha256 != settings["teacher_sha256"]:
        raise RuntimeError("Worker loaded the wrong frozen teacher checkpoint")
    _WORKER_SETTINGS = settings


def run_episode(task: dict[str, Any]) -> dict[str, Any]:
    settings = _WORKER_SETTINGS
    teacher = _WORKER_TEACHER
    if teacher is None or _WORKER_SESSION is None:
        raise RuntimeError("H15 worker was not initialized")
    spec = task["fixture_spec"]
    entry = H8FixtureEntry(
        seed=task["seed"], encounter_id=spec["encounter_id"],
        native_entry=spec["native_encounter_entry"],
        total_floor=spec["total_floor"], act=spec["act"],
        ascension=spec["ascension"], map_column=spec["map_column"],
        map_row=spec["map_row"], fixture_spec=spec,
    )
    backend = H8ReferenceCombatBackend(
        _WORKER_SESSION, entry, emulator_revision=settings["emulator_revision"],
    )
    sampler = H8ReferenceNaturePosterior(
        backend, max_candidates=settings["max_candidates"],
    )
    start = time.monotonic()
    try:
        actual = backend.begin_reference(task["nature_seed"])
        try:
            public = backend.observe(actual, POLICY)
            history = (PublicHistoryStep(
                public, None, backend.legal_actions(actual),
            ),)
        except BaseException:
            backend.release_many((actual,))
            raise
        result = collect_live_combat_search(
            backend, actual, history, fair_sampler=sampler, teacher=teacher,
            source_seed=task["seed"], entry_hash=public.observation_hash,
            encounter_tier=task["tier"], simulations=settings["simulations"],
            max_searches=settings["max_searches"],
            max_decisions=settings["max_decisions"],
            rollout_depth=settings["rollout_depth"],
            tree_depth=settings["tree_depth"],
            seed=nature_seed(task["split"], task["index"], task["replica"],
                             settings["search_seed"]),
        )
        if backend.live_handles:
            raise RuntimeError("H15 backend leaked H8 reference handles")
        rows = list(result.targets)
        for row in rows:
            row.update({
                "chance_prior_id": H8_REFERENCE_MODEL,
                "experiment_regime": "fixed-h8-fixture-reference-nature",
                "h15_corpus_schema": CORPUS_SCHEMA,
                "source_episode_id": task["id"],
                "fixture_spec_sha256": task["fixture_hash"],
                "split": task["split"],
                "encounter_tier": task["tier"],
            })
        return {
            "ok": True, "id": task["id"], "task": task,
            "targets": rows, "outcome": result.outcome,
            "reward": result.reward, "exit_hp": result.exit_hp,
            "decisions": result.combat_decisions,
            "searched_decisions": result.searched_decisions,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "selection_fallbacks": teacher.public_selection_fallbacks,
            "posterior_candidates": sampler.last_stats.candidates,
            "posterior_accepted": sampler.last_stats.accepted,
        }
    except Exception as exc:
        return {
            "ok": False, "id": task["id"], "task": task,
            "error_type": type(exc).__name__, "error": str(exc),
            "elapsed_seconds": round(time.monotonic() - start, 3),
        }
    finally:
        backend.close()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(canonical(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_targets(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(canonical(row) + "\n")
    os.replace(temporary, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--train-configs", type=int, default=1024)
    p.add_argument("--heldout-configs", type=int, default=256)
    p.add_argument("--replicas", type=int, default=1)
    p.add_argument("--study-seed", type=int, default=20261030)
    p.add_argument("--curriculum-seed", type=int, default=20261028)
    p.add_argument("--search-seed", type=int, default=20261030)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--simulations", type=int, default=16)
    p.add_argument("--max-searches", type=int, default=1)
    p.add_argument("--max-candidates", type=int, default=16384)
    p.add_argument("--max-decisions", type=int, default=512)
    p.add_argument("--tree-depth", type=int, default=16)
    p.add_argument("--rollout-depth", type=int, default=512)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if (not 1 <= args.workers <= 15 or
            min(args.simulations, args.max_searches, args.max_candidates,
                args.max_decisions, args.tree_depth, args.rollout_depth) < 1
            or min(args.study_seed, args.curriculum_seed, args.search_seed) < 0
            or len(args.emulator_revision) != 40):
        p.error("Invalid H15 long-collection configuration")

    tasks = build_plan(
        train_configs=args.train_configs,
        heldout_configs=args.heldout_configs,
        replicas=args.replicas, study_seed=args.study_seed,
        curriculum_seed=args.curriculum_seed,
    )
    teacher = args.teacher.resolve()
    dll = args.emulator_dll.resolve()
    if not teacher.is_file() or not dll.is_file():
        p.error("Teacher checkpoint and compiled H8 emulator DLL are required")
    settings: dict[str, Any] = {
        "emulator_bindings": str((args.emulator_root / "bindings/python").resolve()),
        "emulator_dll": str(dll),
        "emulator_revision": args.emulator_revision,
        "teacher": str(teacher),
        "teacher_sha256": sha_file(teacher),
        "dll_sha256": sha_file(dll),
        "simulations": args.simulations,
        "max_searches": args.max_searches,
        "max_candidates": args.max_candidates,
        "max_decisions": args.max_decisions,
        "tree_depth": args.tree_depth,
        "rollout_depth": args.rollout_depth,
        "search_seed": args.search_seed,
    }
    ident = {
        "schema": CORPUS_SCHEMA, "chance_prior_id": H8_REFERENCE_MODEL,
        "rich_distribution": RICH_DISTRIBUTION,
        "train_configs": args.train_configs,
        "heldout_configs": args.heldout_configs,
        "replicas": args.replicas,
        "study_seed": args.study_seed,
        "curriculum_seed": args.curriculum_seed,
        "settings": settings,
    }
    # Ignore paths and worker count when computing immutable corpus identity.
    ident["settings"] = {k: v for k, v in settings.items() if k not in (
        "emulator_dll", "emulator_bindings", "teacher",
    )}
    ident["plan_sha256"] = digest([
        [t["id"], t["fixture_hash"], t["seed"], t["nature_seed"]] for t in tasks
    ])
    print("[h15-long-plan] " + canonical({
        "identity": digest(ident), "jobs": len(tasks),
        "train": args.train_configs * args.replicas,
        "zero_shot": args.heldout_configs * args.replicas,
        "tiers": dict(Counter(x["tier"] for x in tasks)),
        "workers": args.workers,
        "simulations": args.simulations,
        "searches": args.max_searches,
        "chance_prior": H8_REFERENCE_MODEL,
    }), flush=True)
    if not args.execute:
        print("[h15-long-plan] pass --execute to collect", flush=True)
        return

    root = args.output_root
    if root.exists() and (root / "identity.json").exists():
        existing = json.loads((root / "identity.json").read_text(encoding="utf-8"))
        if existing != ident:
            raise ValueError("Output corpus identity differs; choose a new output directory")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("Output directory exists without corpus identity")
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root / "identity.json", ident)
    pending = []
    recovered = 0
    for task in tasks:
        identity = task["id"]
        result_file = root / "summaries" / (identity + ".json")
        error_file = root / "errors" / (identity + ".json")
        if result_file.exists():
            record = json.loads(result_file.read_text(encoding="utf-8"))
            if (record.get("id") != identity or
                    not (root / "targets" / task["split"] /
                         (identity + ".jsonl")).is_file()):
                raise ValueError("Incomplete committed output for " + identity)
            recovered += 1
            continue
        if error_file.exists() and not args.retry_failed:
            recovered += 1
            continue
        pending.append(task)
    print(f"[h15-long-resume] recovered={recovered} pending={len(pending)}", flush=True)

    launched = time.monotonic()
    counts: Counter[str] = Counter()
    durations: list[float] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=worker_init,
        initargs=(settings,),
    ) as pool:
        # Bound the number of in-flight jobs; the large fixture dictionary
        # never needs to be serialized for all 1280 jobs at once.
        cursor = iter(pending)
        inflight: dict[Any, dict[str, Any]] = {}
        for _ in range(min(args.workers * 2, len(pending))):
            task = next(cursor)
            inflight[pool.submit(run_episode, task)] = task
        completed = 0
        while inflight:
            future = next(as_completed(inflight))
            task = inflight.pop(future)
            result = future.result()
            if result["id"] != task["id"]:
                raise RuntimeError("H15 worker returned wrong episode identity")
            if result["ok"]:
                target_path = root / "targets" / task["split"] / (task["id"] + ".jsonl")
                atomic_targets(target_path, result.pop("targets"))
                atomic_json(root / "summaries" / (task["id"] + ".json"), result)
                counts["success"] += 1
                counts[task["tier"] + "_success"] += 1
                counts[task["split"] + "_success"] += 1
            else:
                atomic_json(root / "errors" / (task["id"] + ".json"), result)
                counts["failed"] += 1
                counts[result["error_type"]] += 1
            durations.append(result["elapsed_seconds"])
            completed += 1
            if completed == 1 or completed % 10 == 0 or not inflight:
                elapsed = time.monotonic() - launched
                print("[h15-long-progress] " + canonical({
                    "completed": completed, "pending": len(pending) - completed,
                    "success": counts["success"], "failed": counts["failed"],
                    "fights_per_hour": round(3600 * completed / max(elapsed, 0.001), 2),
                    "mean_fight_seconds": round(sum(durations) / len(durations), 2),
                    "tiers": {
                        key: value for key, value in counts.items()
                        if key.endswith("_success")
                    },
                }), flush=True)
            replacement = next(cursor, None)
            if replacement is not None:
                inflight[pool.submit(run_episode, replacement)] = replacement
    summary = {
        "schema": CORPUS_SCHEMA, "identity": digest(ident),
        "new_counts": dict(counts),
        "elapsed_seconds": round(time.monotonic() - launched, 2),
        "skipped_existing": recovered,
        "target_root": str((root / "targets").resolve()),
    }
    atomic_json(root / "latest-run.json", summary)
    print("[h15-long-complete] " + canonical(summary), flush=True)


if __name__ == "__main__":
    main()
