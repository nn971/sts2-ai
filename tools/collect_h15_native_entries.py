#!/usr/bin/env python3
"""Resumable native Silent A10 combat-entry collection for H15.

Actual Overgrowth play_run trajectories generate decks/relics/potions/HP and
encounters together. This collector only observes the fair public policy
stream; the native SnapshotCollector stores exact hashes solely for replay
verification. No artificial fixture or hidden RNG fork creates entries.

Train and held-out sets use disjoint source RUNS, never random combat-entry
splits. Multiple pinned H10 macro checkpoints may contribute source policies.
These recipes can immediately feed H10 native on-policy PPO and evaluation.
A native public-history-conditioned search sampler is still REQUIRED before
using them as H15 fair PUCT teachers.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import multiprocessing
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.episode_goal import NATIVE_ACT1_BOSS_GOAL
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.evaluation import play_run
from sts2_ai.evaluation.combat_snapshots import SnapshotCollector

SCHEMA = "sts2-h15-native-a10-combat-entry-corpus-v1"
_WORKER_BACKEND: JsonlEmulatorBackend | None = None
_WORKER_AGENTS: dict[str, Any] = {}
_WORKER_CONFIG: dict[str, Any] = {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(x: Any) -> str:
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def public_build(observation: Any, recipe: Any) -> dict[str, Any]:
    """Audit joint configuration from public observation, without hidden state."""
    state = json.loads(observation.payload_json)
    deck = state.get("deck")
    if not isinstance(deck, list) or any(not isinstance(c, dict) for c in deck):
        raise ValueError("Native combat entry lacks public deck")
    cards = sorted((
        (c.get("card_id"), c.get("upgrade_level", 0))
        for c in deck
    ), key=lambda v: canonical(v))
    if any(type(card_id) is not str or type(level) is not int
           for card_id, level in cards):
        raise ValueError("Malformed public deck card")
    if len(cards) != recipe.deck_size:
        raise ValueError("Native public deck disagrees with snapshot recipe")
    # Fingerprints ignore card instance IDs, so repeated configurations can
    # be counted independently of the order in which cards were acquired.
    deck_hash = digest(cards)
    return {
        "floor": recipe.floor, "tier": recipe.tier,
        "enemy_ids": list(recipe.enemy_ids),
        "hp": recipe.hp, "max_hp": recipe.max_hp,
        "deck_size": len(cards), "deck_unique_cards": len({x[0] for x in cards}),
        "deck_sha256": deck_hash,
        "relic_ids": list(recipe.relic_ids),
        "potion_ids": list(recipe.potion_ids),
        "public_configuration_sha256": digest([
            cards, sorted(recipe.relic_ids), sorted(recipe.potion_ids),
            recipe.hp, recipe.max_hp, recipe.floor, sorted(recipe.enemy_ids),
        ]),
    }


def source_plan(train_runs: int, heldout_runs: int, *,
                seed_prefix: str, macro_sha256: tuple[str, ...]
                ) -> tuple[dict[str, Any], ...]:
    if (train_runs < 1 or heldout_runs < 1 or
            not seed_prefix or not macro_sha256 or
            len(set(macro_sha256)) != len(macro_sha256)):
        raise ValueError("Train/holdout runs and distinct macro sources required")
    rows = []
    for split, count in (("train", train_runs), ("heldout", heldout_runs)):
        for i in range(count):
            macro = macro_sha256[i % len(macro_sha256)]
            rows.append({
                "id": f"{split}-{i:08d}", "split": split, "index": i,
                "seed": f"{seed_prefix}:{split}:{i:08d}",
                "macro_sha256": macro,
            })
    return tuple(rows)


def _close_worker() -> None:
    global _WORKER_BACKEND
    if _WORKER_BACKEND is not None:
        _WORKER_BACKEND.close()
        _WORKER_BACKEND = None


def init_worker(settings: dict[str, Any]) -> None:
    global _WORKER_BACKEND, _WORKER_AGENTS, _WORKER_CONFIG
    import torch
    from sts2_ai.agents.h10_hybrid import H10FrozenHybridAgent

    torch.set_num_threads(1)
    _WORKER_BACKEND = JsonlEmulatorBackend(
        emulator_root=Path(settings["emulator_root"]),
        expected_emulator_revision=settings["emulator_revision"],
        build=False,
    )
    _WORKER_BACKEND.__enter__()
    atexit.register(_close_worker)
    _WORKER_CONFIG = settings
    _WORKER_AGENTS = {}
    for identity, path in settings["macro_paths"].items():
        if sha256_file(Path(path)) != identity:
            raise RuntimeError("Macro checkpoint digest changed")
        _WORKER_AGENTS[identity] = H10FrozenHybridAgent.load(
            macro_model=Path(path),
            combat_checkpoint=Path(settings["combat_checkpoint"]),
            combat_device="cpu",
        )


def run_one(task: dict[str, Any]) -> dict[str, Any]:
    backend = _WORKER_BACKEND
    if backend is None:
        raise RuntimeError("H15 native collector worker is uninitialized")
    agent = _WORKER_AGENTS[task["macro_sha256"]]
    collector = SnapshotCollector(
        backend, seed=task["seed"], source_policy=agent.policy_id,
        ascension=10,
    )
    builds: list[dict[str, Any]] = []
    elapsed = time.monotonic()

    def on_decision(handle: str, observation: Any, legal: Any, decision: Any) -> None:
        previous = len(collector.recipes)
        collector.on_decision(handle, observation, legal, decision)
        if len(collector.recipes) > previous:
            if len(collector.recipes) != previous + 1:
                raise RuntimeError("SnapshotCollector emitted multiple entries per decision")
            builds.append(public_build(observation, collector.recipes[-1]))

    try:
        result = play_run(
            backend, agent, seed=task["seed"],
            policy=InformationPolicy(FAIR_POLICY_ID),
            ascension=10, max_decisions=_WORKER_CONFIG["max_decisions"],
            environment=NATIVE_OVERGROWTH,
            episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
            decision_observer=on_decision,
        )
        recipes = [asdict(x) for x in collector.recipes]
        if len(recipes) != len(builds):
            raise RuntimeError("Native recipes and public configuration descriptors differ")
        return {
            **task, "schema": SCHEMA, "ok": True,
            "emulator_revision": backend.emulator_revision,
            "source_policy": agent.policy_id,
            "outcome": result.outcome, "censored": result.censored,
            "act1_cleared": result.act1_cleared,
            "decisions": result.decisions,
            "combats": len(recipes),
            "recipes": recipes,
            "public_builds": builds,
            "elapsed_seconds": round(time.monotonic() - elapsed, 3),
        }
    except Exception as exc:
        return {
            **task, "schema": SCHEMA, "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "decision_index": len(collector.action_ids),
            "elapsed_seconds": round(time.monotonic() - elapsed, 3),
        }


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    try:
        with temporary.open("w", encoding="utf-8") as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_record(root: Path, task: dict[str, Any]) -> dict[str, Any] | None:
    path = root / "runs" / task["split"] / (task["id"] + ".json")
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if not all(record.get(k) == v for k, v in task.items()):
        raise ValueError(f"Resume record belongs to another run: {path}")
    if record.get("schema") != SCHEMA or record.get("ok") is not True:
        raise ValueError("Incomplete successful source-run record")
    return record


def consolidate(root: Path, tasks: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "heldout": []}
    outcomes: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    enemies: Counter[str] = Counter()
    floors: Counter[int] = Counter()
    configs: set[str] = set()
    decks: set[str] = set()
    source_runs: set[str] = set()
    failures = []
    for task in tasks:
        record = load_record(root, task)
        if record is None:
            failure_path = root / "errors" / task["split"] / (task["id"] + ".json")
            failures.append(task["id"])
            if failure_path.exists():
                outcomes["error"] += 1
            continue
        source_runs.add(record["seed"])
        outcomes[record["outcome"]] += 1
        outcomes["censored"] += int(record["censored"])
        if len(record["recipes"]) != len(record["public_builds"]):
            raise ValueError("Native run contains mismatched build metadata")
        for recipe, build in zip(record["recipes"], record["public_builds"], strict=True):
            if recipe["seed"] != task["seed"] or recipe["emulator_revision"] != (
                    record["emulator_revision"]):
                raise ValueError("Native combat-entry provenance mismatch")
            by_split[task["split"]].append(recipe)
            tiers[recipe["tier"]] += 1
            floors[recipe["floor"]] += 1
            enemies.update(recipe["enemy_ids"])
            configs.add(build["public_configuration_sha256"])
            decks.add(build["deck_sha256"])
    # Output both native H10-compatible recipe JSONL corpora atomically.
    for split, entries in by_split.items():
        output = root / f"{split}.jsonl"
        atomic_write(
            output, "".join(canonical(recipe) + "\n" for recipe in entries),
        )
    return {
        "schema": SCHEMA,
        "source_runs_completed": len(source_runs),
        "source_runs_total": len(tasks),
        "failed_or_missing_run_ids": failures,
        "native_combat_entries": {
            key: len(value) for key, value in by_split.items()
        },
        "distinct_public_configurations": len(configs),
        "distinct_decks": len(decks),
        "outcomes": dict(outcomes), "tiers": dict(tiers),
        "enemies": dict(enemies),
        "floors": {str(k): v for k, v in sorted(floors.items())},
        "train_sha256": sha256_file(root / "train.jsonl"),
        "heldout_sha256": sha256_file(root / "heldout.jsonl"),
        "note": "Exact-native replay recipes; NOT a native fair chance-search posterior",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--combat-checkpoint", type=Path, required=True)
    p.add_argument("--macro-model", action="append", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--train-runs", type=int, default=2048)
    p.add_argument("--heldout-runs", type=int, default=256)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--max-decisions", type=int, default=4096)
    p.add_argument("--seed-prefix", default="h15-native-a10-joint-entry-v1")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if (not 1 <= args.workers <= 15 or
            min(args.train_runs, args.heldout_runs, args.max_decisions) < 1 or
            not args.seed_prefix or len(args.emulator_revision) != 40):
        p.error("Invalid native collection arguments")
    teacher = args.combat_checkpoint.resolve()
    models = tuple(path.resolve() for path in args.macro_model)
    dll = args.emulator_dll.resolve()
    if not dll.is_file() or not teacher.is_file() or any(
            not model.is_file() for model in models):
        p.error("Missing compiled emulator DLL or policy checkpoints")
    macro_paths = {sha256_file(path): str(path) for path in models}
    if len(macro_paths) != len(models):
        p.error("Distinct macro checkpoints required")
    identities = tuple(macro_paths)
    tasks = source_plan(args.train_runs, args.heldout_runs,
                        seed_prefix=args.seed_prefix, macro_sha256=identities)
    revision = args.emulator_revision
    from subprocess import check_output

    actual_revision = check_output(
        ["git", "-C", str(args.emulator_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual_revision != revision:
        p.error("Emulator Git revision differs from pinned revision")
    config = {
        "schema": SCHEMA,
        "emulator_revision": revision,
        "emulator_dll_sha256": sha256_file(dll),
        "combat_checkpoint_sha256": sha256_file(teacher),
        "macro_checkpoint_sha256": sorted(identities),
        "train_runs": args.train_runs, "heldout_runs": args.heldout_runs,
        "seed_prefix": args.seed_prefix, "max_decisions": args.max_decisions,
        "ascension": 10, "environment": NATIVE_OVERGROWTH,
        "observation_policy": FAIR_POLICY_ID,
        "goal": NATIVE_ACT1_BOSS_GOAL,
        "source_policy_family": "H10 frozen H9 combat + pinned phase-split macro",
    }
    print("[h15-native-plan] " + canonical({
        **config, "workers": args.workers, "total_runs": len(tasks),
    }), flush=True)
    if not args.execute:
        return
    root = args.output_root
    identity_path = root / "identity.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError("Cannot resume: corpus configuration or checkpoint changed")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("Output directory is nonempty without corpus identity")
        atomic_write(identity_path, canonical(config) + "\n")

    pending = []
    completed = 0
    for task in tasks:
        success = load_record(root, task)
        error = root / "errors" / task["split"] / (task["id"] + ".json")
        if success is not None:
            completed += 1
        elif error.exists() and not args.retry_failed:
            completed += 1
        else:
            pending.append(task)
    print(f"[h15-native-resume] accounted={completed} pending={len(pending)}",
          flush=True)
    settings = {
        "emulator_root": str(args.emulator_root.resolve()),
        "emulator_revision": revision,
        "combat_checkpoint": str(teacher),
        "macro_paths": macro_paths,
        "max_decisions": args.max_decisions,
    }
    start = time.monotonic()
    counts: Counter[str] = Counter()
    with ProcessPoolExecutor(
        max_workers=min(args.workers, max(1, len(pending))),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker, initargs=(settings,),
    ) as pool:
        queued: dict[Any, dict[str, Any]] = {}
        source = iter(pending)
        for _ in range(min(len(pending), args.workers * 2)):
            task = next(source)
            queued[pool.submit(run_one, task)] = task
        while queued:
            done = next(as_completed(queued))
            task = queued.pop(done)
            record = done.result()
            if record["id"] != task["id"]:
                raise ValueError("Worker returned wrong native source run")
            if record["ok"]:
                output = root / "runs" / task["split"] / (task["id"] + ".json")
                atomic_write(output, canonical(record) + "\n")
                counts["success"] += 1
                counts["combats"] += record["combats"]
                counts[record["outcome"]] += 1
            else:
                output = root / "errors" / task["split"] / (task["id"] + ".json")
                atomic_write(output, canonical(record) + "\n")
                counts["errors"] += 1
                print("[h15-native-error] " + canonical({
                    "seed": task["seed"], "error_type": record["error_type"],
                    "error": record["error"],
                }), flush=True)
            progress = counts["success"] + counts["errors"]
            if progress == 1 or progress % 20 == 0 or not queued:
                duration = max(1e-3, time.monotonic() - start)
                print("[h15-native-progress] " + canonical({
                    "completed": progress,
                    "remaining": len(pending) - progress,
                    "combats": counts["combats"],
                    "failures": counts["errors"],
                    "runs_per_hour": round(3600 * progress / duration, 1),
                    "outcomes": dict(counts),
                }), flush=True)
            following = next(source, None)
            if following is not None:
                queued[pool.submit(run_one, following)] = following
    result = consolidate(root, tasks)
    atomic_write(root / "coverage.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[h15-native-coverage] " + canonical(result), flush=True)
    if result["failed_or_missing_run_ids"]:
        print("[h15-native-warning] Some runs failed or are missing; "
              "review errors and resume with --retry-failed", flush=True)


if __name__ == "__main__":
    main()
