"""Isolated worker pool for same-opening Nature/ordinary combat comparisons.

The canonical record order does not depend on process scheduling. Each worker
owns one external .NET JSONL session; native-ordinary and Nature arms are
recreated from independent source-route entries with a shared seed.
"""
from __future__ import annotations

import importlib
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.util import Finalize
from pathlib import Path
from typing import Any

from sts2_ai.emulator.semantic_p0_frozen_player import FrozenP0Player, FrozenPlayerSpec
from sts2_ai.nature.paired_real_combat import (
    PAIR_SCHEMA,
    PairedSourceFixture,
    paired_record,
    play_paired_source_combat,
)
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.two_site_opponent import TwoSiteNature

BATCH_SCHEMA = "sts2-ai-paired-ordinary-nature-batch-v1"
_worker_session: Any | None = None
_worker_player_model: Any | None = None
_worker_player_spec: FrozenPlayerSpec | None = None


def _close() -> None:
    global _worker_session, _worker_player_model, _worker_player_spec
    if _worker_session is not None:
        _worker_session.__exit__(None, None, None)
    _worker_session = None
    _worker_player_model = None
    _worker_player_spec = None


def _initialize(
    command: tuple[str, ...], binding_dir: str,
    player_spec: FrozenPlayerSpec | None,
) -> None:
    global _worker_session, _worker_player_model, _worker_player_spec
    if _worker_session is not None:
        raise RuntimeError("Duplicate paired emulator session in one worker")
    sys.path.insert(0, binding_dir)
    module = importlib.import_module("sts2_emulator")
    _worker_session = module.Session(list(command))
    _worker_player_spec = player_spec
    try:
        if player_spec is not None:
            _worker_player_model = player_spec.load_pinned()
    except BaseException:
        _close()
        raise
    Finalize(None, _close, exitpriority=10)


def _job_inner(
    index: int, fixture: PairedSourceFixture,
    nature_checkpoint: dict[str, Any], max_boundaries: int,
    capture_player_ppo: bool,
) -> tuple[int, dict[str, Any], int]:
    if _worker_session is None:
        raise RuntimeError("Worker has no emulator session")
    nature = TwoSiteNature.restore(nature_checkpoint)
    if capture_player_ppo and (
        _worker_player_spec is None or _worker_player_spec.greedy
    ):
        raise ValueError("Player PPO requires a real exported sampled player model")
    actors: dict[str, FrozenP0Player] = {}

    def factory(_arm: str):
        if _worker_player_spec is None:
            from sts2_ai.emulator.semantic_p0_combat import strike_first_player
            return strike_first_player
        if _worker_player_model is None:
            raise RuntimeError("Missing frozen player model")
        actor = FrozenP0Player(
            _worker_player_model,
            actor_seed=_worker_player_spec.episode_seed(
                fixture_seed=fixture.seed, fixture_id=fixture.fixture_id,
            ),
            temperature=_worker_player_spec.temperature,
            greedy=_worker_player_spec.greedy,
            action_scope=_worker_player_spec.action_scope,
            capture_training=capture_player_ppo and _arm not in actors,
        )
        if _arm not in actors:
            actors[_arm] = actor
        return actor.choose

    nature_result, ordinary_result = play_paired_source_combat(
        _worker_session, fixture, nature=nature, choose_factory=factory,
        max_boundaries=max_boundaries, verify_replay=True,
    )
    row = paired_record(fixture, nature_result, ordinary_result)
    row["player_policy"] = (
        _worker_player_spec.policy_id if _worker_player_spec is not None
        else "structural-strike-defend-v1"
    )
    row["checkpoint_sha256"] = (
        _worker_player_spec.model_sha256 if _worker_player_spec else None
    )
    row["actor_seed"] = (
        _worker_player_spec.actor_seed if _worker_player_spec else None
    )
    row["nature_policy_sha256"] = canonical_sha256(nature_checkpoint)
    if capture_player_ppo:
        for arm, result in (("nature", nature_result),
                            ("ordinary", ordinary_result)):
            actor = actors[arm]
            if len(actor.training_actions) != sum(
                decision.phase == "player" for decision in result.decisions
            ):
                raise ValueError("Incomplete player PPO action sequence")
        row["player_ppo"] = {
            arm: actors[arm].training_actions
            for arm in ("nature", "ordinary")
        }
        row["ppo_behavior_model_sha256"] = _worker_player_spec.model_sha256
    row["pair_digest"] = canonical_sha256(row)
    return index, row, os.getpid()


def _job(
    index: int, fixture: PairedSourceFixture,
    nature_checkpoint: dict[str, Any], max_boundaries: int,
    capture_player_ppo: bool,
) -> tuple[int, dict[str, Any], int]:
    try:
        return _job_inner(
            index, fixture, nature_checkpoint, max_boundaries,
            capture_player_ppo,
        )
    except Exception:
        # Vendor binding exceptions may not unpickle in the parent process.
        raise RuntimeError(
            f"Paired source fixture {fixture.fixture_id} failed:\n"
            + traceback.format_exc()
        ) from None


def paired_batch(
    fixtures: tuple[PairedSourceFixture, ...], *,
    emulator_command: tuple[str, ...],
    emulator_bindings: Path, emulator_revision: str,
    nature: TwoSiteNature, player_spec: FrozenPlayerSpec | None = None,
    workers: int = 15, max_boundaries: int = 600,
    progress_every: int = 10,
    capture_player_ppo: bool = False,
) -> dict[str, Any]:
    if not fixtures or len({f.fixture_id for f in fixtures}) != len(fixtures):
        raise ValueError("Paired source fixtures must be nonempty with unique IDs")
    source_groups = [(f.region, f.seed) for f in fixtures]
    if len(set(source_groups)) != len(source_groups):
        raise ValueError("Paired source group reuse risks train/test leakage")
    if workers < 1 or max_boundaries < 1 or progress_every < 1:
        raise ValueError("Invalid paired worker configuration")
    if not emulator_command or not emulator_revision:
        raise ValueError("Explicit pinned emulator executable/revision required")
    if not (emulator_bindings / "sts2_emulator" / "__init__.py").is_file():
        raise FileNotFoundError("External emulator Python bindings are missing")
    if capture_player_ppo and (player_spec is None or player_spec.greedy):
        raise ValueError("PPO collection requires a sampled frozen player checkpoint")
    if player_spec is not None:
        player_spec.load_pinned()
    ordered = tuple(sorted(fixtures, key=lambda x: x.fixture_id))
    completed: list[dict[str, Any] | None] = [None] * len(ordered)
    pids: set[int] = set()
    start = time.perf_counter()
    checkpoint = nature.dump()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize,
        initargs=(emulator_command, str(emulator_bindings.resolve()), player_spec),
    ) as pool:
        pending = {
            pool.submit(
                _job, i, fixture, checkpoint, max_boundaries,
                capture_player_ppo,
            ): fixture
            for i, fixture in enumerate(ordered)
        }
        for n, future in enumerate(as_completed(pending), 1):
            fixture = pending[future]
            try:
                index, row, pid = future.result()
            except Exception as exc:
                print(
                    f"[nature-paired] FAIL {fixture.fixture_id}: "
                    f"{type(exc).__name__}: {exc}", flush=True,
                )
                raise
            if ordered[index] != fixture or completed[index] is not None:
                raise ValueError("Paired worker returned a mismatched fixture")
            completed[index] = row
            pids.add(pid)
            if n % progress_every == 0 or n == len(ordered):
                print(
                    f"[nature-paired] {n}/{len(ordered)} "
                    f"last_delta={row['nature_minus_ordinary_return']:+.4f} "
                    f"processes_seen={len(pids)}",
                    flush=True,
                )
    if any(row is None for row in completed):
        raise RuntimeError("Missing paired source combat result")
    records = [row for row in completed if row is not None]
    delta = [float(r["nature_minus_ordinary_return"]) for r in records]
    mean_delta = sum(delta) / len(delta)
    variance = (
        sum((d - mean_delta) ** 2 for d in delta) / (len(delta) - 1)
        if len(delta) > 1 else None
    )
    return {
        "schema": BATCH_SCHEMA,
        "paired_record_schema": PAIR_SCHEMA,
        "emulator_revision": emulator_revision,
        "nature_policy": checkpoint,
        "player_policy": (
            player_spec.policy_id if player_spec else "structural-strike-defend-v1"
        ),
        "player_checkpoint_sha256": (
            player_spec.model_sha256 if player_spec else None
        ),
        "player_ppo_captured": capture_player_ppo,
        "worker_count_requested": workers,
        "worker_processes_observed": len(pids),
        "source_group_count": len(records),
        "nature_victories": sum(r["nature"]["outcome"] == "victory" for r in records),
        "ordinary_victories": sum(
            r["ordinary"]["outcome"] == "victory" for r in records
        ),
        "mean_paired_return_delta": mean_delta,
        "sample_variance_paired_return_delta": variance,
        "records_digest": canonical_sha256([r["pair_digest"] for r in records]),
        "records": records,
        "wall_seconds": time.perf_counter() - start,
        "estimated_training_benefit": "NOT_TESTED",
        "reference_law_calibration": "UNVERIFIED",
        "evidence_level": "integration-smoke",
    }
