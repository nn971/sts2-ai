"""External process-isolated P0 two-site Nature combat rollouts.

Each worker opens its own independently pinned emulator JSONL child and
executes complete source-shaped Silent combats. Replay verifies the exact
player/Nature decision transcript and terminal state. This is a structural
pilot, not a native-RNG player-strength experiment.
"""
from __future__ import annotations

import importlib
import json
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing.util import Finalize
from pathlib import Path
from typing import Any

from sts2_ai.emulator.semantic_p0_combat import (
    CombatFixture,
    play_real_two_site,
    strike_first_player,
)
from sts2_ai.emulator.semantic_p0_frozen_player import FrozenP0Player, FrozenPlayerSpec
from sts2_ai.nature.corpus import atomic_write
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.two_site_opponent import PilotChoice, PilotEpisode, TwoSiteNature

REPORT_SCHEMA = "sts2-ai-nature-p0-real-combat-batch-v1"
_worker_session: Any | None = None
_worker_revision = ""
_worker_player_model: Any | None = None
_worker_player_spec: FrozenPlayerSpec | None = None


def _close_worker() -> None:
    global _worker_session, _worker_player_model, _worker_player_spec
    if _worker_session is not None:
        _worker_session.__exit__(None, None, None)
        _worker_session = None
    _worker_player_model = None
    _worker_player_spec = None


def _initialize(
    command: tuple[str, ...], binding_dir: str, revision: str,
    player_spec: FrozenPlayerSpec | None = None,
) -> None:
    global _worker_session, _worker_revision, _worker_player_model, _worker_player_spec
    if _worker_session is not None:
        raise RuntimeError("Worker already has a live emulator")
    sys.path.insert(0, binding_dir)
    module = importlib.import_module("sts2_emulator")
    _worker_session = module.Session(list(command))
    _worker_revision = revision
    _worker_player_spec = player_spec
    if player_spec is not None:
        try:
            _worker_player_model = player_spec.load_pinned()
        except BaseException:
            _close_worker()
            raise
    Finalize(None, _close_worker, exitpriority=10)


def _execute_job_inner(
    index: int, fixture: CombatFixture, nature_state: dict[str, Any],
) -> tuple[int, dict[str, Any], int]:
    if _worker_session is None:
        raise RuntimeError("Semantic emulator worker is not initialized")
    nature = TwoSiteNature.restore(nature_state)
    if _worker_player_spec is None:
        choose = strike_first_player
        policy_id = "fair-strike-defend-then-end-turn-v1"
        episode_actor_seed = None
    else:
        if _worker_player_model is None:
            raise RuntimeError("Frozen player model unavailable in worker")
        episode_actor_seed = _worker_player_spec.episode_seed(
            fixture_seed=fixture.seed, fixture_id=fixture.fixture_id,
        )
        actor = FrozenP0Player(
            _worker_player_model, actor_seed=episode_actor_seed,
            temperature=_worker_player_spec.temperature,
            greedy=_worker_player_spec.greedy,
            action_scope=_worker_player_spec.action_scope,
        )
        choose = actor.choose
        policy_id = _worker_player_spec.policy_id
    first = play_real_two_site(
        _worker_session, fixture, emulator_revision=_worker_revision,
        nature=nature, player_choose=choose,
        max_boundaries=512,
    )
    repeat = play_real_two_site(
        _worker_session, fixture, emulator_revision=_worker_revision,
        scripted_choices=first.trace.replay_choices(),
        player_choose=choose, max_boundaries=512,
    )
    if first.trace.digest != repeat.trace.digest:
        raise ValueError("Real source combat transcript replay diverged")
    result = {
        "fixture": asdict(fixture),
        "trace": asdict(first.trace),
        "trace_digest": first.trace.digest,
        "player_policy": policy_id,
        "player_actor_seed": episode_actor_seed,
        "return": first.player_return,
        "outcome": first.trace.outcome,
        "exit_hp": first.trace.exit_hp,
        "nature_choices": [asdict(x) for x in first.nature_episode.choices],
        "decision_count": len(first.trace.decisions),
        "nature_count": len(first.nature_episode.choices),
        "probability_status": "UNVERIFIED",
        "source_mode": "restricted-seeded-semantic-actual-combat",
        "replay_verified": True,
    }
    return index, result, os.getpid()

def _execute_job(
    index: int, fixture: CombatFixture, nature_state: dict[str, Any],
) -> tuple[int, dict[str, Any], int]:
    try:
        return _execute_job_inner(index, fixture, nature_state)
    except Exception:
        # Vendor-defined sts2_emulator.EmulatorError cannot be unpickled
        # by the main Python process (which intentionally does not load
        # the external Python binding). Transport a built-in exception,
        # preserving the originating stack as plain diagnostic text.
        raise RuntimeError(
            f"Nature real-combat worker fixture={fixture.fixture_id} failed:\n"
            + traceback.format_exc()
        ) from None



def collect_real_combat_batch(
    fixtures: tuple[CombatFixture, ...], *, nature: TwoSiteNature,
    emulator_command: tuple[str, ...], emulator_bindings: Path,
    emulator_revision: str, workers: int = 15,
    progress_every: int = 10,
    player_spec: FrozenPlayerSpec | None = None,
) -> dict[str, Any]:
    if not fixtures or len({f.fixture_id for f in fixtures}) != len(fixtures):
        raise ValueError("Fixtures must have nonempty unique IDs")
    if type(workers) is not int or workers <= 0 or progress_every <= 0:
        raise ValueError("Invalid Nature rollout pool configuration")
    if not emulator_command or not emulator_revision:
        raise ValueError("Emulator executable and pinned revision required")
    if not (emulator_bindings / "sts2_emulator" / "__init__.py").exists():
        raise ValueError("Pinned emulator Python bindings directory is missing")
    if player_spec is not None:
        # Fail before launching workers if the frozen checkpoint drifted.
        player_spec.load_pinned()
    fixtures = tuple(sorted(fixtures, key=lambda x: x.fixture_id))
    observed: set[int] = set()
    completed: list[dict[str, Any] | None] = [None] * len(fixtures)
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize,
        initargs=(emulator_command, str(emulator_bindings.resolve()),
                  emulator_revision, player_spec),
    ) as pool:
        jobs = {
            pool.submit(_execute_job, i, fixture, nature.dump()): fixture
            for i, fixture in enumerate(fixtures)
        }
        for n, future in enumerate(as_completed(jobs), 1):
            fixture = jobs[future]
            try:
                i, row, pid = future.result()
            except Exception as exc:
                print(
                    f"[nature-p0] FAIL fixture={fixture.fixture_id} "
                    f"{type(exc).__name__}: {exc}", flush=True,
                )
                raise
            completed[i] = row
            observed.add(pid)
            if n % progress_every == 0 or n == len(fixtures):
                print(
                    f"[nature-p0] complete={n}/{len(fixtures)} "
                    f"victory={sum(x is not None and x['outcome'] == 'victory' for x in completed)}"
                    " "
                    f"workers_seen={len(observed)}", flush=True,
                )
    if any(x is None for x in completed):
        raise RuntimeError("Incomplete real Nature pilot batch")
    records = [x for x in completed if x is not None]
    return {
        "schema": REPORT_SCHEMA,
        "emulator_revision": emulator_revision,
        "nature": nature.dump(),
        "player_policy": (
            player_spec.policy_id if player_spec is not None else
            "fair-strike-defend-then-end-turn-v1"
        ),
        "player_checkpoint_sha256": (
            player_spec.model_sha256 if player_spec is not None else None
        ),
        "player_actor_seed": (
            player_spec.actor_seed if player_spec is not None else None
        ),
        "player_temperature": (
            player_spec.temperature if player_spec is not None else None
        ),
        "player_action_scope": (
            player_spec.action_scope if player_spec is not None else None
        ),
        "workers_requested": workers,
        "worker_processes_observed": len(observed),
        "fixture_count": len(records),
        "victory_count": sum(x["outcome"] == "victory" for x in records),
        "defeat_count": sum(x["outcome"] == "defeat" for x in records),
        "mean_player_return": sum(x["return"] for x in records) / len(records),
        "nature_decisions": sum(x["nature_count"] for x in records),
        "records_digest": canonical_sha256([
            x["trace_digest"] for x in records
        ]),
        "records": records,
        "elapsed_seconds": time.perf_counter() - started,
        "native_strength_evaluation": False,
        "original_game_probability_status": "UNVERIFIED",
    }


def train_real_nature_batches(
    *, rounds: int, batch_size: int, seed: int,
    emulator_command: tuple[str, ...], emulator_bindings: Path,
    emulator_revision: str, workers: int = 15,
    start_index: int = 0, output: Path | None = None,
    nature_mix: float = 0.75,
    player_spec: FrozenPlayerSpec | None = None,
) -> dict[str, Any]:
    """Run a *smoke* for opponent learning; no native strength verdict.

    Training seeds/encounters come from separately hashed fixtures. No
    player updates, native law, or source-run heldout comparison here.
    """
    if rounds <= 0 or batch_size < 2:
        raise ValueError("A pilot requires positive rounds and 2+ examples per update")
    nature = TwoSiteNature(seed=seed, mix=nature_mix)
    history: list[dict[str, Any]] = []
    for r in range(rounds):
        fixtures = tuple(
            CombatFixture(
                fixture_id=f"act1-shrinker-{start_index + r*batch_size + i:08d}",
                seed=f"nature-pilot-{seed}-source-{start_index + r*batch_size + i}",
                act=1, encounter_id="proto.encounter.shrinker_beetle_weak",
                native_entry="ShrinkerBeetleWeak", total_floor=2,
            ) for i in range(batch_size)
        )
        report = collect_real_combat_batch(
            fixtures, nature=nature, emulator_command=emulator_command,
            emulator_bindings=emulator_bindings,
            emulator_revision=emulator_revision, workers=workers,
            player_spec=player_spec,
        )
        episodes = tuple(
            PilotEpisode(
                row["return"],
                tuple(PilotChoice(**x) for x in row["nature_choices"]),
            ) for row in report["records"]
        )
        stats = nature.update(episodes)
        row = {
            "round": r + 1,
            "training": stats,
            "combat_count": report["fixture_count"],
            "win_rate": report["victory_count"] / report["fixture_count"],
            "mean_player_return": report["mean_player_return"],
            "replay_verified": True,
            "records_digest": report["records_digest"],
        }
        history.append(row)
        print(
            f"[nature-p0-train] round={r+1}/{rounds} "
            f"win_rate={row['win_rate']:.3f} "
            f"return={row['mean_player_return']:.4f} "
            f"nature_grad={stats['nature_gradient_norm']:.5f}",
            flush=True,
        )
        if output is not None:
            atomic_write(
                output,
                json.dumps({
                    "schema": "sts2-ai-real-two-site-nature-smoke-v1",
                    "emulator_revision": emulator_revision,
                    "player_policy": report["player_policy"],
                    "player_checkpoint_sha256": report["player_checkpoint_sha256"],
                    "nature_checkpoint": nature.dump(),
                    "history": history,
                    "evidence_level": "smoke-only",
                    "native_strength_evaluation": False,
                }, indent=2, sort_keys=True) + "\n",
            )
    return {
        "emulator_revision": emulator_revision,
        "player_policy": (
            player_spec.policy_id if player_spec is not None else
            "fair-strike-defend-then-end-turn-v1"
        ),
        "nature_checkpoint": nature.dump(),
        "history": history,
        "evidence_level": "smoke-only",
        "native_strength_evaluation": False,
    }
