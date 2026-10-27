"""15-process exact-source Nature H1 combat batches with immutable frozen actor.

Each worker owns exactly one external C# JSONL emulator subprocess and loads
a SHA256-pinned exported v15 player once. No source run is silently dropped.
Nature checkpoint is frozen across one on-policy optimizer cohort.
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
from sts2_ai.nature.combat_scenarios import (
    BASELINE_DECK,
    BASELINE_SCENARIO,
    deck_for_seed,
    scenario_for_seed,
)
from sts2_ai.nature.contextual_draw import restore_nature
from sts2_ai.nature.draw_only_episode import DrawOnlyEpisode, play_draw_only_episode
from sts2_ai.nature.two_site_opponent import TwoSiteNature

_worker_session: Any = None
_worker_model: Any = None
_worker_spec: FrozenPlayerSpec | None = None
_worker_revision: str | None = None


def _close_worker() -> None:
    global _worker_session, _worker_model, _worker_spec, _worker_revision
    if _worker_session is not None:
        _worker_session.__exit__(None, None, None)
    _worker_session = None
    _worker_model = None
    _worker_spec = None
    _worker_revision = None


def _initialize(
    emulator_command: tuple[str, ...], binding_dir: str,
    spec: FrozenPlayerSpec, emulator_revision: str,
) -> None:
    global _worker_session, _worker_model, _worker_spec, _worker_revision
    if _worker_session is not None:
        raise RuntimeError("Duplicate draw-only emulator worker process")
    sys.path.insert(0, binding_dir)
    Session = importlib.import_module("sts2_emulator").Session
    _worker_session = Session(list(emulator_command))
    try:
        _worker_model = spec.load_pinned()
        _worker_spec = spec
        _worker_revision = emulator_revision
    except BaseException:
        _close_worker()
        raise
    Finalize(None, _close_worker, exitpriority=10)


def _one(
    index: int, seed: str, reference_seed: int,
    nature_checkpoint: dict[str, Any], draw_mode: str, entry_mode: str,
    max_boundaries: int, scenario_pool: tuple[str, ...],
    deck_pool: tuple[str, ...], capture_player_training: bool = False,
    reference_count: int = 0, score_reference_q: bool = False,
    reward_version: str = "hp_preservation_v1",
    mcts_simulations: int = 0, mcts_max_boundaries: int = 24,
    mcts_combat_budget: int = 4000,
) -> tuple[int, DrawOnlyEpisode, int, tuple[dict[str, Any], ...]]:
    if (_worker_session is None or _worker_spec is None or _worker_model is None
            or _worker_revision is None):
        raise RuntimeError("Draw-only worker was not initialized")
    spec = _worker_spec
    chosen = scenario_for_seed(seed, scenario_pool)
    deck = deck_for_seed(seed, deck_pool)
    nature = restore_nature(nature_checkpoint)
    actual_draw_mode = "reference" if index < reference_count else draw_mode
    actor = FrozenP0Player(
        _worker_model, greedy=spec.greedy, temperature=spec.temperature,
        action_scope=spec.action_scope,
        capture_training=capture_player_training,
        actor_seed=spec.episode_seed(
            fixture_seed=seed, fixture_id=(
                f"draw-only:{entry_mode}"
                if scenario_pool == (BASELINE_SCENARIO,) and deck_pool == (BASELINE_DECK,)
                else f"draw-only:{entry_mode}:{chosen.scenario_id}:{deck}"
            ),
        ),
    )
    result = play_draw_only_episode(
        _worker_session, source_seed=seed, reference_seed=reference_seed,
        nature=nature, emulator_revision=_worker_revision,
        player_choose=actor.choose, player_model_id=spec.policy_id,
        draw_mode=actual_draw_mode, entry_mode=entry_mode,
        max_boundaries=max_boundaries, scenario=chosen,
        deck_preset_id=deck,
        score_reference_q=score_reference_q, reward_version=reward_version,
        mcts_simulations=mcts_simulations,
        mcts_max_boundaries=mcts_max_boundaries,
        mcts_combat_budget=mcts_combat_budget,
    )
    if not result.terminal_replay_verified:
        raise RuntimeError("Draw-only terminal snapshot failed")
    if result.player_model_id != spec.policy_id:
        raise RuntimeError("Draw-only frozen player checksum drift")
    return index, result, os.getpid(), tuple(actor.training_actions)


def _job(
    index: int, seed: str, reference_seed: int,
    nature_checkpoint: dict[str, Any], draw_mode: str, entry_mode: str,
    max_boundaries: int, scenario_pool: tuple[str, ...],
    deck_pool: tuple[str, ...],
    capture_player_training: bool = False, reference_count: int = 0,
    score_reference_q: bool = False,
    reward_version: str = "hp_preservation_v1",
    mcts_simulations: int = 0, mcts_max_boundaries: int = 24,
    mcts_combat_budget: int = 4000,
) -> tuple[int, DrawOnlyEpisode, int, tuple[dict[str, Any], ...]]:
    try:
        return _one(
            index, seed, reference_seed, nature_checkpoint, draw_mode,
            entry_mode, max_boundaries, scenario_pool, deck_pool,
            capture_player_training, reference_count, score_reference_q,
            reward_version, mcts_simulations, mcts_max_boundaries,
            mcts_combat_budget,
        )
    except Exception:
        raise RuntimeError(
            f"Draw-only C# combat failed for index={index} source_seed={seed}:\n"
            + traceback.format_exc()
        ) from None


def draw_only_batch(
    seeds: tuple[str, ...], *, nature: TwoSiteNature, spec: FrozenPlayerSpec,
    emulator_command: tuple[str, ...], emulator_bindings: Path,
    emulator_revision: str, reference_seed: int,
    draw_mode: str = "learned", entry_mode: str = "seeded-fixed-shrinker",
    workers: int = 15, max_boundaries: int = 600,
    progress_every: int = 20,
    scenario_pool: tuple[str, ...] = (BASELINE_SCENARIO,),
    deck_pool: tuple[str, ...] = (BASELINE_DECK,),
    capture_player_training: bool = False, reference_count: int = 0,
    score_reference_q: bool = False,
    reward_version: str = "hp_preservation_v1",
    mcts_simulations: int = 0, mcts_max_boundaries: int = 24,
    mcts_combat_budget: int = 4000,
) -> dict[str, Any]:
    """Return original ordered episodes with real C# process provenance."""
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(not isinstance(x, str) or not x for x in seeds)):
        raise ValueError("Each source-run seed must be nonempty and distinct")
    if draw_mode not in ("learned", "reference"):
        raise ValueError("Unknown draw routing mode")
    if (reference_count < 0 or reference_count >= len(seeds)
            or (reference_count and draw_mode != "learned")
            or (capture_player_training and spec.greedy)):
        raise ValueError("Invalid mixed-Nature/player PPO cohort configuration")
    if (workers <= 0 or max_boundaries <= 0 or progress_every <= 0
            or mcts_simulations < 0 or mcts_max_boundaries < 1
            or mcts_combat_budget < 1
            or (capture_player_training and mcts_simulations)):
        raise ValueError("Invalid emulator worker configuration")
    if not emulator_command or not emulator_revision:
        raise ValueError("Must pin emulator executable and immutable SHA")
    if not (emulator_bindings / "sts2_emulator" / "__init__.py").is_file():
        raise FileNotFoundError("External emulator binding missing")
    if spec.greedy is False:
        # Sampling players are valid but must be independently seeded per fixture.
        if spec.temperature <= 0:
            raise ValueError("Invalid sampled frozen-player temperature")
    spec.load_pinned()
    if entry_mode != "seeded-fixed-shrinker" and (
        scenario_pool != (BASELINE_SCENARIO,) or deck_pool != (BASELINE_DECK,)
    ):
        raise ValueError("Conditioned scenario pool requires fixed-combat entry")
    for seed in seeds:
        scenario_for_seed(seed, scenario_pool)
        deck_for_seed(seed, deck_pool)
    elapsed = time.perf_counter()
    checkpoint = nature.dump()
    ordered: list[DrawOnlyEpisode | None] = [None] * len(seeds)
    process_ids: set[int] = set()
    ordered_player: list[tuple[dict[str, Any], ...] | None] = [None] * len(seeds)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize,
        initargs=(
            emulator_command, str(emulator_bindings.resolve()),
            spec, emulator_revision,
        ),
    ) as pool:
        futures = {
            pool.submit(
                _job, index, seed, reference_seed + index * 1000003,
                checkpoint, draw_mode, entry_mode, max_boundaries,
                scenario_pool, deck_pool,
                capture_player_training, reference_count, score_reference_q,
                reward_version, mcts_simulations, mcts_max_boundaries,
                mcts_combat_budget,
            ): (index, seed)
            for index, seed in enumerate(seeds)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            index, seed = futures[future]
            try:
                resolved_index, episode, pid, player_actions = future.result()
            except Exception as exc:
                print(
                    f"[nature-draw-only] FAIL {seed}: {exc}",
                    flush=True,
                )
                raise
            if resolved_index != index or episode.source_seed != seed:
                raise RuntimeError("Shuffled cross-worker draw-only source result")
            ordered[index] = episode
            ordered_player[index] = player_actions
            process_ids.add(pid)
            if completed % progress_every == 0 or completed == len(seeds):
                print(
                    f"[nature-draw-only] mode={draw_mode} "
                    f"completed={completed}/{len(seeds)} "
                    f"pids={len(process_ids)} elapsed_s="
                    f"{time.perf_counter()-elapsed:.1f}",
                    flush=True,
                )
    if any(item is None for item in ordered) or any(x is None for x in ordered_player):
        raise RuntimeError("Missing an externally executed combat episode")
    results = tuple(item for item in ordered if item is not None)
    return {
        "episodes": results,
        "player_rollouts": tuple(x for x in ordered_player if x is not None),
        "reference_count": reference_count,
        "process_count": len(process_ids),
        "emulator_revision": emulator_revision,
        "source_count": len(results),
        "draw_mode": draw_mode,
        "scenario_pool": scenario_pool,
        "deck_pool": deck_pool,
        "deck_preset_counts": {
            key: sum(e.deck_preset_id == key for e in results)
            for key in sorted(deck_pool)
        },
        "scenario_counts": {
            key: sum(e.scenario_id == key for e in results)
            for key in sorted(scenario_pool)
        },
        "frozen_player": spec.policy_id,
        "nature_revision": nature.revision,
        "elapsed_seconds": time.perf_counter() - elapsed,
        "mcts_simulations_per_decision": mcts_simulations,
        "mcts_rollouts": sum(ep.mcts_rollouts for ep in results),
        "mcts_search_boundaries": sum(ep.mcts_search_boundaries for ep in results),
        "mcts_cutoff_rollouts": sum(ep.mcts_cutoff_rollouts for ep in results),
        "mcts_searched_decisions": sum(ep.mcts_searched_decisions for ep in results),
        "mcts_greedy_fallback_decisions": sum(
            ep.mcts_greedy_fallback_decisions for ep in results),
        "mcts_combat_budget": mcts_combat_budget,
        "total_draws": sum(x.controlled_draw_count for x in results),
        "nontrivial_draws": sum(x.nontrivial_draw_count for x in results),
        "total_intents": sum(x.reference_intent_count for x in results),
    }
