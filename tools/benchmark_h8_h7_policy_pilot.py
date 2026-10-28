#!/usr/bin/env python3
"""Real 14.66M H7 Transformer inference through checked/native H8.

The default Nature sampler uses the exact H7 declared event-scoped Chance
Law and original SeededLawSampler, obtained from a version-checked H8 API.
The optional legacy sequential sampler remains for controlled diagnostics.

Unimplemented player/other chance sites remain explicit unsupported/censored
outcomes. Native-vs-checked H8 agreement alone does not prove legacy-H7
equivalence; benchmark_h8_training_iteration runs the cross-runtime gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import queue
import random
import sys
import tempfile
import time
import traceback
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch

from sts2_ai.emulator.semantic_p0_frozen_player import FrozenP0Player
from sts2_ai.models.h7_batch_actor import RemoteH7Policy, SharedH7Service
from sts2_ai.models.h7_public_policy import H7PlayerSpec
from sts2_ai.models.shared_transformer import H7Config, SharedObjectTransformer
from sts2_ai.nature.combat_scenarios import SCENARIOS
from sts2_ai.nature.draw_only_episode import fair_player_choice_actions
from sts2_ai.nature.draw_only_p0 import DeclaredLaw, SeededLawSampler
from sts2_ai.nature.h4_reward import boss_progress_score, opening_enemy_health
from sts2_ai.nature.h5_curriculum import cell_for_source, source_bank
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.research_combat_fixtures import (
    fixture_from_source,
    rich_source_bank,
    source_bank as structured_source_bank,
    structural_key,
)


def _choose_weighted(rng: random.Random, options: list[Any],
                     raw_weights: list[int]) -> Any:
    if (not options or len(options) != len(raw_weights) or
            any(type(w) is not int or w <= 0 for w in raw_weights)):
        raise ValueError("Malformed source weighted intent support")
    # Integer tickets retain the source raw-weight ratios without float bias.
    ticket = rng.randrange(sum(raw_weights))
    for item, weight in zip(options, raw_weights, strict=True):
        if ticket < weight:
            return item
        ticket -= weight
    raise RuntimeError("Unreachable weighted intent selector")


_worker_transport: Any = None
_worker_spec: H7PlayerSpec | None = None
_worker_policy: RemoteH7Policy | None = None


def _initialize_h8_actor(transport: Any, spec: H7PlayerSpec) -> None:
    """Inherit shared Queue handles at process spawn, never submit them as tasks."""
    global _worker_transport, _worker_spec, _worker_policy
    _worker_transport = transport
    _worker_spec = spec
    _worker_policy = RemoteH7Policy(
        transport, slot=transport.allocate_slot())


def _shard(
    indexed_seeds: list[tuple[int, str]],
    dll: str, bindings: str,
    mode: str, max_boundaries: int, replay_every: int,
    reference_seed: int, capture_player_training: bool = False,
    chance_sampling: str = "event-law",
    replay_indices: frozenset[int] | None = None,
    audit_mechanics: bool = True,
    profile_emulator: bool = False,
    steady_barrier: Any = None,
    warmup_per_worker: int = 0,
    measure_rpc: bool = False,
    local_reference_nature: bool = False,
    inline_player_frame: bool = False,
    structured_curriculum_seed: int | None = None,
    structured_train_specs: int = 256,
    structured_heldout_specs: int = 64,
) -> list[dict[str, Any]]:
    sys.path.insert(0, bindings)
    from sts2_emulator import Session

    if _worker_policy is None or _worker_spec is None:
        raise RuntimeError("H8 shared inference worker not initialized")
    policy = _worker_policy
    spec = _worker_spec
    native = mode == "native"
    rows = []
    worker_started = time.perf_counter()
    rpc_counts: Counter[str] = Counter()
    rpc_seconds: dict[str, float] = {}
    steady_started: float | None = None
    steady_actor_start: dict[str, float | int] | None = None
    with Session(["dotnet", dll, "prototype-ai-jsonl"]) as session:
        initial_session_seconds = time.perf_counter() - worker_started

        def call(op: str, **kwargs: Any) -> Any:
            if not measure_rpc or steady_started is None:
                return session.call(op, **kwargs)
            start = time.perf_counter()
            result = session.call(op, **kwargs)
            rpc_counts[op] += 1
            rpc_seconds[op] = (
                rpc_seconds.get(op, 0.0) + time.perf_counter() - start)
            return result

        if profile_emulator:
            call("h8_perf_reset")
        for position, (index, seed) in enumerate(indexed_seeds):
            if steady_barrier is not None and position == warmup_per_worker:
                # The first barrier ensures absolutely no outstanding
                # player inference requests before the server stats reset.
                steady_barrier.wait(timeout=300)
                if indexed_seeds[0][0] == 0:
                    policy.reset_server_metrics()
                # The second barrier prevents new requests racing the reset.
                steady_barrier.wait(timeout=300)
                if profile_emulator:
                    call("h8_perf_reset")
                steady_started = time.perf_counter()
                steady_actor_start = policy.timings_snapshot()
            if structured_curriculum_seed is None:
                cell = cell_for_source(seed)
                if cell is None:
                    raise RuntimeError("H7 source bank produced uncurated source")
                scenario = SCENARIOS[cell.scenario]
                cell_key = cell.key
                fixture_identity = ("draw-only:seeded-fixed-shrinker:"
                                    + cell.scenario + ":" + cell.deck)
                fixture_kwargs: dict[str, Any] = {
                    "fixture_deck_preset_id": cell.deck}
                source_deck = cell.deck
                source_scenario = cell.scenario
            else:
                scenario, fixture, fixture_index = fixture_from_source(
                    seed, curriculum_seed=structured_curriculum_seed,
                    train_specs=structured_train_specs,
                    heldout_specs=structured_heldout_specs,
                )
                cell_key = f"structured:{fixture_index}:{scenario.scenario_id}"
                fixture_identity = "research:" + structural_key(fixture)
                fixture_kwargs = {"fixture_spec": fixture}
                source_deck = f"structured:{fixture_index}"
                source_scenario = scenario.scenario_id
            actor = FrozenP0Player(
                policy, greedy=spec.greedy, temperature=spec.temperature,
                action_scope=spec.action_scope,
                capture_training=capture_player_training,
                actor_seed=spec.episode_seed(
                    fixture_seed=seed,
                    fixture_id=fixture_identity,
                ),
            )
            # Original H7 per-source reference-seed derivation, with
            # event-scoped deterministic sampling below.
            episode_reference_seed = reference_seed + index * 1000003
            nature_rng = random.Random(episode_reference_seed)
            event_sampler = SeededLawSampler(episode_reference_seed)
            nature_law_trace: list[tuple[str, str, str]] = []
            opening_hp: int | None = None
            opening_max_hp: int | None = None
            initial_enemies_hp: int | None = None
            before_actor = policy.timings_snapshot()
            began = time.perf_counter()
            state = call(
                "h8_begin_seeded",
                seed=seed, act=scenario.act,
                encounter_id=scenario.encounter_id,
                native_entry=scenario.native_entry,
                total_floor=scenario.total_floor,
                ascension=scenario.ascension,
                first_monster_combat_id=scenario.first_monster_combat_id,
                map_column=scenario.map_column,
                map_row=scenario.map_row,
                **fixture_kwargs,
                replay_opening_draw_hooks=True,
                use_m2_automatic_draw=native,
                use_native_draw_transition=native,
                use_native_player_end_turn=native,
                use_trusted_intent=native,
                include_player_frame=inline_player_frame,
                **({"local_reference_seed": episode_reference_seed}
                   if local_reference_nature else {}),
            )["result"]
            handle = state["h8Handle"]
            counts: Counter[str] = Counter()
            transcript: list[tuple[str, Any]] = []
            status = "max_boundaries"
            try:
                for _ in range(max_boundaries):
                    kind = state["kind"]
                    if kind in ("terminal", "unsupported"):
                        status = kind
                        break
                    if kind == "player_choice":
                        pending = state.get("playerChoicePending")
                        if (not isinstance(pending, dict) or pending.get("kind")
                                not in ("player_card_selection",
                                        "draw_hook_action_choice")):
                            status = "unsupported_player_choice"
                            break
                    token = dict(
                        h8_handle=handle,
                        runtime_id=state["runtimeId"],
                        boundary_version=state["boundaryVersion"],
                    )
                    if kind in ("player", "player_choice"):
                        if inline_player_frame:
                            # H8Status already captured the same public frame
                            # to populate legal player actions. Avoid one
                            # redundant emulator RPC and one nested JSON
                            # serialization/parse per player decision.
                            frame = state.get("playerFrame")
                            if not isinstance(frame, dict):
                                raise RuntimeError(
                                    "Missing H8 inline public player frame")
                            obs = frame["observation"]
                            menu = frame["actions"]
                        else:
                            frame = call(
                                "h8_observe", **token,
                                policy_id="prototype-fair-v0",
                            )
                            obs = json.loads(frame["payloadJson"])
                            menu = frame["actions"]
                        if not isinstance(obs, dict) or not isinstance(menu, list):
                            raise RuntimeError("Malformed H8 public policy frame")
                        if opening_hp is None:
                            opening_hp, opening_max_hp = obs.get("hp"), obs.get("max_hp")
                            initial_enemies_hp = opening_enemy_health(obs)
                            if (type(opening_hp) is not int or
                                    type(opening_max_hp) is not int or
                                    opening_max_hp < 1):
                                raise RuntimeError("Missing H8 entry public HP")
                        if [a["actionId"] for a in menu] != state["playerActionIds"]:
                            raise RuntimeError("H8 public action/status mismatch")
                        allowed = (
                            fair_player_choice_actions(
                                state["playerChoicePending"], menu)
                            if kind == "player_choice" else menu
                        )
                        chosen = actor.choose(obs, allowed)
                        if chosen not in [a["actionId"] for a in allowed]:
                            raise RuntimeError("H7 actor selected outside source menu")
                        try:
                            result = call(
                                "h8_step", **token, action_id=chosen,
                                include_player_frame=inline_player_frame,
                                **({"local_reference_seed": episode_reference_seed}
                                   if local_reference_nature else {}),
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                f"H8 scenario={cell_key} source={seed} "
                                f"boundary={state['boundaryVersion']} "
                                f"action={chosen!r}: {exc}") from exc
                    elif kind in ("enemy_intent", "draw_card", "random_target"):
                        if local_reference_nature:
                            # The in-process reference provider consumes
                            # consecutive Nature sites before returning to
                            # the externally controlled player. This branch
                            # also handles any unusual nested continuation
                            # that returns an exposed chance boundary.
                            result = call(
                                "h8_continue_reference", **token,
                                local_reference_seed=episode_reference_seed,
                                include_player_frame=inline_player_frame,
                            )
                            if result["resolvedNatureDecisions"] < 1:
                                raise RuntimeError(
                                    "H8 local reference Nature made no progress")
                            nxt = result["result"]
                            if nxt["boundaryVersion"] <= state["boundaryVersion"]:
                                raise RuntimeError(
                                    "H8 local reference Nature did not advance")
                            state = nxt
                            continue
                        if chance_sampling == "event-law":
                            # The original H7 event-keyed SeededLawSampler
                            # consumes exact emulator-authored Chance Law
                            # identities, conditioning and ordered weights.
                            pending = call(
                                "h8_reference_law", **token)["pending"]
                            law = DeclaredLaw.from_pending(pending)
                            if law.operation != kind:
                                raise RuntimeError("H8 reference law site drift")
                            expected = (
                                tuple(state["intentIds"])
                                if kind == "enemy_intent" else
                                tuple("enemy_instance:" + str(x)
                                      for x in state["targetEnemyInstanceIds"])
                                if kind == "random_target" else
                                tuple("card_instance:" + str(x)
                                      for x in state["cardInstanceIds"])
                            )
                            if law.outcome_ids != expected:
                                raise RuntimeError(
                                    "H8 reference law support order drift")
                            chosen_id = law.outcome_ids[event_sampler.sample(law)]
                            nature_law_trace.append(
                                (law.event_id, law.law_id, chosen_id))
                            chosen = (
                                chosen_id if kind == "enemy_intent" else
                                int(chosen_id.removeprefix(
                                    "enemy_instance:" if kind == "random_target"
                                    else "card_instance:"))
                            )
                        elif chance_sampling == "sequential-raw":
                            if kind == "enemy_intent":
                                chosen = _choose_weighted(
                                    nature_rng, state["intentIds"],
                                    state["intentRawWeights"])
                            else:
                                options = (state["targetEnemyInstanceIds"]
                                           if kind == "random_target"
                                           else state["cardInstanceIds"])
                                if not options:
                                    raise RuntimeError("Empty controlled chance support")
                                chosen = options[nature_rng.randrange(len(options))]
                        else:
                            raise ValueError("Unsupported H8 chance sampling mode")
                        if kind == "enemy_intent":
                            result = call(
                                "h8_resolve_intent", **token,
                                intent_id=chosen,
                                include_player_frame=inline_player_frame,
                                provider_id=(
                                    "h7-event-scoped-reference"
                                    if chance_sampling == "event-law" else
                                    "reference-weighted-h8-pilot"),
                            )
                        elif kind == "random_target":
                            result = call(
                                "h8_resolve_random_target", **token,
                                target_event_id=state["targetEventId"],
                                enemy_instance_id=chosen,
                                include_player_frame=inline_player_frame,
                                provider_id=(
                                    "h7-event-scoped-reference"
                                    if chance_sampling == "event-law" else
                                    "reference-weighted-h8-pilot"),
                            )
                        else:
                            result = call(
                                "h8_resolve_draw_card", **token,
                                card_instance_id=chosen,
                                include_player_frame=inline_player_frame,
                                provider_id=(
                                    "h7-event-scoped-reference"
                                    if chance_sampling == "event-law" else
                                    "reference-weighted-h8-pilot"),
                            )
                    else:
                        raise RuntimeError("Unknown H8 state kind " + str(kind))
                    nxt = result["result"]
                    if (nxt["boundaryVersion"] <= state["boundaryVersion"]
                            or (not local_reference_nature and
                                nxt["boundaryVersion"] !=
                                state["boundaryVersion"] + 1)):
                        raise RuntimeError("H8 boundary skipped/duplicated")
                    if not local_reference_nature:
                        transcript.append((kind, chosen))
                        counts[kind] += 1
                    state = nxt
                else:
                    if state["kind"] in ("terminal", "unsupported"):
                        status = state["kind"]
                    elif state["kind"] == "player_choice":
                        status = "max_boundaries_player_choice"
                execute_elapsed = time.perf_counter() - began
                # Training keeps the actual terminal and sampled replay
                # checks, but skips full mechanics serialization / journal
                # materialization on every unsampled source. The paired
                # research benchmark still opts in to complete auditing.
                audited = (index % replay_every == 0
                           if replay_indices is None else
                           index in replay_indices)
                full_audit = audit_mechanics or audited
                final_sha = (
                    call(
                        "h8_audit_state_fingerprint",
                        h8_handle=handle,
                    )["fingerprint"] if full_audit else None
                )
                history = (
                    call(
                        "h8_decision_history",
                        h8_handle=handle,
                    )["decisions"] if full_audit or local_reference_nature else None
                )
                if local_reference_nature:
                    # The exact externally replayable player/Nature journal
                    # is materialized ONCE per fight, rather than after each
                    # draw and intent. It supplies every skipped Nature choice.
                    transcript = []
                    for entry in history:
                        event_kind = entry["kind"]
                        if event_kind == "player":
                            selected_id = entry["actionId"]
                        elif event_kind == "intent":
                            event_kind = "enemy_intent"
                            selected_id = entry["intentId"]
                        elif event_kind == "draw_card":
                            selected_id = entry["cardInstanceId"]
                        elif event_kind == "random_target":
                            selected_id = entry["targetEnemyInstanceId"]
                        else:
                            raise RuntimeError(
                                "Unsupported H8 local reference journal entry")
                        transcript.append((event_kind, selected_id))
                    counts = Counter(k for k, _ in transcript)
                    if actor.decisions != counts["player"] + counts["player_choice"]:
                        raise RuntimeError(
                            "H8 local reference journal disagrees with player")
                if ((history is not None and len(history) != len(transcript))
                        or state["decisionCount"] != len(transcript)):
                    raise RuntimeError("H8 player/Nature history count mismatch")
                if audited:
                    if profile_emulator:
                        call("h8_perf_stop")
                    try:
                        if not call(
                            "h8_replay_audit", h8_handle=handle
                        )["replayVerified"]:
                            raise RuntimeError(
                                "H8 journal failed deterministic replay")
                    finally:
                        if profile_emulator:
                            call("h8_perf_resume")
                after_actor = policy.timings_snapshot()
                score = None
                if status == "terminal":
                    if (opening_hp is None or opening_max_hp is None or
                            initial_enemies_hp is None):
                        raise RuntimeError("Terminal H8 episode lacks a public opening")
                    score = boss_progress_score(
                        state.get("outcome"),
                        player_exit_hp=state.get("exitHp"),
                        player_max_hp=opening_max_hp,
                        opening_enemy_hp=initial_enemies_hp,
                        terminal_enemy_hp=state.get("terminalEnemyHp"))
                rows.append(dict(
                    seed=seed, cell=cell_key,
                    unsupported_detail=(state.get("unsupportedDetail")
                                        if status == "unsupported" else None),
                    unsupported_boundary_version=(
                        state.get("boundaryVersion")
                        if status == "unsupported" else None),
                    scenario_id=source_scenario,
                    deck_preset_id=source_deck, status=status,
                    is_steady=(steady_barrier is not None and
                               position >= warmup_per_worker),
                    score=score,
                    decision_sequence=[
                        {"phase": (
                            "nature:" + k if k in
                            ("enemy_intent", "draw_card", "random_target") else k),
                         "selected_id": (
                             "card_instance:" + str(v)
                             if k == "draw_card" else
                             "enemy_instance:" + str(v)
                             if k == "random_target" else v)}
                        for k, v in transcript
                    ],
                    player_steps=[
                        {"phase": k, "selected_id": v}
                        for k, v in transcript
                        if k in ("player", "player_choice")
                    ],
                    chance_steps=[
                        {"phase": "nature:" + k,
                         "selected_id": (
                             v if k == "enemy_intent" else
                             "enemy_instance:" + str(v)
                             if k == "random_target" else
                             "card_instance:" + str(v))}
                        for k, v in transcript
                        if k in ("enemy_intent", "draw_card", "random_target")
                    ],
                    training_actions=(
                        tuple(actor.training_actions)
                        if capture_player_training else None
                    ),
                    outcome=state.get("outcome"),
                    exit_hp=state.get("exitHp"),
                    terminal_enemy_hp=state.get("terminalEnemyHp"),
                    boundaries=len(transcript),
                    player_decisions=actor.decisions,
                    captured_player_decisions=len(actor.training_actions),
                    behavior_digest=(
                        canonical_sha256(actor.training_actions)
                        if capture_player_training else None
                    ),
                    nature_draws=counts["draw_card"],
                    nature_targets=counts["random_target"],
                    intents=counts["enemy_intent"],
                    trace_digest=canonical_sha256(transcript),
                    journal_digest=(canonical_sha256(history)
                                    if history is not None else None),
                    reference_law_steps=[
                        {"event_id": event_id, "law_id": law_id,
                         "selected_id": outcome_id}
                        for event_id, law_id, outcome_id in nature_law_trace
                    ],
                    reference_event_digest=(
                        canonical_sha256(nature_law_trace)
                        if chance_sampling == "event-law" and
                        not local_reference_nature else None
                    ),
                    mechanics_sha256=final_sha,
                    replay_verified=audited,
                    episode_wall_seconds=execute_elapsed,
                    actor_metrics={
                        key: float(after_actor[key]) - float(before_actor[key])
                        for key in before_actor
                    },
                ))
            finally:
                released = call("h8_release_many", h8_handles=[handle])
                if released["released"] != 1:
                    raise RuntimeError("H8 session handle was not released")
        if profile_emulator and rows:
            rows[0]["emulator_hotpath_profile"] = call(
                "h8_perf_stop")["profile"]
        if steady_barrier is not None and rows:
            if steady_started is None or steady_actor_start is None:
                raise RuntimeError("H8 worker did not reach steady-state barrier")
            steady_end = time.perf_counter()
            actor_end = policy.timings_snapshot()
            rows[0]["steady_worker_metrics"] = {
                "start": steady_started,
                "end": steady_end,
                "session_startup_seconds": initial_session_seconds,
                "warmup_episodes": warmup_per_worker,
                "steady_episodes": len(rows) - warmup_per_worker,
                "actor_encoding_seconds": (
                    float(actor_end["encoding_seconds"]) -
                    float(steady_actor_start["encoding_seconds"])),
                "actor_wait_seconds": (
                    float(actor_end["wait_seconds"]) -
                    float(steady_actor_start["wait_seconds"])),
                "actor_calls": int(actor_end["calls"]) -
                    int(steady_actor_start["calls"]),
                "rpc_calls": dict(rpc_counts),
                "rpc_seconds": rpc_seconds,
            }
    return rows


def _worker_main(
    transport: Any, spec: H7PlayerSpec,
    shard: list[tuple[int, str]], dll: str, bindings: str,
    mode: str, max_boundaries: int, replay_every: int,
    reference_seed: int, results: Any,
    capture_player_training: bool = False,
    chance_sampling: str = "event-law",
    replay_indices: frozenset[int] | None = None,
    audit_mechanics: bool = True,
    profile_emulator: bool = False,
    steady_barrier: Any = None,
    warmup_per_worker: int = 0,
    measure_rpc: bool = False,
    local_reference_nature: bool = False,
    inline_player_frame: bool = False,
    structured_curriculum_seed: int | None = None,
    structured_train_specs: int = 256,
    structured_heldout_specs: int = 64,
) -> None:
    # Queue handles cross the process boundary *only at spawn*, never
    # inside a ProcessPoolExecutor task queue. Keep one emulator process
    # alive for the entire shard; return only plain JSON-like results.
    try:
        _initialize_h8_actor(transport, spec)
        rows = _shard(
            shard, dll, bindings, mode, max_boundaries,
            replay_every, reference_seed, capture_player_training,
            chance_sampling, replay_indices, audit_mechanics,
            profile_emulator, steady_barrier, warmup_per_worker, measure_rpc,
            local_reference_nature, inline_player_frame,
            structured_curriculum_seed, structured_train_specs,
            structured_heldout_specs)
        results.put(("ok", rows))
    except BaseException:
        results.put(("error", traceback.format_exc()))


def _cohort(seeds: tuple[str, ...], *, dll: str, bindings: str,
            spec: H7PlayerSpec, mode: str, workers: int,
            actor_device: str, actor_max_batch: int,
            actor_delay_ms: float, max_boundaries: int,
            replay_every: int, reference_seed: int,
            capture_player_training: bool = False,
            chance_sampling: str = "event-law",
            replay_indices: frozenset[int] | None = None,
            audit_mechanics: bool = True,
             profile_emulator: bool = False,
             warmup_per_worker: int = 0,
             measure_rpc: bool = False,
             actor_collation: str = "legacy",
             actor_batch_trace: bool = False,
             local_reference_nature: bool = False,
             inline_player_frame: bool = False,
             structured_curriculum_seed: int | None = None,
             structured_train_specs: int = 256,
             structured_heldout_specs: int = 64) -> dict[str, Any]:
    if (replay_indices is not None and
            (not replay_indices or
             any(type(i) is not int or i < 0 or i >= len(seeds)
                 for i in replay_indices))):
        raise ValueError("Invalid H8 preselected terminal replay indices")
    indexed = list(enumerate(seeds))
    shards = [indexed[i::workers] for i in range(min(workers, len(seeds)))]
    if warmup_per_worker < 0 or (
            warmup_per_worker and
            any(len(shard) <= warmup_per_worker for shard in shards)):
        raise ValueError("Each H8 worker needs at least one steady-state episode")
    began = time.perf_counter()
    ctx = multiprocessing.get_context("spawn")
    steady_barrier = (ctx.Barrier(len(shards))
                      if warmup_per_worker else None)
    with ExitStack() as stack:
        server = stack.enter_context(SharedH7Service(
            checkpoint=Path(spec.model_path), digest=spec.model_sha256,
            workers=len(shards), device=actor_device,
            max_batch=actor_max_batch, max_delay_ms=actor_delay_ms,
            collation=actor_collation,
            batch_trace=actor_batch_trace,
        ))
        results = ctx.Queue(maxsize=len(shards) + 1)
        processes = [
            ctx.Process(
                target=_worker_main,
                args=(server.transport, spec, shard, dll, bindings,
                      mode, max_boundaries, replay_every, reference_seed,
                      results, capture_player_training, chance_sampling,
                      replay_indices, audit_mechanics, profile_emulator,
                      steady_barrier, warmup_per_worker, measure_rpc,
                      local_reference_nature, inline_player_frame,
                      structured_curriculum_seed, structured_train_specs,
                      structured_heldout_specs),
            )
            for shard in shards
        ]
        for process in processes:
            process.start()
        rows: list[dict[str, Any]] = []
        errors = []
        try:
            for _ in shards:
                try:
                    kind, payload = results.get(timeout=900)
                except queue.Empty as exc:
                    status = [(p.pid, p.exitcode) for p in processes]
                    raise RuntimeError(
                        f"H8 workers timed out or crashed: {status}") from exc
                if kind == "ok":
                    rows.extend(payload)
                    print(
                        f"[h8-h7-cohort] mode={mode} "
                        f"sources_done={len(rows)}/{len(seeds)} "
                        f"terminals={sum(x['status'] == 'terminal' for x in rows)} "
                        f"elapsed_s={time.perf_counter()-began:.1f}",
                        flush=True,
                    )
                else:
                    errors.append(str(payload))
            if errors:
                raise RuntimeError("H8 worker failure:\\n" +
                                   "\\n".join(errors))
        finally:
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
            results.close()
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError(
                "H8 worker process exited abnormally: " +
                str([(p.pid, p.exitcode) for p in processes]))
    server_stats = dict(server.metrics)
    wall = time.perf_counter() - began
    by_seed = {row["seed"]: row for row in rows}
    if len(rows) != len(seeds) or set(by_seed) != set(seeds):
        raise RuntimeError("Missing/duplicate H8 H7-policy fixture")
    profiles = [row.pop("emulator_hotpath_profile") for row in rows
                if "emulator_hotpath_profile" in row]
    worker_stages = [row.pop("steady_worker_metrics") for row in rows
                     if "steady_worker_metrics" in row]
    if warmup_per_worker and len(worker_stages) != len(shards):
        raise RuntimeError("H8 steady-state worker timing snapshots missing")
    if profile_emulator and len(profiles) != len(shards):
        raise RuntimeError("One or more H8 worker profiler snapshots missing")
    merged: dict[str, dict[str, float | int]] = {}
    for profile in profiles:
        if profile.get("schemaId") != "sts2-h8-nested-hotpath-performance-v1":
            raise RuntimeError("Wrong H8 C# profile schema")
        for key, values in profile["operations"].items():
            target = merged.setdefault(key, {
                "count": 0, "totalSeconds": 0.0, "maxSeconds": 0.0,
                "allocatedBytes": 0})
            target["count"] += int(values["count"])
            target["totalSeconds"] += float(values["totalSeconds"])
            target["maxSeconds"] = max(
                target["maxSeconds"], float(values["maxSeconds"]))
            target["allocatedBytes"] += int(values["allocatedBytes"])
    steady_metrics = None
    if warmup_per_worker:
        rpc_calls: Counter[str] = Counter()
        rpc_seconds: dict[str, float] = {}
        for worker in worker_stages:
            rpc_calls.update(worker["rpc_calls"])
            for name, value in worker["rpc_seconds"].items():
                rpc_seconds[name] = rpc_seconds.get(name, 0.0) + value
        steady_span = max(x["end"] for x in worker_stages) - min(
            x["start"] for x in worker_stages)
        measured = [row for row in rows if row["is_steady"]]
        terminal = sum(row["status"] == "terminal" for row in measured)
        steady_metrics = {
            "schema": "sts2-h8-steady-state-epoch-v1",
            "warmup_per_worker": warmup_per_worker,
            "warmup_episodes": len(rows) - len(measured),
            "measured_episodes": len(measured),
            "terminal_episodes": terminal,
            "wall_seconds": steady_span,
            "terminal_combats_per_second": terminal / steady_span,
            "actor_encoding_seconds": sum(
                x["actor_encoding_seconds"] for x in worker_stages),
            "actor_wait_seconds": sum(
                x["actor_wait_seconds"] for x in worker_stages),
            "actor_requests": sum(x["actor_calls"] for x in worker_stages),
            "emulator_session_startup_seconds_per_worker": [
                x["session_startup_seconds"] for x in worker_stages],
            "emulator_rpc_calls": dict(rpc_calls),
            "emulator_rpc_seconds": rpc_seconds,
            "worker_elapsed_seconds": [
                x["end"] - x["start"] for x in worker_stages],
            "actor_service_counters_exclude_warmup": True,
            "interpretation": (
                "RPC and actor times overlap across processes; GPU forward "
                "and collation are serialized service time, not additive "
                "with worker wait or rollout wall-clock."),
        }
    counts = Counter(row["status"] for row in rows)
    players = sum(row["player_decisions"] for row in rows)
    boundaries = sum(row["boundaries"] for row in rows)
    return dict(
        mode=mode, workers=len(shards), episodes=len(rows),
        statuses=dict(counts), player_decisions=players,
        all_boundaries=boundaries, wall_seconds=wall,
        player_decisions_per_second=players / wall,
        boundaries_per_second=boundaries / wall,
        terminal_combats_per_second=counts["terminal"] / wall,
        replay_verified=sum(row["replay_verified"] for row in rows),
        actor_metrics=server_stats, source_digest=canonical_sha256(seeds),
        steady_state=steady_metrics,
        emulator_hotpath_profile=(
            {"schemaId": "sts2-h8-merged-worker-profile-v1",
             "workerProcesses": len(profiles),
             "inclusiveNestedOperations": True,
             "operations": merged,
             "interpretation": "Nested scopes overlap. Do not sum parent "
             "and child totalSeconds or allocatedBytes."}
            if profile_emulator else None),
        trajectories=by_seed,
    )


def _validate_pair(a: dict[str, Any], b: dict[str, Any]) -> None:
    if a["source_digest"] != b["source_digest"]:
        raise RuntimeError("H8 policy cohort source mismatch")
    for seed, left in a["trajectories"].items():
        right = b["trajectories"][seed]
        for key in (
            "cell", "status", "outcome", "exit_hp", "score",
            "player_steps", "chance_steps", "decision_sequence",
            "terminal_enemy_hp", "boundaries", "player_decisions",
            "nature_draws", "intents", "trace_digest",
            "journal_digest", "mechanics_sha256",
            "captured_player_decisions", "behavior_digest",
            "reference_event_digest", "reference_law_steps",
        ):
            if left[key] != right[key]:
                raise RuntimeError(
                    f"checked/native H8 H7 policy drift {seed}: {key} "
                    f"{left[key]!r} != {right[key]!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--emulator-bindings", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--player-checkpoint", type=Path)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--curriculum", choices=("h5", "h8-structured", "h9-rich"),
                   default="h5")
    p.add_argument("--structured-curriculum-seed", type=int,
                   default=20261027)
    p.add_argument("--require-all-terminal", action="store_true",
                   help="Reject any incomplete/unsupported structured combat")
    p.add_argument("--max-boundaries", type=int, default=1200)
    p.add_argument("--replay-every", type=int, default=8)
    p.add_argument("--study-seed", type=int, default=73127)
    p.add_argument("--actor-seed", type=int, default=31337)
    p.add_argument("--reference-seed", type=int, default=23387)
    p.add_argument("--chance-sampling", choices=("event-law", "sequential-raw"),
                   default="event-law",
                   help="Use H7 declared event-scoped Chance Law (default); "
                        "retain original H8 sequential sampler for diagnosis")
    p.add_argument("--actor-device", default="cuda")
    p.add_argument("--actor-max-batch", type=int, default=16)
    p.add_argument("--inline-player-frame", action="store_true",
                   help="Embed source-public typed observation/actions in "
                        "H8 transition status; skip separate h8_observe RPC "
                        "and nested JSON decoding for every player choice")
    p.add_argument("--local-reference-nature", action="store_true",
                   help="Opt into H8 in-process event-keyed reference Nature; "
                        "eliminates per-draw and intent chance-law/resolution "
                        "RPCs. Uses a new reference RNG algorithm, so compare "
                        "throughput across independently source-valid runs.")
    p.add_argument("--actor-batch-trace", action="store_true",
                   help="Collect per-batch occupancy, padded tensor shapes, "
                        "fill-window, collation, forward and reply timings; "
                        "reset together with actor counters after warmup")
    p.add_argument("--actor-delay-ms", type=float, default=3.0)
    p.add_argument("--actor-collation",
                   choices=("legacy", "packed", "packed-contiguous"),
                   default="legacy",
                   help="Select tensor-identical legacy or fused-transfer actor input")
    p.add_argument("--order", choices=("native", "checked", "ab", "abba"),
                   default="ab")
    p.add_argument("--warmup-per-worker", type=int, default=0,
                   help="With 15 workers, 2 warms 30 of 1024 fights, then "
                        "synchronizes workers and resets shared GPU metrics")
    p.add_argument("--measure-rpc", action="store_true",
                   help="Attribute steady-state Python-to-C# RPC time by op")
    p.add_argument("--sampled-mechanics-audit", action="store_true",
                   help="Serialize full mechanics only at sampled replay sites")
    p.add_argument("--profile-emulator", action="store_true",
                   help="Collect .NET nested stage counters for steady state")
    p.add_argument(
        "--capture-player-training", action="store_true",
        help="Also record exact H7 on-policy public behavior logprob/value "
             "digest per source (does not perform PPO)",
    )
    args = p.parse_args()
    if (args.workers < 1 or args.episodes < 16 or
            args.episodes % 16 or args.max_boundaries < 1 or
            args.replay_every < 1 or args.warmup_per_worker < 0 or
            args.episodes <= args.workers * args.warmup_per_worker):
        p.error("Need balanced 16-cell source cohorts and at least one "
                "measured episode per worker")
    if args.local_reference_nature and args.chance_sampling != "event-law":
        p.error("Local reference Nature is a separate event-scoped "
                "reference provider; use the default --chance-sampling event-law")
    if args.warmup_per_worker and args.order != "native":
        p.error("Steady-state warmup epoch requires --order native; "
                "run unprofiled ABBA separately for revision comparisons")
    if not args.emulator_dll.is_file():
        p.error("Missing H8 emulator DLL")
    if not (args.emulator_bindings / "sts2_emulator" / "__init__.py").is_file():
        p.error("Missing external Python emulator bindings")
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="h8-h7-actor-") as tmp:
        if args.player_checkpoint is not None:
            checkpoint = args.player_checkpoint
            model = SharedObjectTransformer.load_checkpoint(checkpoint)
        else:
            checkpoint = Path(tmp) / "h7-public-player.pt"
            torch.manual_seed(args.study_seed)
            model = SharedObjectTransformer(H7Config(
                width=384, layers=6, heads=8, feedforward=1536,
            ))
            model.save_checkpoint(checkpoint)
        params = sum(p.numel() for p in model.parameters())
        del model
        spec = H7PlayerSpec.from_file(
            checkpoint, seed=args.actor_seed,
            temperature=0.85, greedy=False,
        )
        seeds = (
            rich_source_bank(
                "train", args.episodes, study_seed=args.study_seed)
            if args.curriculum == "h9-rich" else
            structured_source_bank(
                "train", args.episodes, study_seed=args.study_seed)
            if args.curriculum == "h8-structured" else
            source_bank("train", args.episodes, study_seed=args.study_seed)
        )
        order = ["checked", "native"]
        if args.order == "abba":
            order = ["checked", "native", "native", "checked"]
        elif args.order in ("native", "checked"):
            order = [args.order]
        cohorts = []
        incomplete: list[tuple[str, dict[str, Any]]] = []
        for ordinal, mode in enumerate(order, 1):
            print(
                f"[h8-h7-pilot] cohort {ordinal}/{len(order)} mode={mode} "
                f"parameters={params} workers={args.workers}",
                flush=True,
            )
            x = _cohort(
                seeds, dll=str(args.emulator_dll.resolve()),
                bindings=str(args.emulator_bindings.resolve()), spec=spec,
                mode=mode, workers=args.workers,
                actor_device=args.actor_device,
                actor_max_batch=args.actor_max_batch,
                actor_delay_ms=args.actor_delay_ms,
                max_boundaries=args.max_boundaries,
                replay_every=args.replay_every,
                reference_seed=args.reference_seed,
                capture_player_training=args.capture_player_training,
                chance_sampling=args.chance_sampling,
                audit_mechanics=not args.sampled_mechanics_audit,
                profile_emulator=args.profile_emulator,
                warmup_per_worker=args.warmup_per_worker,
                measure_rpc=args.measure_rpc,
                actor_collation=args.actor_collation,
                actor_batch_trace=args.actor_batch_trace,
                local_reference_nature=args.local_reference_nature,
                inline_player_frame=args.inline_player_frame,
                structured_curriculum_seed=(
                    args.structured_curriculum_seed
                    if args.curriculum in ("h8-structured", "h9-rich") else None),
            )
            cohorts.append(x)
            if args.require_all_terminal:
                for seed, row in x["trajectories"].items():
                    if row["status"] != "terminal":
                        incomplete.append((mode, row))
                        print(
                            "[h9-rich-incomplete] "
                            f"mode={mode} source={seed} "
                            f"encounter={row['scenario_id']} "
                            f"status={row['status']} "
                            f"boundary={row['unsupported_boundary_version']} "
                            f"reason={row['unsupported_detail']} "
                            f"last_steps={row['decision_sequence'][-6:]}",
                            flush=True,
                        )
            print(
                f"[h8-h7-pilot] {mode} {x['wall_seconds']:.2f}s "
                f"full_combat={x['statuses'].get('terminal', 0)} "
                f"unsupported={x['statuses'].get('unsupported', 0)} "
                f"unsupported_choice={x['statuses'].get('unsupported_player_choice', 0)} "
                f"player/s={x['player_decisions_per_second']:.2f}",
                flush=True,
            )
        for x in cohorts[1:]:
            _validate_pair(cohorts[0], x)
        if args.order == "native" and args.warmup_per_worker:
            steady = cohorts[0]["steady_state"]
            if steady is None:
                raise RuntimeError("Missing H8 synchronized steady-state epoch")
            # Unsupported/censored environments are themselves essential
            # source coverage failures. Retain their decision journals and
            # timing report instead of discarding all measurements. Never
            # count them as completed combats in throughput.
            steady["all_sources_terminal"] = (
                cohorts[0]["statuses"].get("terminal", 0) == args.episodes)
            steady["status_counts"] = dict(cohorts[0]["statuses"])
            steady["accepted_for_complete_combat_target"] = (
                steady["all_sources_terminal"] and
                steady["terminal_episodes"] == steady["measured_episodes"])
            expected_replays = sum(
                index % args.replay_every == 0
                for index in range(args.episodes))
            if cohorts[0]["replay_verified"] != expected_replays:
                raise RuntimeError(
                    "H8 steady-state replay count differs from sampling plan")
            if (cohorts[0]["actor_metrics"]["requests"] !=
                    steady["actor_requests"]):
                raise RuntimeError(
                    "Steady-state GPU request count disagrees with worker actor")
            print(
                f"[h8-steady] measured={steady['measured_episodes']} "
                f"warmup={steady['warmup_episodes']} "
                f"wall_s={steady['wall_seconds']:.3f} "
                f"combat/s={steady['terminal_combats_per_second']:.3f} "
                f"gpu_requests={steady['actor_requests']} "
                f"status={steady['status_counts']} "
                f"accepted={steady['accepted_for_complete_combat_target']}",
                flush=True,
            )
        result = dict(
            schema="sts2-h8-h7-public-policy-inference-pilot-v1",
            evidence=(
                "real-H7-shared-Transformer-on-H8-with-in-process-"
                "H8-local-reference-Nature;source-distribution-"
                "same-provider-replay-only"
                if args.local_reference_nature else
                "real-H7-shared-Transformer-on-H8-with-H7-event-scoped-law;"
                "cross-emulator-source-parity-pending"
                if args.chance_sampling == "event-law" else
                "real-H7-shared-Transformer-on-H8-sequential-pilot-law"
            ),
            model_sha256=spec.model_sha256,
            parameters=params, emulator_revision=args.emulator_revision,
            emulator_dll_sha256=hashlib.sha256(
                args.emulator_dll.read_bytes()).hexdigest(),
            chance_sampling=(
                "h8-local-event-keyed-source-law-v1"
                if args.local_reference_nature else
                "h7-declared-event-scoped-reference-v1"
                if args.chance_sampling == "event-law" else
                "independent-reference-raw-weight-pilot-v1"
            ),
            capture_player_training=args.capture_player_training,
            source_digest=canonical_sha256(seeds),
            order=order, paired_equivalence=len(cohorts) > 1,
            warmup_per_worker=args.warmup_per_worker,
            profile_emulator=args.profile_emulator,
            sampled_mechanics_audit=args.sampled_mechanics_audit,
            actor_collation=args.actor_collation,
            local_reference_nature=args.local_reference_nature,
            inline_player_frame=args.inline_player_frame,
            actor_max_batch=args.actor_max_batch,
            actor_batch_trace=args.actor_batch_trace,
            actor_overlap=False,
            actor_delay_ms=args.actor_delay_ms,
            actor_device=args.actor_device,
            cohorts=cohorts,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("[h8-h7-pilot] report:", args.output, flush=True)
        if incomplete:
            raise RuntimeError(
                "H9 structured curriculum has incomplete combats: "
                + str(Counter(row["status"] for _, row in incomplete))
                + f"; inspect preserved report {args.output}"
            )


if __name__ == "__main__":
    main()
