#!/usr/bin/env python3
"""H8 + actual 14.66M H7 shared-CUDA player-policy integration pilot.

Unlike the scripted H8 benchmark, this evaluates real source-visible H7
public observations and *all* full-player-boundary legal actions through
the pretrained/random frozen H7 Transformer. Nature samples from actual
physical draw support and the source's weighted intent support.

STRICT LIMITS: H8 player sub-choices/other stochastic sites remain incomplete;
they are reported as unsupported, never counted as completed combats. H8's
temporary reference-weighted sampler here does not reproduce the old H7
law's exact event-scoped reference RNG stream; no H7 vs H8 policy-quality
claim is made. This pilot is solely a paired H8 checked/native performance
and model-interface acceptance test, with matched H8 source/choice streams.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import random
import sys
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
import queue
import traceback
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
from sts2_ai.nature.h5_curriculum import cell_for_source, source_bank
from sts2_ai.nature.protocol import canonical_sha256


def _choose_weighted(rng: random.Random, options: list[Any],
                     raw_weights: list[int]) -> Any:
    if (not options or len(options) != len(raw_weights) or
            any(type(w) is not int or w <= 0 for w in raw_weights)):
        raise ValueError("Malformed source weighted intent support")
    # Integer tickets retain the source raw-weight ratios without float bias.
    ticket = rng.randrange(sum(raw_weights))
    for item, weight in zip(options, raw_weights):
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
    reference_law_mode: str = "event-keyed-v111",
) -> list[dict[str, Any]]:
    sys.path.insert(0, bindings)
    from sts2_emulator import Session

    if _worker_policy is None or _worker_spec is None:
        raise RuntimeError("H8 shared inference worker not initialized")
    policy = _worker_policy
    spec = _worker_spec
    native = mode == "native"
    rows = []
    with Session(["dotnet", dll, "prototype-ai-jsonl"]) as session:
        for index, seed in indexed_seeds:
            cell = cell_for_source(seed)
            if cell is None:
                raise RuntimeError("H7 source bank produced uncurated source")
            scenario = SCENARIOS[cell.scenario]
            actor = FrozenP0Player(
                policy, greedy=spec.greedy, temperature=spec.temperature,
                action_scope=spec.action_scope,
                capture_training=capture_player_training,
                actor_seed=spec.episode_seed(
                    fixture_seed=seed,
                    fixture_id=("draw-only:seeded-fixed-shrinker:"
                                + cell.scenario + ":" + cell.deck),
                ),
            )
            # Stable per-source independent reference-weighted choices.
            # This sampler is not the H7 event-indexed reference sampler.
            nature_rng = random.Random(reference_seed + index * 1000003)
            nature_law_sampler = SeededLawSampler(
                reference_seed + index * 1000003)
            before_actor = policy.timings_snapshot()
            began = time.perf_counter()
            state = session.call(
                "h8_begin_seeded",
                seed=seed, act=scenario.act,
                encounter_id=scenario.encounter_id,
                native_entry=scenario.native_entry,
                total_floor=scenario.total_floor,
                ascension=scenario.ascension,
                first_monster_combat_id=scenario.first_monster_combat_id,
                map_column=scenario.map_column,
                map_row=scenario.map_row,
                fixture_deck_preset_id=cell.deck,
                replay_opening_draw_hooks=True,
                use_m2_automatic_draw=native,
                use_native_draw_transition=native,
                use_native_player_end_turn=native,
                use_trusted_intent=native,
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
                    exact_law = None
                    if (reference_law_mode == "event-keyed-v111"
                            and kind in ("enemy_intent", "draw_card")):
                        # Explicit source-authentic declared law request.
                        # Temporary v111 projection is used only for the
                        # reproducible reference-law acceptance experiment;
                        # native H8 typed law projection remains M3 work.
                        pending = session.call(
                            "h8_reference_law", **token)["pending"]
                        exact_law = DeclaredLaw.from_pending(pending)
                        if exact_law.operation != kind:
                            raise RuntimeError("H8 typed reference law site mismatch")
                    if kind in ("player", "player_choice"):
                        frame = session.call(
                            "h8_observe", **token,
                            policy_id="prototype-fair-v0",
                        )
                        obs = json.loads(frame["payloadJson"])
                        menu = frame["actions"]
                        if not isinstance(obs, dict) or not isinstance(menu, list):
                            raise RuntimeError("Malformed H8 public policy frame")
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
                        result = session.call("h8_step", **token, action_id=chosen)
                    elif kind == "enemy_intent":
                        if exact_law is not None:
                            if list(exact_law.outcome_ids) != state["intentIds"]:
                                raise RuntimeError(
                                    "H8 exact intent law support mismatch")
                            chosen = exact_law.outcome_ids[
                                nature_law_sampler.sample(exact_law)]
                        else:
                            chosen = _choose_weighted(
                                nature_rng, state["intentIds"],
                                state["intentRawWeights"],
                            )
                        result = session.call(
                            "h8_resolve_intent", **token, intent_id=chosen,
                            provider_id=(
                                "h7-event-keyed-reference"
                                if exact_law is not None
                                else "h8-raw-weight-pilot"),
                        )
                    elif kind == "draw_card":
                        options = state["cardInstanceIds"]
                        if not options:
                            raise RuntimeError("Empty controlled draw support")
                        if exact_law is not None:
                            expected = [
                                f"card_instance:{card}" for card in options
                            ]
                            if list(exact_law.outcome_ids) != expected:
                                raise RuntimeError(
                                    "H8 exact physical draw support mismatch")
                            outcome = exact_law.outcome_ids[
                                nature_law_sampler.sample(exact_law)]
                            chosen = int(outcome.removeprefix("card_instance:"))
                        else:
                            chosen = options[nature_rng.randrange(len(options))]
                        result = session.call(
                            "h8_resolve_draw_card", **token,
                            card_instance_id=chosen,
                            provider_id=(
                                "h7-event-keyed-reference"
                                if exact_law is not None
                                else "h8-raw-weight-pilot"),
                        )
                    else:
                        raise RuntimeError("Unknown H8 state kind " + str(kind))
                    nxt = result["result"]
                    if nxt["boundaryVersion"] != state["boundaryVersion"] + 1:
                        raise RuntimeError("H8 boundary skipped/duplicated")
                    transcript.append((kind, chosen))
                    counts[kind] += 1
                    state = nxt
                else:
                    if state["kind"] in ("terminal", "unsupported"):
                        status = state["kind"]
                    elif state["kind"] == "player_choice":
                        status = "max_boundaries_player_choice"
                execute_elapsed = time.perf_counter() - began
                # Canonical hash is an explicit audit, never an inner step.
                final_sha = session.call(
                    "h8_audit_state_fingerprint",
                    h8_handle=handle,
                )["fingerprint"]
                history = session.call(
                    "h8_decision_history",
                    h8_handle=handle,
                )["decisions"]
                if len(history) != len(transcript):
                    raise RuntimeError("H8 player/Nature history count mismatch")
                audited = index % replay_every == 0
                if audited and not session.call(
                    "h8_replay_audit", h8_handle=handle
                )["replayVerified"]:
                    raise RuntimeError("H8 journal failed deterministic replay")
                after_actor = policy.timings_snapshot()
                if (capture_player_training and
                        len(actor.training_actions) != actor.decisions):
                    raise RuntimeError(
                        "H8 H7 player choices missing on-policy behavior records")
                rows.append(dict(
                    seed=seed, cell=cell.key, status=status,
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
                    intents=counts["enemy_intent"],
                    trace_digest=canonical_sha256(transcript),
                    journal_digest=canonical_sha256(history),
                    mechanics_sha256=final_sha,
                    replay_verified=audited,
                    episode_wall_seconds=execute_elapsed,
                    actor_metrics={
                        key: float(after_actor[key]) - float(before_actor[key])
                        for key in before_actor
                    },
                ))
            finally:
                released = session.call("h8_release_many", h8_handles=[handle])
                if released["released"] != 1:
                    raise RuntimeError("H8 session handle was not released")
    return rows


def _worker_main(
    transport: Any, spec: H7PlayerSpec,
    shard: list[tuple[int, str]], dll: str, bindings: str,
    mode: str, max_boundaries: int, replay_every: int,
    reference_seed: int, results: Any,
    capture_player_training: bool = False,
    reference_law_mode: str = "event-keyed-v111",
) -> None:
    # Queue handles cross the process boundary *only at spawn*, never
    # inside a ProcessPoolExecutor task queue. Keep one emulator process
    # alive for the entire shard; return only plain JSON-like results.
    try:
        _initialize_h8_actor(transport, spec)
        rows = _shard(
            shard, dll, bindings, mode, max_boundaries,
            replay_every, reference_seed, capture_player_training,
            reference_law_mode)
        results.put(("ok", rows))
    except BaseException:
        results.put(("error", traceback.format_exc()))


def _cohort(seeds: tuple[str, ...], *, dll: str, bindings: str,
            spec: H7PlayerSpec, mode: str, workers: int,
            actor_device: str, actor_max_batch: int,
            actor_delay_ms: float, max_boundaries: int,
            replay_every: int, reference_seed: int,
            capture_player_training: bool = False,
            reference_law_mode: str = "event-keyed-v111") -> dict[str, Any]:
    indexed = list(enumerate(seeds))
    shards = [indexed[i::workers] for i in range(min(workers, len(seeds)))]
    began = time.perf_counter()
    ctx = multiprocessing.get_context("spawn")
    with ExitStack() as stack:
        server = stack.enter_context(SharedH7Service(
            checkpoint=Path(spec.model_path), digest=spec.model_sha256,
            workers=len(shards), device=actor_device,
            max_batch=actor_max_batch, max_delay_ms=actor_delay_ms,
        ))
        results = ctx.Queue(maxsize=len(shards) + 1)
        processes = [
            ctx.Process(
                target=_worker_main,
                args=(server.transport, spec, shard, dll, bindings,
                      mode, max_boundaries, replay_every, reference_seed,
                      results, capture_player_training,
                      reference_law_mode),
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
        trajectories=by_seed,
    )


def _validate_pair(a: dict[str, Any], b: dict[str, Any]) -> None:
    if a["source_digest"] != b["source_digest"]:
        raise RuntimeError("H8 policy cohort source mismatch")
    for seed, left in a["trajectories"].items():
        right = b["trajectories"][seed]
        for key in (
            "cell", "status", "outcome", "exit_hp",
            "terminal_enemy_hp", "boundaries", "player_decisions",
            "nature_draws", "intents", "trace_digest",
            "journal_digest", "mechanics_sha256",
            "captured_player_decisions", "behavior_digest",
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
    p.add_argument("--max-boundaries", type=int, default=1200)
    p.add_argument("--replay-every", type=int, default=8)
    p.add_argument("--study-seed", type=int, default=73127)
    p.add_argument("--actor-seed", type=int, default=31337)
    p.add_argument("--reference-seed", type=int, default=23387)
    p.add_argument("--actor-device", default="cuda")
    p.add_argument("--actor-max-batch", type=int, default=16)
    p.add_argument("--actor-delay-ms", type=float, default=3.0)
    p.add_argument("--order", choices=("ab", "abba"), default="ab")
    p.add_argument(
        "--reference-law-mode",
        choices=("event-keyed-v111", "raw-weight-pilot"),
        default="event-keyed-v111",
        help="Exact H7 reference law is audit-compatible but temporarily "
             "costs legacy event projection; raw-weight mode is only a "
             "separate fast performance pilot, not H7 reference-equivalent.",
    )
    p.add_argument(
        "--capture-player-training", action="store_true",
        help="Also record exact H7 on-policy public behavior logprob/value "
             "digest per source (does not perform PPO)",
    )
    args = p.parse_args()
    if (args.workers < 1 or args.episodes < 16 or
            args.max_boundaries < 1 or args.replay_every < 1):
        p.error("Positive workers, boundaries, audits and >=16 source cells needed")
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
        seeds = source_bank("train", args.episodes, study_seed=args.study_seed)
        order = ["checked", "native"]
        if args.order == "abba":
            order = ["checked", "native", "native", "checked"]
        cohorts = []
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
                reference_law_mode=args.reference_law_mode,
            )
            cohorts.append(x)
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
        result = dict(
            schema="sts2-h8-h7-public-policy-inference-pilot-v1",
            evidence="real-H7-shared-Transformer-on-H8-not-legacy-H7-equivalent-RNG",
            model_sha256=spec.model_sha256,
            parameters=params, emulator_revision=args.emulator_revision,
            emulator_dll_sha256=hashlib.sha256(
                args.emulator_dll.read_bytes()).hexdigest(),
            chance_sampling=args.reference_law_mode,
            capture_player_training=args.capture_player_training,
            source_digest=canonical_sha256(seeds),
            order=order, paired_equivalence=True, cohorts=cohorts,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("[h8-h7-pilot] paired mechanics and journals exact:", args.output,
              flush=True)


if __name__ == "__main__":
    main()
