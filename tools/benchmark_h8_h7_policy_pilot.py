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
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from sts2_ai.emulator.semantic_p0_frozen_player import FrozenP0Player
from sts2_ai.models.h7_batch_actor import RemoteH7Policy, SharedH7Service
from sts2_ai.models.h7_public_policy import H7PlayerSpec
from sts2_ai.models.shared_transformer import H7Config, SharedObjectTransformer
from sts2_ai.nature.combat_scenarios import SCENARIOS
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


def _shard(
    indexed_seeds: list[tuple[int, str]],
    dll: str, bindings: str, spec: H7PlayerSpec, transport: Any,
    mode: str, max_boundaries: int, replay_every: int,
    reference_seed: int,
) -> list[dict[str, Any]]:
    sys.path.insert(0, bindings)
    from sts2_emulator import Session

    policy = RemoteH7Policy(transport, slot=transport.allocate_slot())
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
                action_scope=spec.action_scope, capture_training=False,
                actor_seed=spec.episode_seed(
                    fixture_seed=seed,
                    fixture_id=("draw-only:seeded-fixed-shrinker:"
                                + cell.scenario + ":" + cell.deck),
                ),
            )
            # Stable per-source independent reference-weighted choices.
            # This sampler is not the H7 event-indexed reference sampler.
            nature_rng = random.Random(reference_seed + index * 1000003)
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
                    if kind in ("terminal", "unsupported", "player_choice"):
                        status = kind
                        break
                    token = dict(
                        h8_handle=handle,
                        runtime_id=state["runtimeId"],
                        boundary_version=state["boundaryVersion"],
                    )
                    if kind == "player":
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
                        chosen = actor.choose(obs, menu)
                        if chosen not in state["playerActionIds"]:
                            raise RuntimeError("H7 actor selected outside source menu")
                        result = session.call("h8_step", **token, action_id=chosen)
                    elif kind == "enemy_intent":
                        chosen = _choose_weighted(
                            nature_rng, state["intentIds"],
                            state["intentRawWeights"],
                        )
                        result = session.call(
                            "h8_resolve_intent", **token, intent_id=chosen,
                            provider_id="reference-weighted-h8-pilot",
                        )
                    elif kind == "draw_card":
                        options = state["cardInstanceIds"]
                        if not options:
                            raise RuntimeError("Empty controlled draw support")
                        chosen = options[nature_rng.randrange(len(options))]
                        result = session.call(
                            "h8_resolve_draw_card", **token,
                            card_instance_id=chosen,
                            provider_id="reference-weighted-h8-pilot",
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
                    if state["kind"] in ("terminal", "unsupported", "player_choice"):
                        status = state["kind"]
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
                rows.append(dict(
                    seed=seed, cell=cell.key, status=status,
                    boundaries=len(transcript),
                    player_decisions=counts["player"],
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


def _cohort(seeds: tuple[str, ...], *, dll: str, bindings: str,
            spec: H7PlayerSpec, mode: str, workers: int,
            actor_device: str, actor_max_batch: int,
            actor_delay_ms: float, max_boundaries: int,
            replay_every: int, reference_seed: int) -> dict[str, Any]:
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
        pool = stack.enter_context(ProcessPoolExecutor(
            max_workers=len(shards), mp_context=ctx,
        ))
        futures = [
            pool.submit(
                _shard, shard, dll, bindings, spec, server.transport,
                mode, max_boundaries, replay_every, reference_seed,
            )
            for shard in shards
        ]
        rows = [x for future in futures for x in future.result()]
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
            "cell", "status", "boundaries", "player_decisions",
            "nature_draws", "intents", "trace_digest",
            "journal_digest", "mechanics_sha256",
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
            )
            cohorts.append(x)
            print(
                f"[h8-h7-pilot] {mode} {x['wall_seconds']:.2f}s "
                f"full_combat={x['statuses'].get('terminal', 0)} "
                f"unsupported={x['statuses'].get('unsupported', 0)} "
                f"player_choice={x['statuses'].get('player_choice', 0)} "
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
            chance_sampling="independent-reference-raw-weight-pilot-v1",
            source_digest=canonical_sha256(seeds),
            order=order, paired_equivalence=True, cohorts=cohorts,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("[h8-h7-pilot] paired mechanics and journals exact:", args.output,
              flush=True)


if __name__ == "__main__":
    main()
