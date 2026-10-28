#!/usr/bin/env python3
"""Full H8 15-worker real-Transformer rollout and one genuine H7 PPO update.

The reference-law mode uses precisely the H7 event-scoped reference provider.
Optional --compare-legacy collects the same source worlds and frozen actor
through the original semantic H7 path, asserting exact trajectories before
the optimizer mutates the policy. This is a benchmark, not long-term training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

try:
    from benchmark_h8_h7_policy_pilot import _cohort
except ModuleNotFoundError as exc:
    if exc.name != "benchmark_h8_h7_policy_pilot":
        raise
    from tools.benchmark_h8_h7_policy_pilot import _cohort
from sts2_ai.models.h7_public_policy import H7PlayerSpec
from sts2_ai.models.shared_transformer import H7Config, SharedObjectTransformer
from sts2_ai.nature.draw_only_workers import draw_only_batch
from sts2_ai.nature.h5_curriculum import DECK_POOL, SCENARIO_POOL, source_bank
from sts2_ai.nature.h4_reward import H4_REWARD
from sts2_ai.nature.ppo_draw import PpoDrawNature
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.training.h7_combat_ppo import H7CombatPPO


def validate_training_cohort(
    cohort: dict[str, Any], seeds: tuple[str, ...],
    *, replay_every: int,
    replay_indices: frozenset[int] | None = None,
    require_capture: bool = True,
    require_reference_event_digest: bool = True,
) -> tuple[tuple[Any, ...], tuple[tuple[dict[str, Any], ...], ...]]:
    """Reject censored, non-reference, unverified or incomplete on-policy data."""
    if cohort["episodes"] != len(seeds) or (
        cohort["statuses"].get("terminal", 0) != len(seeds)
    ):
        raise RuntimeError(
            "CENSORED H8 TRAINING: each source must reach a true terminal; "
            + str(cohort["statuses"])
        )
    episodes = []
    captures = []
    for index, seed in enumerate(seeds):
        row = cohort["trajectories"][seed]
        audited = (index % replay_every == 0 if replay_indices is None
                   else index in replay_indices)
        if (row["status"] != "terminal" or
                row["outcome"] not in ("victory", "defeat") or
                row["replay_verified"] != audited or
                row["score"] is None or
                (require_reference_event_digest and
                 row["reference_event_digest"] is None) or
                (require_capture and (
                    row["captured_player_decisions"] != row["player_decisions"] or
                    row["captured_player_decisions"] != len(row["player_steps"]) or
                    not isinstance(row["training_actions"], (tuple, list)) or
                    len(row["training_actions"]) != row["player_decisions"]))):

            raise RuntimeError("Missing terminal or source-public PPO data: " + seed)
        episodes.append(SimpleNamespace(
            source_seed=seed, outcome=row["outcome"], score=row["score"],
            exit_hp=row["exit_hp"],
            terminal_enemy_hp=row["terminal_enemy_hp"],
            reward_version=row.get("reward_version", H4_REWARD),
            draw_mode="reference", terminal_replay_verified=audited,
            decisions=tuple(SimpleNamespace(**x) for x in row["decision_sequence"]),
        ))
        captures.append(tuple(row["training_actions"] or ()))
    return tuple(episodes), tuple(captures)


def validate_legacy_parity(
    seeds: tuple[str, ...], h8: dict[str, Any], legacy: dict[str, Any],
) -> dict[str, int]:
    """Same source, actor, reference event law, and ordered player/Nature outcomes."""
    if len(legacy["episodes"]) != len(seeds):
        raise RuntimeError("Legacy source episodes missing")
    checked = 0
    for seed, ep, cap in zip(
        seeds, legacy["episodes"], legacy["player_rollouts"], strict=True,
    ):
        row = h8["trajectories"][seed]
        h8_seq = [
            (x["phase"], x["selected_id"]) for x in row["decision_sequence"]
        ]
        h7_seq = [(x.phase, x.selected_id) for x in ep.decisions]
        if h7_seq != h8_seq:
            offset = next(
                (i for i, (a, b) in enumerate(zip(h7_seq, h8_seq))
                 if a != b), min(len(h7_seq), len(h8_seq)),
            )
            raise RuntimeError(
                f"Legacy/H8 choice drift: {seed} at boundary {offset}; "
                f"H7={h7_seq[offset:offset+1]}, H8={h8_seq[offset:offset+1]}"
            )
        h7_laws = [
            (x.law_id, x.selected_id)
            for x in ep.decisions if x.phase.startswith("nature:")
        ]
        h8_laws = [
            (x["law_id"], x["selected_id"])
            for x in row["reference_law_steps"]
        ]
        if h7_laws != h8_laws:
            raise RuntimeError("Legacy/H8 declared Chance Law drift: " + seed)
        if (row["outcome"] != ep.outcome or row["exit_hp"] != ep.exit_hp or
                abs(row["score"] - ep.score) > 1e-10 or
                row["terminal_enemy_hp"] != ep.terminal_enemy_hp):
            raise RuntimeError("Legacy/H8 terminal reward or HP drift: " + seed)
        if len(row["training_actions"]) != len(cap):
            raise RuntimeError("Legacy/H8 on-policy record count drift: " + seed)
        for offset, (h8_action, h7_action) in enumerate(zip(
            row["training_actions"], cap, strict=True,
        )):
            # Shared CUDA inference can produce tiny roundoff differences
            # under different microbatch groupings. Match source/public
            # support, sampled action and frozen model exactly; permit
            # only the PPO optimizer's existing 5e-4 FP acceptance window.
            exact_fields = (
                "source_support_identity", "source_observation",
                "source_legal_actions", "observation", "legal_actions",
                "chosen_index", "selected_id", "behavior_model_id",
                "temperature", "action_scope",
            )
            if any(h8_action[k] != h7_action[k] for k in exact_fields):
                raise RuntimeError(
                    f"Legacy/H8 public behavior drift {seed} at {offset}")
            if any(abs(float(h8_action[k]) - float(h7_action[k])) > 5e-4
                   for k in ("behavior_logprob", "behavior_value")):
                raise RuntimeError(
                    f"Legacy/H8 behavior FP drift {seed} at {offset}")
        checked += 1
    return {"matching_source_trajectories": checked}


def _legacy_cohort(
    seeds: tuple[str, ...], *, args: Any, spec: H7PlayerSpec,
    reference_seed: int,
) -> dict[str, Any]:
    # Exactly the frozen reference law and seeded source corpus used by H7 PPO.
    return draw_only_batch(
        seeds,
        nature=PpoDrawNature(seed=109, mix=0.75),
        spec=spec,
        emulator_command=("dotnet", str(args.emulator_dll.resolve()),
                          "prototype-ai-jsonl"),
        emulator_bindings=args.emulator_bindings,
        emulator_revision=args.emulator_revision,
        reference_seed=reference_seed, draw_mode="reference",
        entry_mode="seeded-fixed-shrinker",
        workers=args.workers, max_boundaries=args.max_boundaries,
        scenario_pool=SCENARIO_POOL, deck_pool=DECK_POOL,
        capture_player_training=True, reward_version=H4_REWARD,
        actor_backend="shared", actor_device=args.actor_device,
        actor_max_batch=args.actor_max_batch,
        actor_delay_ms=args.actor_delay_ms,
        terminal_replay_indices=tuple(
            range(0, len(seeds), args.replay_every)),
        progress_every=max(1, len(seeds) // 4),
        profile_actor=True,
    )


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
    p.add_argument("--learner-device", default="cuda")
    p.add_argument("--actor-max-batch", type=int, default=16)
    p.add_argument("--actor-delay-ms", type=float, default=3.0)
    p.add_argument("--actor-collation",
                   choices=("legacy", "packed"), default="packed",
                   help="Packed actor serving is faster for the frozen H7 "
                        "Transformer without changing model semantics")
    p.add_argument("--separate-player-observe", action="store_true",
                   help="Compatibility path retaining the per-player "
                        "h8_observe RPC; default uses typed inline H8 frame")
    p.add_argument("--local-reference-nature", action="store_true",
                   help="Use H8's in-emulator source-law reference provider. "
                        "No per-draw/intent JSONL round trips; deterministic "
                        "seeded outcomes are recorded in the H8 journal.")
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--ppo-minibatch", type=int, default=64)
    p.add_argument("--ppo-sample-limit", type=int, default=4096)
    p.add_argument("--compare-legacy", action="store_true")
    p.add_argument("--profile-emulator", action="store_true",
                   help="Collect opt-in .NET H8 nested time/allocation counters")
    p.add_argument("--sampled-mechanics-audit", action="store_true",
                   help="Avoid full state SHA/journal materialization on non-audited sources")
    args = p.parse_args()
    if (args.workers < 1 or args.episodes < 16 or
            args.episodes % 16 or args.max_boundaries < 1 or
            args.replay_every < 1 or args.ppo_epochs < 1 or
            args.ppo_minibatch < 1 or args.ppo_sample_limit < 2):
        p.error("H8 PPO benchmark needs positive budgets and balanced 16-cell sources")
    if args.local_reference_nature and args.compare_legacy:
        p.error("Local reference Nature uses a new event-keyed ticket "
                "algorithm; exact legacy MT19937 trajectory comparison "
                "is invalid. Compare performance separately.")
    if (not args.emulator_dll.is_file() or
            not (args.emulator_bindings / "sts2_emulator" /
                 "__init__.py").is_file()):
        p.error("The paired H8 emulator DLL and Python bindings are required")
    for device in (args.actor_device, args.learner_device):
        if device.startswith("cuda") and not torch.cuda.is_available():
            p.error("CUDA device requested but unavailable: " + device)
    torch.set_num_threads(1)
    torch.manual_seed(args.study_seed)
    with tempfile.TemporaryDirectory(prefix="h8-training-benchmark-") as tmp:
        if args.player_checkpoint:
            model = SharedObjectTransformer.load_checkpoint(args.player_checkpoint)
        else:
            model = SharedObjectTransformer(H7Config(
                width=384, layers=6, heads=8, feedforward=1536))
        frozen = Path(tmp) / "frozen-actor.pt"
        model.save_checkpoint(frozen)
        spec = H7PlayerSpec.from_file(
            frozen, seed=args.actor_seed, temperature=0.85, greedy=False)
        params = sum(p.numel() for p in model.parameters())
        learner = H7CombatPPO(
            model, device=args.learner_device, lr=args.lr,
            temperature=spec.temperature, epochs=args.ppo_epochs,
            minibatch=args.ppo_minibatch, sample_limit=args.ppo_sample_limit)
        seeds = source_bank(
            "train", args.episodes, study_seed=args.study_seed)
        print(
            f"[h8-ppo] actor={spec.model_sha256[:16]} params={params} "
            f"workers={args.workers} episodes={len(seeds)} "
            f"reference={'h8-local' if args.local_reference_nature else 'h7-event-law'}",
            flush=True)
        h8 = _cohort(
            seeds, dll=str(args.emulator_dll.resolve()),
            bindings=str(args.emulator_bindings.resolve()), spec=spec,
            mode="native", workers=args.workers,
            actor_device=args.actor_device,
            actor_max_batch=args.actor_max_batch,
            actor_delay_ms=args.actor_delay_ms,
            actor_collation=args.actor_collation,
            local_reference_nature=args.local_reference_nature,
            inline_player_frame=not args.separate_player_observe,
            max_boundaries=args.max_boundaries,
            replay_every=args.replay_every,
            reference_seed=args.reference_seed,
            capture_player_training=True, chance_sampling="event-law",
            profile_emulator=args.profile_emulator,
            audit_mechanics=not args.sampled_mechanics_audit)
        episodes, captures = validate_training_cohort(
            h8, seeds, replay_every=args.replay_every,
            require_reference_event_digest=not args.local_reference_nature)
        print(
            f"[h8-ppo] completed={len(episodes)} "
            f"rollout_s={h8['wall_seconds']:.2f} "
            f"decisions={h8['player_decisions']}",
            flush=True)
        legacy_report = None
        if args.compare_legacy:
            previous = _legacy_cohort(
                seeds, args=args, spec=spec,
                reference_seed=args.reference_seed)
            parity = validate_legacy_parity(seeds, h8, previous)
            legacy_report = {
                "wall_seconds": previous["elapsed_seconds"],
                "process_count": previous["process_count"],
                "actor_metrics": previous["actor_metrics"],
                "player_decisions": sum(
                    sum(x.phase in ("player", "player_choice")
                        for x in ep.decisions)
                    for ep in previous["episodes"]),
                **parity,
            }
            print(
                f"[h8-ppo] legacy parity={parity['matching_source_trajectories']} "
                f"legacy_rollout_s={legacy_report['wall_seconds']:.2f}",
                flush=True)
        began_ppo = time.perf_counter()
        metrics = learner.update(
            episodes, captures,
            behavior_model_id=f"h7-{spec.model_sha256[:16]}",
            seed=args.study_seed * 1000003,
            allow_sampled_terminal_replay=args.replay_every > 1)
        ppo_wall = time.perf_counter() - began_ppo
        print(
            f"[h8-ppo] updates={metrics['updates']} "
            f"source_decisions={metrics['source_decisions']} "
            f"ppo_s={ppo_wall:.2f} kl={metrics['approx_kl']:.5f}",
            flush=True)
        report = {
            "schema": "sts2-h8-h7-reference-ppo-benchmark-v1",
            "model_sha256": spec.model_sha256,
            "model_parameters": params,
            "emulator_revision": args.emulator_revision,
            "emulator_dll_sha256": hashlib.sha256(
                args.emulator_dll.read_bytes()).hexdigest(),
            "source_digest": canonical_sha256(seeds),
            "reference_law": (
                "h8-local-event-keyed-source-law-v1"
                if args.local_reference_nature else
                "h7-event-scoped-declared-chance-law"),
            "local_reference_nature": args.local_reference_nature,
            "actor_collation": args.actor_collation,
            "inline_player_frame": not args.separate_player_observe,
            "source_count": len(seeds),
            "workers": args.workers,
            "max_boundaries": args.max_boundaries,
            "replay_every": args.replay_every,
            "terminal_counts": h8["statuses"],
            "score_mean": statistics.mean(ep.score for ep in episodes),
            "victories": sum(ep.outcome == "victory" for ep in episodes),
            "defeats": sum(ep.outcome == "defeat" for ep in episodes),
            "native_h8_rollout_seconds": h8["wall_seconds"],
            "native_h8_completed_per_second": (
                len(episodes) / h8["wall_seconds"]),
            "native_h8_player_decisions_per_second": (
                h8["player_decisions"] / h8["wall_seconds"]),
            "native_h8_actor_metrics": h8["actor_metrics"],
            "emulator_hotpath_profile": h8["emulator_hotpath_profile"],
            "emulator_profile_instrumented": args.profile_emulator,
            "full_mechanics_audit_every_episode": not args.sampled_mechanics_audit,
            "legacy_h7": legacy_report,
            "ppo_seconds": ppo_wall,
            "total_h8_iteration_seconds": h8["wall_seconds"] + ppo_wall,
            "ppo": metrics,
            "verified_terminal_replays": sum(
                ep.terminal_replay_verified for ep in episodes),
        }
        if legacy_report:
            report["matched_speedup"] = (
                legacy_report["wall_seconds"] / h8["wall_seconds"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n",
                               encoding="utf-8")
        print("[h8-ppo] result:", args.output, flush=True)


if __name__ == "__main__":
    main()
