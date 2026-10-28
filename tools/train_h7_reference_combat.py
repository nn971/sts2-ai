#!/usr/bin/env python3
"""H7 fresh-start Silent A10 reference-law combat PPO, using real P0 combats.

No previous actor, distillation data, learned Nature, or emulator edits.
Requires an already-built pinned semantic-combat emulator and Python bindings.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from sts2_ai.models.h7_public_policy import H7PlayerSpec
from sts2_ai.models.shared_game_objects import SHARED_OBJECT_SCHEMA
from sts2_ai.models.shared_transformer import H7_MODEL_FORMAT, H7Config, SharedObjectTransformer
from sts2_ai.nature.corpus import file_sha256
from sts2_ai.nature.draw_only_workers import draw_only_batch
from sts2_ai.nature.h4_reward import H4_REWARD
from sts2_ai.nature.h5_curriculum import (
    DECK_POOL,
    SCENARIO_POOL,
    TRAIN_CELLS,
    cell_for_source,
    source_bank,
)
from sts2_ai.nature.research_combat_fixtures import (
    fixture_from_source, source_bank as structured_source_bank,
)
from sts2_ai.nature.ppo_draw import PpoDrawNature
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.training.h7_combat_ppo import H7_PPO_SCHEMA, H7CombatPPO

RUN_SCHEMA = "sts2-h7-reference-combat-run-v1"


def _source_meta(seed: str, args: Any) -> tuple[str, str, str]:
    """Source-visible identity, structural deck label and audit stratum."""
    if getattr(args, "curriculum", "h5") == "h8-structured":
        scenario, _, index = fixture_from_source(
            seed, curriculum_seed=args.structured_curriculum_seed)
        return scenario.scenario_id, f"structured:{index}", scenario.scenario_id
    cell = cell_for_source(seed)
    if cell is None:
        raise ValueError("Uncurated combat source")
    return cell.scenario, cell.deck, cell.key


def _source_bank(split: str, count: int, *, args: Any,
                 round_index: int = 0) -> tuple[str, ...]:
    if getattr(args, "curriculum", "h5") == "h8-structured":
        return structured_source_bank(
            split, count, study_seed=args.study_seed, round_index=round_index)
    return source_bank(
        split, count, study_seed=args.study_seed, round_index=round_index)


def terminal_replay_audit_indices(
    seeds: tuple[str, ...], every: int, *,
    curriculum: str = "h5", curriculum_seed: int = 20261027,
) -> tuple[int, ...]:
    """Deterministic pre-outcome audit: at least one combat per source cell.

    Audits are selected before rollout by original seed ordering, independently
    of outcomes, worker scheduling, actor action selection and RNG draws.
    """
    if type(every) is not int or every < 1 or not seeds:
        raise ValueError("Invalid terminal replay audit rate")
    seen: dict[str, int] = {}
    indices: list[int] = []
    for index, seed in enumerate(seeds):
        if curriculum == "h8-structured":
            scenario, _, _ = fixture_from_source(
                seed, curriculum_seed=curriculum_seed)
            key = scenario.scenario_id
        else:
            cell = cell_for_source(seed)
            if cell is None:
                raise ValueError("Terminal audit cannot sample uncurated sources")
            key = cell.key
        ordinal = seen.get(key, 0)
        if ordinal % every == 0:
            indices.append(index)
        seen[key] = ordinal + 1
    return tuple(indices)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".pending", delete=False,
        ) as f:
            temporary = Path(f.name)
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def summarize(
    episodes: tuple[Any, ...], *, curriculum: str = "h5",
    curriculum_seed: int = 20261027,
) -> dict[str, Any]:
    def summary(items: list[Any]) -> dict[str, float | int | None]:
        wins = [x for x in items if x.outcome == "victory"]
        defeats = [x for x in items if x.outcome == "defeat"]
        loss_hp = [
            x.terminal_enemy_hp for x in defeats
            if x.terminal_enemy_hp is not None
        ]
        return {
            "combats": len(items),
            "terminal_replay_verified": sum(
                bool(x.terminal_replay_verified) for x in items
            ),
            "terminal_replay_not_checked": sum(
                not x.terminal_replay_verified for x in items
            ),
            "wins": len(wins),
            "defeats": len(defeats),
            "mean_score": statistics.mean(x.score for x in items),
            "victory_mean_exit_hp": (
                statistics.mean(x.exit_hp for x in wins) if wins else None
            ),
            "defeat_mean_remaining_enemy_hp": (
                statistics.mean(loss_hp) if loss_hp else None
            ),
        }
    cells: dict[str, list[Any]] = {}
    for ep in episodes:
        if curriculum == "h8-structured":
            scenario, _, index = fixture_from_source(
                ep.source_seed, curriculum_seed=curriculum_seed)
            key = f"structured:{index}:{scenario.scenario_id}"
        else:
            cell = cell_for_source(ep.source_seed)
            if cell is None:
                raise ValueError("H7 uncurated combat source")
            key = cell.key
        cells.setdefault(key, []).append(ep)
    return {
        **summary(list(episodes)),
        "by_cell": {name: summary(items) for name, items in sorted(cells.items())},
    }


def collect_h8(
    seeds: tuple[str, ...], *, spec: H7PlayerSpec, args: Any,
    reference_seed: int, capture: bool,
) -> dict[str, Any]:
    """Use the validated native H8 reference-law path for an H7 PPO cohort.

    H8 shares exactly the H7 public actor, source bank, Chance Law and reward.
    It keeps only sampled explicit mechanics hashes/journals during training;
    the runtime still owns its complete reproducible decision journal.
    """
    # Work equally when invoked as a tools/ CLI script and when imported
    # as tools.train_h7_reference_combat by test suites or other callers.
    try:
        from benchmark_h8_h7_policy_pilot import _cohort
        from benchmark_h8_training_iteration import validate_training_cohort
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "benchmark_h8_h7_policy_pilot",
            "benchmark_h8_training_iteration",
        }:
            raise
        from tools.benchmark_h8_h7_policy_pilot import _cohort
        from tools.benchmark_h8_training_iteration import (
            validate_training_cohort,
        )

    if getattr(args, "actor_backend", "local") != "shared":
        raise ValueError("H8 training requires the shared actor backend")
    audit_every = (
        getattr(args, "train_terminal_replay_every", 1)
        if capture else 1
    )
    audit_indices = terminal_replay_audit_indices(
        seeds, audit_every, curriculum=args.curriculum,
        curriculum_seed=args.structured_curriculum_seed)
    audit_set = frozenset(audit_indices)
    h8 = _cohort(
        seeds,
        dll=str(args.emulator_dll.resolve()),
        bindings=str(args.emulator_bindings.resolve()),
        spec=spec, mode="native", workers=args.workers,
        actor_device=args.actor_device,
        actor_max_batch=args.actor_max_batch,
        actor_delay_ms=args.actor_delay_ms,
        max_boundaries=args.max_boundaries,
        replay_every=audit_every,
        replay_indices=audit_set,
        reference_seed=reference_seed,
        capture_player_training=capture,
        chance_sampling="event-law",
        audit_mechanics=False,
        actor_collation=args.actor_collation,
        local_reference_nature=args.local_reference_nature,
        inline_player_frame=True,
        structured_curriculum_seed=(
            args.structured_curriculum_seed
            if args.curriculum == "h8-structured" else None),
    )
    episodes, captures = validate_training_cohort(
        h8, seeds, replay_every=audit_every,
        replay_indices=audit_set, require_capture=capture,
    )
    for seed, ep in zip(seeds, episodes, strict=True):
        scenario_id, deck_label, _ = _source_meta(seed, args)
        row = h8["trajectories"][seed]
        if (row["scenario_id"] != scenario_id or
                row["deck_preset_id"] != deck_label):
            raise RuntimeError("H8 returned wrong structured source")
        ep.scenario_id = scenario_id
        ep.deck_preset_id = deck_label
        if ep.reward_version != H4_REWARD:
            raise RuntimeError("H8 reward differs from H7 PPO")
    return {
        "episodes": episodes,
        "player_rollouts": captures,
        "source_count": len(episodes),
        "process_count": h8["workers"],
        "elapsed_seconds": h8["wall_seconds"],
        "actor_metrics": {
            "server": h8["actor_metrics"],
            "rollout_boundaries": h8["all_boundaries"],
            "player_decisions": h8["player_decisions"],
            "terminal_combats_per_second": h8["terminal_combats_per_second"],
        },
        "terminal_replay_verified": len(audit_indices),
        "terminal_replay_not_checked": len(seeds) - len(audit_indices),
        "terminal_replay_audit_seed_digest": (
            None if len(audit_indices) == len(seeds)
            else canonical_sha256([seeds[i] for i in audit_indices])
        ),
    }


def collect(
    seeds: tuple[str, ...], *, spec: H7PlayerSpec, args: Any,
    reference_seed: int, capture: bool,
) -> dict[str, Any]:
    if getattr(args, "rollout_backend", "legacy-h7") == "h8-native":
        return collect_h8(
            seeds, spec=spec, args=args, reference_seed=reference_seed,
            capture=capture)
    # This object is inert: draws are explicitly routed to the original
    # reference Chance Law by draw_mode="reference"; it is never trained.
    inert_nature = PpoDrawNature(seed=109, mix=0.75)
    audit_every = (
        getattr(args, "train_terminal_replay_every", 1)
        if capture else 1
    )
    audit_indices = terminal_replay_audit_indices(seeds, audit_every)
    audit_set = frozenset(audit_indices)
    cohort = draw_only_batch(
        seeds, nature=inert_nature, spec=spec,
        emulator_command=("dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl"),
        emulator_bindings=args.emulator_bindings,
        emulator_revision=args.emulator_revision,
        reference_seed=reference_seed, draw_mode="reference",
        workers=args.workers, max_boundaries=args.max_boundaries,
        scenario_pool=SCENARIO_POOL, deck_pool=DECK_POOL,
        capture_player_training=capture, reward_version=H4_REWARD,
        progress_every=max(1, len(seeds) // 4),
        actor_backend=getattr(args, "actor_backend", "local"),
        actor_device=getattr(args, "actor_device", "cpu"),
        actor_max_batch=getattr(args, "actor_max_batch", 16),
        actor_delay_ms=getattr(args, "actor_delay_ms", 3.0),
        profile_actor=getattr(args, "profile_actor", False),
        terminal_replay_indices=audit_indices,
    )
    episodes = cohort["episodes"]
    if len(episodes) != len(seeds):
        raise RuntimeError("H7 source world lost during collection")
    for index, (seed, ep) in enumerate(zip(seeds, episodes, strict=True)):
        cell = cell_for_source(seed)
        if (cell is None or ep.source_seed != seed
                or ep.scenario_id != cell.scenario
                or ep.deck_preset_id != cell.deck
                or ep.draw_mode != "reference" or ep.reward_version != H4_REWARD
                or ep.outcome not in ("victory", "defeat")
                or ep.terminal_replay_verified != (index in audit_set)):
            raise RuntimeError("H7 nonterminal or incompatible reference-law combat")
    if (cohort["terminal_replay_verified"] != len(audit_indices)
            or cohort["terminal_replay_not_checked"] !=
            len(seeds) - len(audit_indices)):
        raise RuntimeError("H7 source audit count drift")
    return cohort


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--emulator-bindings", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--episodes-per-round", type=int, default=256)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--max-boundaries", type=int, default=1200)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=160)
    p.add_argument("--study-seed", type=int, default=73127)
    p.add_argument("--actor-seed", type=int, default=31337)
    p.add_argument("--reference-seed", type=int, default=23387)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--feedforward", type=int, default=1536)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--ppo-minibatch", type=int, default=64)
    p.add_argument("--ppo-sample-limit", type=int, default=4096)
    p.add_argument("--curriculum", choices=("h5", "h8-structured"),
                   default="h5")
    p.add_argument("--structured-curriculum-seed", type=int,
                   default=20261027)
    p.add_argument("--actor-collation", choices=("legacy", "packed",
                   "packed-contiguous"), default="packed")
    p.add_argument("--local-reference-nature", action="store_true",
                   help="In-emulator source-law Nature for fast H8 rollouts")
    p.add_argument("--rollout-backend", choices=("legacy-h7", "h8-native"),
                   default="legacy-h7",
                   help="Use existing checked semantic H7 or native H8 event-law combat")
    p.add_argument("--actor-backend", choices=("local", "shared"), default="local")
    p.add_argument("--actor-device", default="cpu")
    p.add_argument("--actor-max-batch", type=int, default=16)
    p.add_argument("--actor-delay-ms", type=float, default=3.0)
    p.add_argument("--profile-actor", action="store_true")
    p.add_argument(
        "--train-terminal-replay-every", type=int, default=1,
        help="Audit one terminal per N episodes in each source cell; 1 means all. "
             "Selection/held-out evaluations always replay every terminal.",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if (min(args.rounds, args.episodes_per_round, args.workers,
            args.max_boundaries, args.eval_every, args.ppo_epochs,
            args.ppo_minibatch) < 1 or args.ppo_sample_limit < 2
            or args.episodes_per_round < (256 if args.curriculum ==
                "h8-structured" else len(TRAIN_CELLS))
            or args.eval_episodes < (1 if args.curriculum ==
                "h8-structured" else len(TRAIN_CELLS))
            or (args.curriculum == "h5" and
                args.eval_episodes % len(TRAIN_CELLS))
            or args.study_seed < 0 or args.train_terminal_replay_every < 1
            or args.structured_curriculum_seed < 0):
        p.error("Invalid H7 balanced-source or PPO configuration")
    if args.rollout_backend == "h8-native" and args.actor_backend != "shared":
        p.error("--rollout-backend h8-native requires --actor-backend shared")
    if args.curriculum == "h8-structured" and args.rollout_backend != "h8-native":
        p.error("Structured curriculum requires the native H8 rollout backend")
    if args.local_reference_nature and args.rollout_backend != "h8-native":
        p.error("Local reference Nature requires the native H8 backend")
    config_model = H7Config(
        width=args.width, layers=args.layers, heads=args.heads,
        feedforward=args.feedforward,
    )
    config = {
        "schema": RUN_SCHEMA, "ppo_schema": H7_PPO_SCHEMA,
        "model_schema": H7_MODEL_FORMAT, "object_schema": SHARED_OBJECT_SCHEMA,
        "h7_config": asdict(config_model),
        "emulator_revision": args.emulator_revision,
        "emulator_dll": str(args.emulator_dll.resolve()),
        "emulator_bindings": str(args.emulator_bindings.resolve()),
        "emulator_dll_sha256": (
            file_sha256(args.emulator_dll) if args.emulator_dll.is_file() else None
        ),
        "emulator_bindings_sha256": (
            file_sha256(args.emulator_bindings / "sts2_emulator" / "__init__.py")
            if (args.emulator_bindings / "sts2_emulator" / "__init__.py").is_file()
            else None
        ),
        "chance_law": "reference", "combat_reward": H4_REWARD,
        "curriculum": ("h8-structured-256+64-v1"
                       if args.curriculum == "h8-structured"
                       else "h5-16-train-cells"),
        "training_cells": (256 if args.curriculum == "h8-structured"
                           else len(TRAIN_CELLS)),
        "rounds": args.rounds, "episodes_per_round": args.episodes_per_round,
        "workers": args.workers, "max_boundaries": args.max_boundaries,
        "eval_every": args.eval_every, "eval_episodes": args.eval_episodes,
        "study_seed": args.study_seed, "actor_seed": args.actor_seed,
        "reference_seed": args.reference_seed,
        "device": args.device, "lr": args.lr,
        "temperature": args.temperature,
        "ppo_epochs": args.ppo_epochs, "ppo_minibatch": args.ppo_minibatch,
        "ppo_sample_limit": args.ppo_sample_limit,
    }
    # Preserve legacy H7 local-run fingerprints and resume compatibility.
    # New backend-specific runs are explicitly fingerprinted.
    if args.rollout_backend != "legacy-h7":
        config["structured_curriculum_seed"] = (
            args.structured_curriculum_seed
            if args.curriculum == "h8-structured" else None)
        config["local_reference_nature"] = args.local_reference_nature
        config["actor_collation"] = args.actor_collation
        config["rollout_backend"] = {
            "name": args.rollout_backend,
            "reference_law": "h7-declared-event-scoped-v1",
            "native_optimized": True,
            "mechanics_audit": "stratified-sampled-v1",
        }
    if (args.actor_backend != "local" or args.actor_device != "cpu"
            or args.actor_max_batch != 16 or args.actor_delay_ms != 3.0):
        config["actor_inference"] = {
            "backend": args.actor_backend,
            "device": args.actor_device,
            "max_batch": args.actor_max_batch,
            "delay_ms": args.actor_delay_ms,
        }
    if args.train_terminal_replay_every != 1:
        config["terminal_replay_audit"] = {
            "policy": "stratified-per-cell-preoutcome-v1",
            "train_every": args.train_terminal_replay_every,
            "evaluation_every": 1,
        }
    fingerprint = canonical_sha256(config)
    print(json.dumps({
        "fingerprint": fingerprint, "parameter_target": asdict(config_model),
        "planned_combats": args.rounds * args.episodes_per_round,
        "execute": args.execute,
    }, indent=2), flush=True)
    if not args.execute:
        return
    if not (args.emulator_dll.is_file()
            and (args.emulator_bindings / "sts2_emulator" / "__init__.py").is_file()):
        p.error("Missing pinned semantic emulator DLL or Python bindings")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA Torch is required for --device cuda")
    torch.set_num_threads(1)
    torch.manual_seed(args.study_seed)
    model = SharedObjectTransformer(config_model)
    learner = H7CombatPPO(
        model, device=args.device, lr=args.lr, temperature=args.temperature,
        epochs=args.ppo_epochs, minibatch=args.ppo_minibatch,
        sample_limit=args.ppo_sample_limit,
    )
    checkpoint = args.output_root / "optimizer-checkpoint.pt"
    model_dir = args.output_root / "models"
    progress = args.output_root / "progress.json"
    history: list[dict[str, Any]] = []
    if args.resume:
        if not checkpoint.is_file():
            p.error("H7 --resume requires optimizer-checkpoint.pt")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (saved["schema"] != RUN_SCHEMA or saved["fingerprint"] != fingerprint):
            p.error("H7 resume checkpoint/config fingerprint mismatch")
        learner.model.load_state_dict(saved["model"], strict=True)
        learner.optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        if len(history) > args.rounds:
            p.error("H7 resume history longer than configured run")
        torch.set_rng_state(saved["torch_rng"])
        if args.device.startswith("cuda") and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    elif checkpoint.exists() or (model_dir.exists() and any(model_dir.iterdir())):
        p.error("H7 output contains an existing run; use --resume")
    model_dir.mkdir(parents=True, exist_ok=True)

    for index in range(len(history), args.rounds):
        snapshot = model_dir / f"player-r{index:04d}.pt"
        learner.model.save_checkpoint(snapshot)
        spec = H7PlayerSpec.from_file(
            snapshot, seed=args.actor_seed, temperature=args.temperature,
            greedy=False,
        )
        seeds = _source_bank(
            "train", args.episodes_per_round, args=args, round_index=index,
        )
        batch = collect(
            seeds, spec=spec, args=args,
            reference_seed=args.reference_seed + index * 1000003,
            capture=True,
        )
        diagnostics = learner.update(
            batch["episodes"], batch["player_rollouts"],
            behavior_model_id=f"h7-{spec.model_sha256[:16]}",
            seed=args.study_seed * 1000003 + index,
            allow_sampled_terminal_replay=(
                args.train_terminal_replay_every > 1
            ),
        )
        row: dict[str, Any] = {
            "round": index + 1, "behavior_sha256": spec.model_sha256,
            "source_digest": canonical_sha256(seeds),
            "train": summarize(
                batch["episodes"], curriculum=args.curriculum,
                curriculum_seed=args.structured_curriculum_seed),
            "terminal_replay_audit": {
                "policy": "stratified-per-cell-preoutcome-v1",
                "every": args.train_terminal_replay_every,
                "verified": batch["terminal_replay_verified"],
                "not_checked": batch["terminal_replay_not_checked"],
                "verified_source_digest": batch["terminal_replay_audit_seed_digest"],
            },
            "rollout_seconds": batch["elapsed_seconds"],
            "actor_metrics": batch["actor_metrics"],
            "rollout_backend": args.rollout_backend,
            "training_iteration_seconds": (
                batch["elapsed_seconds"] + diagnostics["total_update_seconds"]
            ),
            **diagnostics,
        }
        if (index + 1) % args.eval_every == 0 or index + 1 == args.rounds:
            selection_model = model_dir / f"player-r{index+1:04d}.pt"
            learner.model.save_checkpoint(selection_model)
            selection = H7PlayerSpec.from_file(
                selection_model, seed=args.actor_seed,
                temperature=1.0, greedy=True,
            )
            eval_seeds = _source_bank(
                "selection", args.eval_episodes, args=args,
            )
            heldout = collect(
                eval_seeds, spec=selection, args=args,
                reference_seed=args.reference_seed + 700000003, capture=False,
            )
            row["selection"] = summarize(
                heldout["episodes"], curriculum=args.curriculum,
                curriculum_seed=args.structured_curriculum_seed)
            row["selection_digest"] = canonical_sha256(eval_seeds)
        history.append(row)
        atomic_torch(checkpoint, {
            "schema": RUN_SCHEMA, "fingerprint": fingerprint,
            "model": learner.model.state_dict(),
            "optimizer": learner.optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if args.device.startswith("cuda") else [],
            "history": history,
        })
        progress.write_text(json.dumps({
            "schema": RUN_SCHEMA, "fingerprint": fingerprint,
            "config": config, "history": history,
        }, indent=2) + "\n", encoding="utf-8")
        batching = batch["actor_metrics"].get("server", {})
        if batching:
            requests, batches = batching.get("requests", 0), batching.get("batches", 0)
            print(
                f"[h7:{args.rollout_backend}:actor] "
                f"requests={requests} batches={batches} "
                f"mean_batch={requests / batches if batches else 0:.2f} "
                f"forward_s={batching.get('forward_seconds', 0):.3f} "
                f"collate_s={batching.get('collate_seconds', 0):.3f}",
                flush=True,
            )
        print(
            f"[h7:{args.rollout_backend}] round={index+1}/{args.rounds} "
            f"wins={row['train']['wins']}/{row['train']['combats']} "
            f"score={row['train']['mean_score']:.4f} "
            f"decisions={row['decisions']} kl={row['approx_kl']:.5f} "
            f"rollout_s={row['rollout_seconds']:.1f} "
            f"ppo_s={row['total_update_seconds']:.1f} "
            f"ratio_verify_s={row['ratio_verification_seconds']:.1f} "
            f"optimizer_s={row['optimizer_seconds']:.1f}",
            flush=True,
        )
    learner.model.save_checkpoint(args.output_root / "final-player.pt")
    print(f"[h7] complete checkpoint={checkpoint}", flush=True)


if __name__ == "__main__":
    main()
