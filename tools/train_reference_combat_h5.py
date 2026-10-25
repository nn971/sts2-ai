#!/usr/bin/env python3
"""H5 reference-law-only player PPO across supported Silent A10 combat cells.

No learned Nature, no MCTS, and no fidelity inspector dependencies.
This initial curriculum uses 16 explicitly curated source-conditioned cells;
it is an executable bridge toward larger engine-supported encounter/deck pools,
NOT yet hundreds of independently customized card/relic/HP builds.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

from sts2_ai.emulator.semantic_p0_frozen_player import FrozenPlayerSpec
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.nature.corpus import atomic_write, file_sha256
from sts2_ai.nature.draw_only_workers import draw_only_batch
from sts2_ai.nature.h4_reward import H4_REWARD
from sts2_ai.nature.h5_curriculum import (
    SCHEMA as CURRICULUM_SCHEMA, SCENARIO_POOL, DECK_POOL,
    TRAIN_CELLS, cell_for_source, source_bank, source_cells,
)
from sts2_ai.nature.joint_player_ppo import JointPlayerPPO
from sts2_ai.nature.ppo_draw import PpoDrawNature
from sts2_ai.nature.protocol import canonical_sha256

SCHEMA = "sts2-ai-h5-reference-only-player-ppo-v1"


def _atomic_torch(path: Path, value: dict[str, Any], torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".pending", delete=False,
        ) as stream:
            pending = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def _spec(model: Path, actor_seed: int, temperature: float,
          *, greedy: bool) -> FrozenPlayerSpec:
    return FrozenPlayerSpec.from_file(
        model, seed=actor_seed, temperature=temperature,
        greedy=greedy, action_scope="all-legal-v1",
    )


def _batch(seeds: tuple[str, ...], *, args: Any, model_path: Path,
           capture: bool, nature: PpoDrawNature, command: tuple[str, ...],
           reference_seed: int) -> dict[str, Any]:
    spec = _spec(
        model_path, args.actor_seed, (
            args.player_temperature if capture else 1.0
        ), greedy=not capture,
    )
    cohort = draw_only_batch(
        seeds, nature=nature, spec=spec, emulator_command=command,
        emulator_bindings=args.emulator_bindings,
        emulator_revision=args.emulator_revision,
        reference_seed=reference_seed,
        workers=args.workers, max_boundaries=args.max_boundaries,
        scenario_pool=SCENARIO_POOL, deck_pool=DECK_POOL,
        capture_player_training=capture,
        draw_mode="reference", reward_version=H4_REWARD,
        progress_every=max(1, len(seeds)//4),
    )
    if len(cohort["episodes"]) != len(seeds):
        raise RuntimeError("H5 lost reference-law combat trajectories")
    for wanted, ep in zip(seeds, cohort["episodes"], strict=True):
        cell = cell_for_source(wanted)
        if (cell is None or ep.source_seed != wanted
                or ep.draw_mode != "reference"
                or ep.scenario_id != cell.scenario
                or ep.deck_preset_id != cell.deck
                or not ep.terminal_replay_verified):
            raise RuntimeError("H5 source/cell provenance mismatch")
    return cohort


def _summary(episodes: tuple) -> dict:
    by_cell: dict[str, list] = {}
    for ep in episodes:
        cell = cell_for_source(ep.source_seed)
        if cell is None:
            raise ValueError("Unrecognized H5 source cell")
        by_cell.setdefault(cell.key, []).append(ep)
    def stats(items: list | tuple) -> dict[str, float | int]:
        return {
            "combats": len(items),
            "score": statistics.mean(ep.score for ep in items),
            "win_rate": statistics.mean(ep.outcome == "victory" for ep in items),
            "exit_hp": statistics.mean(ep.exit_hp for ep in items),
        }
    return {
        **stats(episodes),
        "by_cell": {key: stats(items)
                    for key, items in sorted(by_cell.items())},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--player-model", type=Path, required=True)
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--emulator-bindings", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--episodes-per-round", type=int, default=256)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--max-boundaries", type=int, default=1200)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=160)
    p.add_argument("--study-seed", type=int, default=29)
    p.add_argument("--actor-seed", type=int, default=31337)
    p.add_argument("--reference-seed", type=int, default=23387)
    p.add_argument("--player-lr", type=float, default=0.0003)
    p.add_argument("--player-temperature", type=float, default=0.85)
    p.add_argument("--player-ppo-epochs", type=int, default=3)
    p.add_argument("--player-ppo-batch-size", type=int, default=128)
    p.add_argument("--player-ppo-sample-limit", type=int, default=2048)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()

    if (args.rounds < 1 or args.episodes_per_round < len(TRAIN_CELLS)
            or args.eval_episodes < len(TRAIN_CELLS)
            or args.eval_episodes % len(TRAIN_CELLS)
            or min(args.workers, args.max_boundaries, args.eval_every,
                   args.player_ppo_epochs, args.player_ppo_batch_size,
                   args.player_ppo_sample_limit) < 1
            or args.study_seed < 0 or args.player_lr <= 0
            or args.player_temperature <= 0):
        p.error("Invalid H5 training/evaluation hyperparameters")
    binding = args.emulator_bindings/"sts2_emulator"/"__init__.py"
    if not (args.player_model.is_file() and args.emulator_dll.is_file()
            and binding.is_file()):
        p.error("Missing pretrained player or pinned semantic emulator inputs")

    initial_spec = _spec(
        args.player_model, args.actor_seed, args.player_temperature,
        greedy=False,
    )
    config = {
        "schema": SCHEMA,
        "curriculum_schema": CURRICULUM_SCHEMA,
        "training_cells": [cell.key for cell in TRAIN_CELLS],
        "initial_player_sha256": initial_spec.model_sha256,
        "emulator_revision": args.emulator_revision,
        "emulator_dll_sha256": file_sha256(args.emulator_dll),
        "emulator_bindings_sha256": file_sha256(binding),
        "rounds": args.rounds,
        "episodes_per_round": args.episodes_per_round,
        "workers": args.workers,
        "max_boundaries": args.max_boundaries,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "eval_split": "selection",
        "study_seed": args.study_seed,
        "actor_seed": args.actor_seed,
        "reference_seed": args.reference_seed,
        "player_lr": args.player_lr,
        "player_temperature": args.player_temperature,
        "player_ppo_epochs": args.player_ppo_epochs,
        "player_ppo_batch_size": args.player_ppo_batch_size,
        "player_ppo_sample_limit": args.player_ppo_sample_limit,
        "draw_mode": "reference",
        "reward_version": H4_REWARD,
    }
    fingerprint = canonical_sha256(config)
    print(f"[h5] config={fingerprint[:16]} cells={len(TRAIN_CELLS)} "
          f"planned_combats={args.rounds*args.episodes_per_round} "
          f"workers={args.workers} execute={args.execute}", flush=True)
    if not args.execute:
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    import torch
    torch.set_num_threads(1)
    torch.manual_seed(args.study_seed)
    player = JointPlayerPPO(
        PhaseSplitNeuralModel.load(args.player_model),
        torch=torch, learning_rate=args.player_lr,
        temperature=args.player_temperature,
        ppo_epochs=args.player_ppo_epochs,
        batch_size=args.player_ppo_batch_size,
        sample_limit=args.player_ppo_sample_limit,
        gae_lambda=1.0,
    )
    # Inert protocol adapter. All physical draws and enemy intents use
    # DeclaredLaw through draw_mode="reference"; Nature is never updated.
    nature = PpoDrawNature(seed=109, mix=0.75)
    checkpoint = args.output_root/"player-checkpoint.pt"
    models_dir = args.output_root/"player_models"
    rows: list[dict] = []
    if args.resume:
        if not checkpoint.is_file():
            p.error("H5 --resume requires player-checkpoint.pt")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (saved["schema"] != SCHEMA
                or saved["fingerprint"] != fingerprint):
            p.error("H5 resume configuration/checkpoint mismatch")
        player.restore(saved["player"])
        rows = saved["rows"]
        if player.revision != len(rows) or len(rows) > args.rounds:
            p.error("H5 checkpoint revision mismatch")
        torch.set_rng_state(saved["torch_rng_state"])
    else:
        if checkpoint.exists() or (
            models_dir.exists() and any(models_dir.iterdir())
        ):
            p.error("H5 output already contains a run; use --resume")
    args.output_root.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    command = ("dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl")

    for index in range(len(rows), args.rounds):
        behavior = models_dir/f"player-r{player.revision:04d}.json"
        atomic_write(
            behavior, json.dumps(player.model.to_dict(), sort_keys=True)+"\n",
        )
        seeds = source_bank(
            "train", args.episodes_per_round,
            study_seed=args.study_seed, round_index=index,
        )
        cohort = _batch(
            seeds, args=args, model_path=behavior, capture=True,
            nature=nature, command=command,
            reference_seed=args.reference_seed+index*1000003,
        )
        episodes = cohort["episodes"]
        captures = cohort["player_rollouts"]
        if len(captures) != len(episodes) or cohort["process_count"] < 1:
            raise RuntimeError("H5 incomplete player trajectory provenance")
        for ep, rec in zip(episodes, captures, strict=True):
            JointPlayerPPO.validate_capture(ep, rec, player.model)
        stats = player.update(
            tuple(zip(episodes, captures, strict=True)),
            seed=args.study_seed*1000003+index,
        )
        if player.revision != index+1:
            raise RuntimeError("H5 PPO player revision drift")
        row = {
            "round": index+1,
            "behavior_player_sha256": file_sha256(behavior),
            "source_digest": canonical_sha256(seeds),
            "source_cells": source_cells(seeds),
            "train": _summary(episodes),
            "rollout_seconds": cohort["elapsed_seconds"],
            "process_count": cohort["process_count"],
            **stats,
        }
        if (index+1) % args.eval_every == 0 or index+1 == args.rounds:
            current = models_dir/f"player-r{player.revision:04d}.json"
            atomic_write(
                current, json.dumps(player.model.to_dict(), sort_keys=True)+"\n",
            )
            holdout_seeds = source_bank(
                "selection", args.eval_episodes,
                study_seed=args.study_seed,
            )
            evaluation = _batch(
                holdout_seeds, args=args, model_path=current,
                capture=False, nature=nature, command=command,
                reference_seed=args.reference_seed+700000009,
            )
            row["selection"] = _summary(evaluation["episodes"])
            row["selection"]["source_digest"] = canonical_sha256(holdout_seeds)
        rows.append(row)
        _atomic_torch(checkpoint, {
            "schema": SCHEMA, "config": config, "fingerprint": fingerprint,
            "player": player.dump(), "rows": rows,
            "torch_rng_state": torch.get_rng_state(),
            "next_round": index+1,
            "native_fidelity": "INDEPENDENT_NOT_TRAINING_GATE",
        }, torch)
        atomic_write(
            args.output_root/"progress.json",
            json.dumps({"config": config, "rows": rows},
                       indent=2, sort_keys=True)+"\n",
        )
        selected = row.get("selection")
        print(
            f"[h5] round={index+1}/{args.rounds} "
            f"train_score={row['train']['score']:.4f} "
            f"train_win={row['train']['win_rate']:.3f} "
            f"ppo_steps={int(stats['player_ppo_steps'])} "
            f"ppo_kl={stats['player_ppo_kl']:.5f}"
            + (f" selection_score={selected['score']:.4f} "
               f"selection_win={selected['win_rate']:.3f}"
               if selected is not None else ""),
            flush=True,
        )


if __name__ == "__main__":
    main()
