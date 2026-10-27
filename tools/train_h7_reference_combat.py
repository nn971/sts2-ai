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
    source_bank as h5_source_bank,
)
from sts2_ai.nature.combat_scenarios import BASELINE_DECK
from sts2_ai.nature.research_combat_fixtures import (
    audited_corpus,
    fixture_from_source,
    source_bank as h8_source_bank,
)
from sts2_ai.nature.ppo_draw import PpoDrawNature
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.training.h7_combat_ppo import H7_PPO_SCHEMA, H7CombatPPO

RUN_SCHEMA = "sts2-h7-reference-combat-run-v1"


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
        if curriculum == "h8":
            _, fixture, index = fixture_from_source(
                ep.source_seed, curriculum_seed=curriculum_seed,
            )
            key = f"fixture-{index:04d}:{fixture['encounter_id']}"
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


def collect(
    seeds: tuple[str, ...], *, spec: H7PlayerSpec, args: Any,
    reference_seed: int, capture: bool,
) -> dict[str, Any]:
    # This object is inert: draws are explicitly routed to the original
    # reference Chance Law by draw_mode="reference"; it is never trained.
    inert_nature = PpoDrawNature(seed=109, mix=0.75)
    cohort = draw_only_batch(
        seeds, nature=inert_nature, spec=spec,
        emulator_command=("dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl"),
        emulator_bindings=args.emulator_bindings,
        emulator_revision=args.emulator_revision,
        reference_seed=reference_seed, draw_mode="reference",
        workers=args.workers, max_boundaries=args.max_boundaries,
        scenario_pool=SCENARIO_POOL, deck_pool=DECK_POOL,
        structured_curriculum_seed=(
            args.curriculum_seed if args.curriculum == "h8" else None
        ),
        capture_player_training=capture, reward_version=H4_REWARD,
        progress_every=max(1, len(seeds) // 4),
    )
    episodes = cohort["episodes"]
    if len(episodes) != len(seeds):
        raise RuntimeError("H7 source world lost during collection")
    for seed, ep in zip(seeds, episodes, strict=True):
        if args.curriculum == "h8":
            chosen, fixture, _ = fixture_from_source(
                seed, curriculum_seed=args.curriculum_seed,
            )
            expected_scenario, expected_deck = chosen.scenario_id, BASELINE_DECK
            if fixture["hp"] != ep.entry_hp:
                raise RuntimeError("H8 starting HP differs from validated fixture")
        else:
            cell = cell_for_source(seed)
            if cell is None:
                raise RuntimeError("H7 uncurated source cell")
            expected_scenario, expected_deck = cell.scenario, cell.deck
        if (ep.source_seed != seed
                or ep.scenario_id != expected_scenario
                or ep.deck_preset_id != expected_deck
                or ep.draw_mode != "reference" or ep.reward_version != H4_REWARD
                or ep.outcome not in ("victory", "defeat")
                or not ep.terminal_replay_verified):
            raise RuntimeError("H7 nonterminal or incompatible reference-law combat")
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
    p.add_argument("--curriculum", choices=("h5", "h8"), default="h5")
    p.add_argument("--curriculum-seed", type=int, default=20261027)
    p.add_argument("--train-specs", type=int, default=256)
    p.add_argument("--heldout-specs", type=int, default=64)
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
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if (min(args.rounds, args.episodes_per_round, args.workers,
            args.max_boundaries, args.eval_every, args.ppo_epochs,
            args.ppo_minibatch) < 1 or args.ppo_sample_limit < 2
            or (args.curriculum == "h5" and (
                args.episodes_per_round < len(TRAIN_CELLS)
                or args.eval_episodes < len(TRAIN_CELLS)
                or args.eval_episodes % len(TRAIN_CELLS)))
            or args.curriculum_seed < 0
            or (args.curriculum == "h8" and (
                args.train_specs != 256 or args.heldout_specs != 64))
            or args.study_seed < 0):
        p.error("Invalid H7 balanced-source or PPO configuration")
    if args.curriculum == "h8":
        # Structural census is cheap and must finish before any GPU/process use.
        corpus = audited_corpus(
            args.train_specs + args.heldout_specs,
            curriculum_seed=args.curriculum_seed,
        )
        if len({str(item) for item in corpus}) != len(corpus):
            p.error("H8 structural fixture duplication detected")
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
        "curriculum": (
            "h8-structured-a10-v1" if args.curriculum == "h8"
            else "h5-16-train-cells"
        ),
        "training_cells": args.train_specs if args.curriculum == "h8"
                          else len(TRAIN_CELLS),
        "curriculum_seed": args.curriculum_seed if args.curriculum == "h8" else None,
        "heldout_specs": args.heldout_specs if args.curriculum == "h8" else None,
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
        seeds = (
            h8_source_bank(
                "train", args.episodes_per_round, study_seed=args.study_seed,
                round_index=index,
            ) if args.curriculum == "h8"
            else h5_source_bank(
                "train", args.episodes_per_round,
                study_seed=args.study_seed, round_index=index,
            )
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
        )
        row: dict[str, Any] = {
            "round": index + 1, "behavior_sha256": spec.model_sha256,
            "source_digest": canonical_sha256(seeds),
            "train": summarize(batch["episodes"], curriculum=args.curriculum,
                               curriculum_seed=args.curriculum_seed),
            "rollout_seconds": batch["elapsed_seconds"],
            **diagnostics,
        }
        if (index + 1) % args.eval_every == 0 or index + 1 == args.rounds:
            selection_model = model_dir / f"player-r{index+1:04d}.pt"
            learner.model.save_checkpoint(selection_model)
            selection = H7PlayerSpec.from_file(
                selection_model, seed=args.actor_seed,
                temperature=1.0, greedy=True,
            )
            eval_seeds = (
                h8_source_bank(
                    "selection", args.eval_episodes, study_seed=args.study_seed,
                ) if args.curriculum == "h8"
                else h5_source_bank(
                    "selection", args.eval_episodes, study_seed=args.study_seed,
                )
            )
            heldout = collect(
                eval_seeds, spec=selection, args=args,
                reference_seed=args.reference_seed + 700000003, capture=False,
            )
            row["selection"] = summarize(heldout["episodes"], curriculum=args.curriculum,
                                         curriculum_seed=args.curriculum_seed)
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
        print(
            f"[h7] round={index+1}/{args.rounds} "
            f"wins={row['train']['wins']}/{row['train']['combats']} "
            f"score={row['train']['mean_score']:.4f} "
            f"decisions={row['decisions']} kl={row['approx_kl']:.5f} "
            f"rollout_s={row['rollout_seconds']:.1f}",
            flush=True,
        )
    learner.model.save_checkpoint(args.output_root / "final-player.pt")
    print(f"[h7] complete checkpoint={checkpoint}", flush=True)


if __name__ == "__main__":
    main()
