#!/usr/bin/env python3
"""H3 shared-rollout, alternating player/Nature PPO combat training.

One immutable cohort freezes both policies. Reference-law draws are mixed in
for player robustness; only genuine learned-draw episodes train Nature.
Native-game fidelity is NOT a training dependency.
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
from sts2_ai.nature.combat_scenarios import (
    parse_deck_pool,
    parse_scenario_pool,
)
from sts2_ai.nature.contextual_draw import restore_nature
from sts2_ai.nature.corpus import atomic_write, file_sha256
from sts2_ai.nature.draw_only_workers import draw_only_batch
from sts2_ai.nature.h2_benchmark import (
    balanced_h2_source_seeds,
    balanced_h2_training_source_seeds,
)
from sts2_ai.nature.joint_player_ppo import JointPlayerPPO
from sts2_ai.nature.ppo_draw import PpoDrawNature
from sts2_ai.nature.protocol import canonical_sha256

SCHEMA = "sts2-ai-h3-joint-player-nature-ppo-v1"


def _atomic_torch(path: Path, value: dict[str, Any], torch: Any) -> None:
    """Replace one pair-state artifact atomically, including both optimizers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".pending", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _spec(
    path: Path, seed: int, temperature: float, *, greedy: bool,
) -> FrozenPlayerSpec:
    return FrozenPlayerSpec.from_file(
        path, seed=seed, temperature=temperature, greedy=greedy,
        action_scope="all-legal-v1",
    )


def _evaluate(
    *, nature: PpoDrawNature, model_path: Path, args: Any,
    scenario_pool: tuple[str, ...], deck_pool: tuple[str, ...],
    command: tuple[str, ...],
) -> dict[str, float]:
    # Holdout is not used by either optimizer. Use the same source worlds for
    # the reference vs Nature arms under the *current* greedy player.
    seeds = balanced_h2_source_seeds(
        "selection", args.eval_episodes, scenario_pool, deck_pool,
    )
    spec = _spec(model_path, args.actor_seed + 7000, 1.0, greedy=True)
    kwargs = dict(
        nature=nature, spec=spec, emulator_command=command,
        emulator_bindings=args.emulator_bindings,
        emulator_revision=args.emulator_revision,
        reference_seed=args.reference_seed + 700000009,
        workers=args.workers, max_boundaries=args.max_boundaries,
        scenario_pool=scenario_pool, deck_pool=deck_pool,
        progress_every=max(1, args.eval_episodes),
    )
    reference = draw_only_batch(seeds, draw_mode="reference", **kwargs)
    learned = draw_only_batch(seeds, draw_mode="learned", **kwargs)
    for left, right in zip(
        reference["episodes"], learned["episodes"], strict=True
    ):
        if (left.source_seed != right.source_seed
                or left.scenario_id != right.scenario_id
                or left.deck_preset_id != right.deck_preset_id
                or left.source_opening_fingerprint !=
                right.source_opening_fingerprint):
            raise ValueError("Unpaired H3 evaluation source world")
    refs = reference["episodes"]
    evils = learned["episodes"]
    return {
        "reference_score": statistics.mean(x.score for x in refs),
        "reference_victory": statistics.mean(
            x.outcome == "victory" for x in refs
        ),
        "nature_score": statistics.mean(x.score for x in evils),
        "nature_victory": statistics.mean(
            x.outcome == "victory" for x in evils
        ),
        "nature_minus_reference_score": statistics.mean(
            b.score - a.score for a, b in zip(refs, evils, strict=True)
        ),
        "eval_source_count": len(seeds),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--player-model", type=Path, required=True)
    p.add_argument("--nature-checkpoint", type=Path,
                   help="H2 Nature checkpoint.json (warm-start adversary)")
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--emulator-bindings", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--scenario-pool",
                   default="mawler-a10,cubex-a10,vine-shambler-a10")
    p.add_argument("--deck-pool",
                   default=("silent-starter-v1,silent-strike-heavy-v1,"
                            "silent-defend-heavy-v1,silent-poison-kit-v1"))
    p.add_argument("--training-seed", type=int, default=29)
    p.add_argument("--actor-seed", type=int, default=31337)
    p.add_argument("--reference-seed", type=int, default=23387)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--episodes-per-round", type=int, default=256)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--max-boundaries", type=int, default=600)
    p.add_argument("--reference-fraction", type=float, default=0.25)
    p.add_argument("--player-lr", type=float, default=0.0003)
    p.add_argument("--player-temperature", type=float, default=0.85)
    p.add_argument("--player-ppo-epochs", type=int, default=3)
    p.add_argument("--player-ppo-sample-limit", type=int, default=2048)
    p.add_argument("--player-ppo-batch-size", type=int, default=128)
    p.add_argument("--nature-start-lr", type=float, default=0.001)
    p.add_argument("--nature-mix", type=float, default=0.75)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=120)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    try:
        scenarios = parse_scenario_pool(args.scenario_pool)
        decks = parse_deck_pool(args.deck_pool)
    except ValueError as exc:
        p.error(str(exc))
    if any(not key.endswith("-a10") for key in scenarios):
        p.error("H3 initial study is restricted to A10")
    if (min(args.rounds, args.episodes_per_round, args.workers,
            args.eval_every, args.eval_episodes, args.max_boundaries) < 1
            or args.episodes_per_round < 8
            or not 0 < args.reference_fraction < 0.75
            or int(args.episodes_per_round * (1-args.reference_fraction)) < 2
            or args.eval_episodes % (len(scenarios) * len(decks))):
        p.error("Invalid H3 cohort sizes, reference mix, or evaluation balance")
    if (not 0 < args.player_lr < 1
            or not 0 < args.nature_start_lr < 1
            or not 0 < args.player_temperature
            or not 0 < args.nature_mix < 1):
        p.error("Invalid H3 PPO learning settings")
    if not args.player_model.is_file() or not args.emulator_dll.is_file():
        p.error("Missing player model or pinned emulator DLL")
    if not (args.emulator_bindings/"sts2_emulator"/"__init__.py").is_file():
        p.error("Missing emulator binding")
    if args.nature_checkpoint and not args.nature_checkpoint.is_file():
        p.error("Missing optional H2 Nature checkpoint")
    starter = PhaseSplitNeuralModel.load(args.player_model)
    initial_spec = _spec(
        args.player_model, args.actor_seed,
        args.player_temperature, greedy=False,
    )
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(args.training_seed)
    player = JointPlayerPPO(
        starter, torch=torch, learning_rate=args.player_lr,
        temperature=args.player_temperature,
        ppo_epochs=args.player_ppo_epochs,
        batch_size=args.player_ppo_batch_size,
        sample_limit=args.player_ppo_sample_limit,
    )
    initial_nature_digest = (
        file_sha256(args.nature_checkpoint) if args.nature_checkpoint else None
    )
    if args.nature_checkpoint is None:
        nature = PpoDrawNature(
            seed=args.training_seed+100, mix=args.nature_mix,
            learning_rate=args.nature_start_lr,
        )
    else:
        raw = json.loads(args.nature_checkpoint.read_text(encoding="utf-8"))
        nature = restore_nature(raw["nature"])
        if type(nature) is not PpoDrawNature:
            p.error("Warm-start Nature must be H2 PPO, not tabular/REINFORCE")
        if abs(nature.mix - args.nature_mix) > 1e-12:
            p.error("Nature mix differs from pretrained Nature checkpoint")
    config = {
        "schema": SCHEMA,
        "initial_player_sha256": initial_spec.model_sha256,
        "initial_nature_checkpoint_sha256": initial_nature_digest,
        "emulator_revision": args.emulator_revision,
        "emulator_dll_sha256": file_sha256(args.emulator_dll),
        "emulator_bindings_sha256": file_sha256(
            args.emulator_bindings/"sts2_emulator"/"__init__.py"
        ),
        "scenario_pool": sorted(scenarios), "deck_pool": sorted(decks),
        "training_seed": args.training_seed,
        "actor_seed": args.actor_seed,
        "reference_seed": args.reference_seed,
        "rounds": args.rounds,
        "episodes_per_round": args.episodes_per_round,
        "workers": args.workers,
        "max_boundaries": args.max_boundaries,
        "reference_fraction": args.reference_fraction,
        "player_lr": args.player_lr,
        "player_temperature": args.player_temperature,
        "player_ppo_epochs": args.player_ppo_epochs,
        "player_ppo_sample_limit": args.player_ppo_sample_limit,
        "player_ppo_batch_size": args.player_ppo_batch_size,
        "nature_start_lr": args.nature_start_lr,
        "nature_mix": args.nature_mix,
        "eval_every": args.eval_every, "eval_episodes": args.eval_episodes,
    }
    fingerprint = canonical_sha256(config)
    print(
        f"[h3] config={fingerprint[:16]} cells={len(scenarios)*len(decks)} "
        f"combats={args.rounds*args.episodes_per_round} "
        f"workers={args.workers} execute={args.execute}",
        flush=True,
    )
    if not args.execute:
        print(json.dumps(config, sort_keys=True, indent=2))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_root / "joint-checkpoint.pt"
    models_dir = args.output_root/"player_models"
    models_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    if args.resume:
        if not checkpoint.is_file():
            p.error("H3 --resume requires joint-checkpoint.pt")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["schema"] != SCHEMA or saved["fingerprint"] != fingerprint:
            p.error("H3 checkpoint configuration mismatch")
        player.restore(saved["player"])
        nature = restore_nature(saved["nature"])
        if type(nature) is not PpoDrawNature:
            p.error("H3 resumed Nature has wrong architecture")
        rows = saved["rows"]
        if len(rows) != player.revision or len(rows) > args.rounds:
            p.error("H3 checkpoint round index mismatch")
        torch.set_rng_state(saved["torch_rng_state"])
    else:
        if checkpoint.exists() or any(models_dir.iterdir()):
            p.error("H3 output contains existing training: use --resume")
    command = ("dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl")
    for index in range(len(rows), args.rounds):
        model_path = models_dir/f"player-r{player.revision:04d}.json"
        atomic_write(
            model_path,
            json.dumps(player.model.to_dict(), sort_keys=True) + "\n",
        )
        spec = _spec(
            model_path, args.actor_seed, args.player_temperature,
            greedy=False,
        )
        seeds = balanced_h2_training_source_seeds(
            args.training_seed, index, args.episodes_per_round,
            scenarios, decks,
        )
        reference_count = int(args.episodes_per_round * args.reference_fraction)
        cohort = draw_only_batch(
            seeds, nature=nature, spec=spec,
            emulator_command=command,
            emulator_bindings=args.emulator_bindings,
            emulator_revision=args.emulator_revision,
            reference_seed=args.reference_seed+index*1000003,
            workers=args.workers,
            max_boundaries=args.max_boundaries,
            scenario_pool=scenarios, deck_pool=decks,
            capture_player_training=True,
            reference_count=reference_count,
        )
        episodes = cohort["episodes"]
        records = cohort["player_rollouts"]
        if cohort["process_count"] < 1 or len(records) != len(episodes):
            raise RuntimeError("H3 missing public player rollout")
        # Do NOT update Nature until both behavior policies have been
        # fully validated: otherwise half-applied cohorts could train stale.
        for ep, rec in zip(episodes, records, strict=True):
            JointPlayerPPO.validate_capture(ep, rec, player.model)
        nature_episodes = tuple(
            ep.for_nature_training()
            for ep in episodes if ep.draw_mode == "learned"
        )
        player_stats = player.update(
            tuple(zip(episodes, records, strict=True)),
            seed=args.training_seed*1000003+index,
        )
        nature_stats = nature.update(nature_episodes)
        if player.revision != index+1:
            raise RuntimeError("Player joint PPO revision desynchronized")
        record = {
            "round": index+1,
            "behavior_player_sha256": spec.model_sha256,
            "behavior_nature_revision": nature.revision-1,
            "source_digest": canonical_sha256(seeds),
            "reference_training_combats": reference_count,
            "learned_nature_training_combats": len(nature_episodes),
            "mean_score": statistics.mean(ep.score for ep in episodes),
            "player_victory_rate": statistics.mean(
                ep.outcome == "victory" for ep in episodes
            ),
            "emulator_pids": cohort["process_count"],
            "rollout_seconds": cohort["elapsed_seconds"],
            **player_stats,
            **nature_stats,
        }
        if ((index+1) % args.eval_every == 0 or index+1 == args.rounds):
            evaluation_path = models_dir/f"player-r{player.revision:04d}.json"
            atomic_write(
                evaluation_path,
                json.dumps(player.model.to_dict(), sort_keys=True)+"\n",
            )
            record["holdout"] = _evaluate(
                nature=nature, model_path=evaluation_path, args=args,
                scenario_pool=scenarios, deck_pool=decks, command=command,
            )
        rows.append(record)
        _atomic_torch(checkpoint, {
            "schema": SCHEMA, "config": config, "fingerprint": fingerprint,
            "player": player.dump(), "nature": nature.dump(),
            "rows": rows, "torch_rng_state": torch.get_rng_state(),
            "next_round": index+1,
            "native_probability_status": "UNVERIFIED_NOT_TRAINING_GATE",
        }, torch)
        atomic_write(
            args.output_root/"progress.json",
            json.dumps({"config": config, "rows": rows}, indent=2,
                       sort_keys=True) + "\n",
        )
        holdout = record.get("holdout")
        print(
            f"[h3] round={index+1}/{args.rounds} "
            f"score={record['mean_score']:.4f} "
            f"player_ppo_steps={int(player_stats['player_ppo_steps'])} "
            f"nature_ppo_steps={int(nature_stats['nature_optimizer_steps'])} "
            f"player_kl={player_stats['player_ppo_kl']:.5f} "
            f"nature_kl={nature_stats['nature_ppo_approx_kl']:.5f} "
            + (f"ref_eval={holdout['reference_score']:.4f} "
               f"nature_eval={holdout['nature_score']:.4f}"
               if holdout is not None else "") ,
            flush=True,
        )


if __name__ == "__main__":
    main()
