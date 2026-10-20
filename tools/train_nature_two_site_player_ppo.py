#!/usr/bin/env python3
"""Train v15 pooled/relational combat PPO from genuine P0 semantic combats.

This is an opt-in source-pinned two-site training pilot. Behavior is exported
once each round, loaded read-only in independent emulator workers, and
validated against the SAME frozen pre-update Torch residual before PPO.
The player and Nature are updated only after completed terminal episodes.
No Steam/native-game law, full Silent cards, or model-performance claims.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from sts2_ai.emulator.semantic_p0_frozen_player import FrozenPlayerSpec
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.models.pooled_objects import (
    PooledObjectTorch,
    PortablePooledObjectModel,
)
from sts2_ai.models.relational_objects import (
    PortableRelationalObjectModel,
    RelationalObjectTorch,
)
from sts2_ai.nature.corpus import atomic_write, file_sha256
from sts2_ai.nature.paired_real_combat import PairedSourceFixture
from sts2_ai.nature.paired_real_workers import paired_batch
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.two_site_opponent import PilotChoice, PilotEpisode, TwoSiteNature
from sts2_ai.training.nature_real_combat_ppo import prepare_real_combat_ppo_samples
from sts2_ai.training.pooled_ppo_v15 import (
    initialize_combat_head,
    ppo_update_pooled,
)

SCHEMA = "sts2-ai-p0-real-combat-v15-ppo-training-v1"


def _read_source_train(path: Path) -> tuple[PairedSourceFixture, ...]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw.get("partition") != "train":
            raise ValueError(
                f"Source-run {number} is not marked train: never leak holdout to PPO"
            )
        rows.append(PairedSourceFixture(
            raw["fixture_id"], raw["seed"], raw["region"],
        ))
    if len({(r.region, r.seed) for r in rows}) != len(rows):
        raise ValueError("Duplicate source-run group across PPO training")
    if len({r.fixture_id for r in rows}) != len(rows):
        raise ValueError("Duplicate source fixture ID")
    return tuple(rows)


def _save_torch_atomic(path: Path, payload: dict, torch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".pending", delete=False,
        ) as fp:
            temporary = Path(fp.name)
            torch.save(payload, fp)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initial-model", type=Path, required=True,
                   help="Actual trained v15c pooled or v15d relational exported model.json")
    p.add_argument("--source-fixtures", type=Path, required=True,
                   help="Source-group disjoint train-only JSONL")
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--emulator-bindings", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--training-arm", choices=("nature", "ordinary"), required=True)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--episodes-per-round", type=int, default=15)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--nature-mix", type=float, default=0.75)
    p.add_argument("--nature-update", action="store_true",
                   help="Alternate terminal Nature REINFORCE update in Nature arm")
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sample-limit", type=int, default=2048)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.98)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    if args.nature_update and args.training_arm != "nature":
        p.error("Nature updates only belong in the Nature training arm")
    if min(args.rounds, args.episodes_per_round, args.workers) <= 0:
        p.error("Rounds, episode count and worker count must be positive")
    if not args.emulator_dll.is_file():
        p.error("Pinned C# emulator CLI DLL is missing")
    if not (args.emulator_bindings / "sts2_emulator" / "__init__.py").is_file():
        p.error("Pinned emulator bindings are missing")
    if args.nature_update and args.episodes_per_round < 2:
        p.error("Nature REINFORCE requires at least two terminal episodes per batch")
    source = _read_source_train(args.source_fixtures)
    if len(source) < args.rounds * args.episodes_per_round:
        p.error("Insufficient distinct training source groups for requested rounds")

    import torch

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    initial = PhaseSplitNeuralModel.load(args.initial_model)
    if initial.combat_value_objective != "hp_preservation":
        p.error("v15 combat model must declare hp_preservation critic objective")
    combat = initial.combat
    if isinstance(combat, PortableRelationalObjectModel):
        head = RelationalObjectTorch(combat.vocabulary, width=combat.width)
    elif isinstance(combat, PortablePooledObjectModel):
        head = PooledObjectTorch(combat.vocabulary, width=combat.width)
    else:
        p.error("P0 v15 PPO requires a pretrained portable pooled or relational head")
    base = PhaseSplitNeuralModel(
        initial.strategy, combat.base, "frozen-base-for-P0-nature",
        combat_value_objective="hp_preservation",
    )
    initialize_combat_head(head, base, initial)
    optimizer = torch.optim.AdamW(head.module.parameters(), lr=args.learning_rate)
    nature = TwoSiteNature(seed=args.seed + 100, mix=args.nature_mix)
    config = {
        "schema": SCHEMA,
        "initial_model_sha256": file_sha256(args.initial_model),
        "source_fixtures_sha256": file_sha256(args.source_fixtures),
        "emulator_revision": args.emulator_revision,
        "emulator_dll_sha256": file_sha256(args.emulator_dll),
        "emulator_bindings_sha256": file_sha256(
            args.emulator_bindings / "sts2_emulator" / "__init__.py"
        ),
        "arm": args.training_arm, "seed": args.seed,
        "episodes_per_round": args.episodes_per_round,
        "workers": args.workers, "nature_update": args.nature_update,
        "nature_mix": args.nature_mix,
        "temperature": args.temperature, "learning_rate": args.learning_rate,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "sample_limit": args.sample_limit, "gamma": args.gamma,
        "gae_lambda": args.gae_lambda, "scope": "silent-basic-v1",
        "reward": "production-v15-within-combat-hp-preservation",
        "source_law": "UNVERIFIED",
        "evidence_level": "research-training-not-feature-judgment",
    }
    fingerprint = canonical_sha256(config)
    out = args.output_root
    checkpoint = out / "trainer-state.pt"
    metrics: list[dict] = []
    start_round = 0
    if args.resume:
        stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (stored.get("schema") != SCHEMA
                or stored.get("config_fingerprint") != fingerprint):
            raise ValueError("PPO resume mismatch: changed model, emulator or fixture corpus")
        head.module.load_state_dict(stored["head"])
        optimizer.load_state_dict(stored["optimizer"])
        torch.set_rng_state(stored["torch_rng"])
        nature = TwoSiteNature.restore(stored["nature"])
        metrics = stored["metrics"]
        start_round = len(metrics)
        if start_round > args.rounds:
            raise ValueError("Checkpoint has more rounds than requested")
    else:
        if checkpoint.exists():
            p.error("Existing trainer checkpoint; choose --resume or a new output root")
    out.mkdir(parents=True, exist_ok=True)
    atomic_write(out / "experiment.json", json.dumps(
        {"config": config, "fingerprint": fingerprint}, indent=2, sort_keys=True
    ) + "\n")

    command = ("dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl")
    for i in range(start_round, args.rounds):
        model = PhaseSplitNeuralModel(
            base.strategy, head.export(base.combat),
            f"p0-real-player-behavior-round-{i}-{fingerprint[:12]}",
            combat_value_objective="hp_preservation",
        )
        model_path = out / f"behavior-round-{i:05d}.json"
        model.save(model_path)
        spec = FrozenPlayerSpec.from_file(
            model_path, seed=args.seed + i * 100003,
            temperature=args.temperature, greedy=False,
            action_scope="silent-basic-v1",
        )
        chosen_fixtures = tuple(source[
            i * args.episodes_per_round:(i + 1) * args.episodes_per_round
        ])
        start = time.perf_counter()
        report = paired_batch(
            chosen_fixtures, emulator_command=command,
            emulator_bindings=args.emulator_bindings,
            emulator_revision=args.emulator_revision,
            nature=nature, player_spec=spec,
            workers=args.workers, capture_player_ppo=True,
        )
        rollout_seconds = time.perf_counter() - start
        start = time.perf_counter()
        samples, scores = prepare_real_combat_ppo_samples(
            report["records"], arm=args.training_arm,
            base=base.combat, head=head,
            behavior_model_id=model.model_id,
            behavior_sha256=spec.model_sha256,
            temperature=args.temperature, action_scope=spec.action_scope,
            gamma=args.gamma, gae_lambda=args.gae_lambda,
        )
        preparation_seconds = time.perf_counter() - start
        ppo_start = time.perf_counter()
        optimization = ppo_update_pooled(
            head, samples, optimizer,
            temperature=args.temperature,
            epochs=args.epochs,
            batch_size=args.batch_size, sample_limit=args.sample_limit,
            seed=args.seed + 1000003 * i, diagnostics=True,
        )
        ppo_seconds = time.perf_counter() - ppo_start
        nature_start = time.perf_counter()
        nature_update = None
        if args.nature_update:
            rows = tuple(PilotEpisode(
                score,
                tuple(PilotChoice(**raw) for raw in row["nature"]["nature_choices"]),
            ) for row, score in zip(report["records"], scores, strict=True))
            nature_update = nature.update(rows)
        nature_update_seconds = time.perf_counter() - nature_start
        round_record = {
            "round": i, "source_run_count": len(chosen_fixtures),
            "sample_count": len(samples), "completed": len(scores),
            "player_ppo_steps": optimization.steps,
            "mean_combat_score": sum(scores) / len(scores),
            "mean_ppo_loss": optimization.loss,
            "native_comparator_mean_delta": report["mean_paired_return_delta"],
            "nature_update": nature_update,
            "source_digest": report["records_digest"],
            "behavior_sha256": spec.model_sha256,
            "rollout_seconds": rollout_seconds,
            "sample_preparation_seconds": preparation_seconds,
            "player_ppo_seconds": ppo_seconds,
            "nature_update_seconds": nature_update_seconds,
            "optimizer_seconds": time.perf_counter() - start,
            "worker_processes_observed": report["worker_processes_observed"],
            "training_arm": args.training_arm,
        }
        metrics.append(round_record)
        # Atomic continuation with optimizer state. The portable model is
        # separately exported; metadata and source groups are fingerprinted.
        _save_torch_atomic(checkpoint, {
            "schema": SCHEMA, "config_fingerprint": fingerprint,
            "head": head.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "nature": nature.dump(), "metrics": metrics,
        }, torch)
        atomic_write(out / "history.json", json.dumps(
            {"schema": SCHEMA, "config": config, "metrics": metrics,
             "nature": nature.dump(), "decision_grade": False},
            indent=2, sort_keys=True,
        ) + "\n")
        print(
            f"[nature-player-ppo] round={i+1}/{args.rounds} "
            f"arm={args.training_arm} episodes={len(scores)} "
            f"actions={len(samples)} ppo_steps={optimization.steps} "
            f"score={round_record['mean_combat_score']:.4f} "
            f"loss={optimization.loss} native_pair_delta="
            f"{report['mean_paired_return_delta']:+.4f}",
            flush=True,
        )
    final_model = PhaseSplitNeuralModel(
        base.strategy, head.export(base.combat),
        f"p0-player-trained-{args.training_arm}-{fingerprint[:12]}",
        combat_value_objective="hp_preservation",
    )
    final_model.save(out / "final-model.json")
    print(
        f"[nature-player-ppo] complete={len(metrics)} "
        f"final_model={out / 'final-model.json'} "
        "evidence=training-infrastructure-not-feature-judgment",
        flush=True,
    )


if __name__ == "__main__":
    main()
