#!/usr/bin/env python3
"""H10 native-state on-policy PPO for the existing H9 shared Transformer.

Mix exact-replayed native A10 combat forks with a small H9 rich synthetic PPO
cohort, under one frozen actor revision and one optimizer update per round.
Fresh source-run-disjoint native holdouts are required; this CLI never trains
on heldout entries or silently restores an incompatible optimizer.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from sts2_ai.evaluation.combat_snapshots import load_recipes
from sts2_ai.models.h7_public_policy import H7PlayerSpec
from sts2_ai.models.shared_transformer import SharedObjectTransformer
from sts2_ai.nature.corpus import file_sha256
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.research_combat_fixtures import rich_source_bank
from sts2_ai.training.h7_combat_ppo import H7CombatPPO
from sts2_ai.training.h10_native_transformer import (
    NATIVE_REWARD, native_cohort, select_native, summarize_native,
    verify_corpora,
)
from train_h7_reference_combat import (
    atomic_torch, collect, summarize as summarize_synthetic,
)

RUN_SCHEMA = "sts2-h10-native-h9-mixed-onpolicy-v1"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix="." + path.name + "-", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def synthetic_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        emulator_dll=args.emulator_dll,
        emulator_bindings=args.emulator_bindings,
        emulator_revision=args.emulator_revision,
        curriculum="h9-rich",
        structured_curriculum_seed=args.structured_curriculum_seed,
        rollout_backend="h8-native",
        actor_backend="shared",
        actor_device=args.actor_device,
        actor_max_batch=16, actor_delay_ms=3.0,
        actor_collation="packed",
        local_reference_nature=True,
        train_terminal_replay_every=16,
        max_boundaries=1200,
        workers=args.workers,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-corpus", type=Path, required=True)
    p.add_argument("--holdout-corpus", type=Path, required=True)
    p.add_argument("--initialize-from-model", type=Path, required=True)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--emulator-bindings", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--native-per-round", type=int, default=64)
    p.add_argument("--synthetic-per-round", type=int, default=32)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--max-native-decisions", type=int, default=512)
    p.add_argument("--study-seed", type=int, default=74027)
    p.add_argument("--actor-seed", type=int, default=31337)
    p.add_argument("--reference-seed", type=int, default=23387)
    p.add_argument("--structured-curriculum-seed", type=int, default=20261028)
    p.add_argument("--actor-device", default="cuda")
    p.add_argument("--device", default="cuda")
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--lr", type=float, default=0.00003)
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--ppo-minibatch", type=int, default=64)
    p.add_argument("--ppo-sample-limit", type=int, default=4096)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()

    if (min(args.rounds, args.native_per_round, args.workers,
            args.max_native_decisions, args.ppo_epochs, args.ppo_minibatch,
            args.checkpoint_every) < 1 or args.synthetic_per_round < 0
            or args.ppo_sample_limit < 2 or args.study_seed < 0
            or args.structured_curriculum_seed < 0
            or not 0 < args.temperature <= 5):
        p.error("Invalid H10 native PPO configuration")
    train = load_recipes(args.train_corpus)
    heldout = load_recipes(args.holdout_corpus)
    verify_corpora(train, heldout, emulator_revision=args.emulator_revision)
    config = {
        "schema": RUN_SCHEMA,
        "native_reward": NATIVE_REWARD,
        "synthetic_reward": "encounter_nonminion_total_hp_v1",
        "train_corpus_sha256": file_sha256(args.train_corpus),
        "holdout_corpus_sha256": file_sha256(args.holdout_corpus),
        "initial_model_sha256": file_sha256(args.initialize_from_model),
        "emulator_revision": args.emulator_revision,
        "emulator_dll_sha256": file_sha256(args.emulator_dll),
        "emulator_bindings_sha256": file_sha256(
            args.emulator_bindings / "sts2_emulator" / "__init__.py",
        ),
        "counts": {"train": len(train), "heldout": len(heldout)},
        "args": {
            k: v for k, v in vars(args).items() if k not in (
                "execute", "resume", "train_corpus", "holdout_corpus",
                "initialize_from_model", "emulator_root", "emulator_dll",
                "emulator_bindings", "output_root",
            )
        },
    }
    fingerprint = canonical_sha256(config)
    print(json.dumps({
        "fingerprint": fingerprint,
        "train_entries": len(train),
        "heldout_entries": len(heldout),
        "planned_native": args.native_per_round * args.rounds,
        "planned_synthetic": args.synthetic_per_round * args.rounds,
        "execute": args.execute,
    }, indent=2), flush=True)
    if not args.execute:
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(args.study_seed)
    learner = H7CombatPPO(
        SharedObjectTransformer.load_checkpoint(args.initialize_from_model),
        device=args.device, lr=args.lr, temperature=args.temperature,
        epochs=args.ppo_epochs, minibatch=args.ppo_minibatch,
        sample_limit=args.ppo_sample_limit,
    )
    out = args.output_root
    model_dir = out / "models"
    save_path = out / "optimizer-checkpoint.pt"
    progress_path = out / "progress.json"
    history: list[dict[str, object]] = []
    if args.resume:
        if not save_path.exists():
            p.error("--resume needs an existing optimizer-checkpoint.pt")
        saved = torch.load(save_path, map_location="cpu", weights_only=True)
        if saved["schema"] != RUN_SCHEMA or saved["fingerprint"] != fingerprint:
            p.error("Native PPO optimizer/config fingerprint mismatch")
        learner.model.load_state_dict(saved["model"], strict=True)
        learner.optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        if len(history) > args.rounds:
            p.error("Existing PPO checkpoint exceeds configured rounds")
        torch.set_rng_state(saved["torch_rng"])
        if args.device.startswith("cuda") and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    elif save_path.exists() or (model_dir.is_dir() and any(model_dir.iterdir())):
        p.error("Output already contains training; pass --resume or use new path")
    model_dir.mkdir(parents=True, exist_ok=True)
    synth_args = synthetic_args(args)
    actor_path = out / "actor-current.pt"

    for round_index in range(len(history), args.rounds):
        begun = time.monotonic()
        learner.model.save_checkpoint(actor_path)
        spec = H7PlayerSpec.from_file(
            actor_path, seed=args.actor_seed,
            temperature=args.temperature, greedy=False,
        )
        selected = select_native(
            train, count=args.native_per_round,
            study_seed=args.study_seed, round_index=round_index,
        )
        native = native_cohort(
            selected=selected, round_index=round_index,
            spec=spec, emulator_root=args.emulator_root,
            emulator_revision=args.emulator_revision,
            workers=args.workers, max_decisions=args.max_native_decisions,
        )
        if any(x.outcome == "censored" for x in native):
            raise RuntimeError(
                "Censored native PPO fights; increase --max-native-decisions "
                "before starting a new fingerprinted run"
            )
        prepared = [step for ep in native for step in ep.samples]
        synthetic_row = None
        if args.synthetic_per_round:
            seeds = rich_source_bank(
                "train", args.synthetic_per_round,
                study_seed=args.study_seed + 1000001,
                round_index=round_index,
            )
            synth_batch = collect(
                seeds, spec=spec, args=synth_args,
                reference_seed=args.reference_seed + 1000003 * round_index,
                capture=True,
            )
            for ep, capture in zip(
                synth_batch["episodes"], synth_batch["player_rollouts"],
                strict=True,
            ):
                prepared.extend(learner._prepare_episode(
                    ep, capture,
                    behavior_model_id=f"h7-{spec.model_sha256[:16]}",
                    temperature=args.temperature,
                    gamma=learner.gamma,
                    gae_lambda=learner.gae_lambda,
                    allow_sampled_terminal_replay=True,
                ))
            synthetic_row = {
                "source_digest": canonical_sha256(seeds),
                "summary": {
                    key: value for key, value in summarize_synthetic(
                        synth_batch["episodes"], curriculum="h9-rich",
                        curriculum_seed=args.structured_curriculum_seed,
                    ).items() if key != "by_cell"
                },
                "terminal_replay_verified": synth_batch["terminal_replay_verified"],
            }
        diagnostics = learner.update_prepared(
            prepared, seed=args.study_seed * 1000003 + round_index,
        )
        row = {
            "round": round_index + 1,
            "behavior_sha256": spec.model_sha256,
            "native_recipe_digest": canonical_sha256([
                [recipe.seed, recipe.exact_hash] for recipe in selected
            ]),
            "native": summarize_native(native),
            "synthetic": synthetic_row,
            "elapsed_seconds": time.monotonic() - begun,
            **diagnostics,
        }
        history.append(row)
        atomic_torch(save_path, {
            "schema": RUN_SCHEMA, "fingerprint": fingerprint,
            "model": learner.model.state_dict(),
            "optimizer": learner.optimizer.state_dict(),
            "history": history,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all()
                if args.device.startswith("cuda") else None
            ),
        })
        if (round_index + 1) % args.checkpoint_every == 0 or round_index + 1 == args.rounds:
            learner.model.save_checkpoint(
                model_dir / f"player-r{round_index+1:04d}.pt"
            )
        atomic_json(progress_path, {
            "schema": RUN_SCHEMA, "fingerprint": fingerprint,
            "config": config, "history": history,
        })
        print(
            f"[h10-native] round={round_index+1}/{args.rounds} "
            f"native={len(native)} native_wins={row['native']['outcomes'].get('victory', 0)} "
            f"synthetic={args.synthetic_per_round} "
            f"updates={diagnostics['updates']} "
            f"decisions={diagnostics['source_decisions']} "
            f"elapsed_s={row['elapsed_seconds']:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
