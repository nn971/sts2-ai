#!/usr/bin/env python3
"""Measure H7 full-model throughput on replay-verified real A10 combat worlds.

No optimizer step, distillation, macro policy, or native RNG fidelity check.
Requires an H5-compatible already-built semantic emulator binary.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from sts2_ai.evaluation.h7_readiness import make_readiness_report
from sts2_ai.models.h7_public_policy import H7PlayerSpec
from sts2_ai.models.shared_transformer import H7Config, SharedObjectTransformer
from sts2_ai.nature.corpus import file_sha256
from sts2_ai.nature.h5_curriculum import source_bank
from sts2_ai.nature.protocol import canonical_sha256

from train_h7_reference_combat import collect


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--emulator-bindings", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--player-checkpoint", type=Path)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--feedforward", type=int, default=1536)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--max-boundaries", type=int, default=1200)
    p.add_argument("--study-seed", type=int, default=73127)
    p.add_argument("--actor-seed", type=int, default=31337)
    p.add_argument("--reference-seed", type=int, default=23387)
    p.add_argument("--actor-backend", choices=("local", "shared"), default="local")
    p.add_argument("--actor-device", default="cpu")
    p.add_argument("--actor-max-batch", type=int, default=16)
    p.add_argument("--actor-delay-ms", type=float, default=3.0)
    args = p.parse_args()
    if (args.workers < 1 or args.episodes < 16 or args.max_boundaries < 1
            or min(args.study_seed, args.actor_seed, args.reference_seed) < 0):
        p.error("Require positive workers, all 16 train cells, and valid seeds")
    if (not args.emulator_dll.is_file()
            or not (args.emulator_bindings / "sts2_emulator" / "__init__.py").is_file()):
        p.error("Missing compatible semantic emulator DLL or Python bindings")
    if args.player_checkpoint is not None and not args.player_checkpoint.is_file():
        p.error("Player checkpoint is missing")

    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="sts2-h7-readiness-") as tmp:
        if args.player_checkpoint:
            path = args.player_checkpoint
            model = SharedObjectTransformer.load_checkpoint(path)
            start = "existing-h7-checkpoint"
        else:
            path = Path(tmp) / "h7-actor.pt"
            torch.manual_seed(args.study_seed)
            model = SharedObjectTransformer(H7Config(
                width=args.width, layers=args.layers, heads=args.heads,
                feedforward=args.feedforward,
            ))
            model.save_checkpoint(path)
            start = "fresh-random-weights"
        parameter_count = sum(param.numel() for param in model.parameters())
        model_config = asdict(model.config)
        del model
        spec = H7PlayerSpec.from_file(
            path, seed=args.actor_seed, temperature=0.85, greedy=False,
        )
        seeds = source_bank("train", args.episodes, study_seed=args.study_seed)
        print(
            f"[h7-readiness] {start} parameters={parameter_count} "
            f"workers={args.workers} combats={len(seeds)}",
            flush=True,
        )
        cohort = collect(
            seeds, spec=spec, args=SimpleNamespace(
                emulator_dll=args.emulator_dll,
                emulator_bindings=args.emulator_bindings,
                emulator_revision=args.emulator_revision,
                workers=args.workers, max_boundaries=args.max_boundaries,
                actor_backend=args.actor_backend, actor_device=args.actor_device,
                actor_max_batch=args.actor_max_batch,
                actor_delay_ms=args.actor_delay_ms, profile_actor=True,
            ),
            reference_seed=args.reference_seed, capture=False,
        )
        report = make_readiness_report(
            cohort, seeds, workers=args.workers,
            model_sha256=spec.model_sha256,
            parameter_count=parameter_count,
        )
        report.update({
            "initialization": start,
            "actor_backend": args.actor_backend,
            "actor_device": args.actor_device,
            "actor_max_batch": args.actor_max_batch,
            "actor_delay_ms": args.actor_delay_ms,
            "actor_metrics": cohort["actor_metrics"],
            "model_config": model_config,
            "source_digest": canonical_sha256(seeds),
            "emulator_revision": args.emulator_revision,
            "emulator_dll": str(args.emulator_dll.resolve()),
            "emulator_dll_sha256": file_sha256(args.emulator_dll),
            "emulator_bindings_sha256": file_sha256(
                args.emulator_bindings / "sts2_emulator" / "__init__.py"
            ),
            "emulator_bindings": str(args.emulator_bindings.resolve()),
            "max_boundaries": args.max_boundaries,
            "study_seed": args.study_seed,
            "actor_seed": args.actor_seed,
            "reference_seed": args.reference_seed,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[h7-readiness] decisions/s={report['player_decisions_per_second']:.2f} "
        f"combats/s={report['combats_per_second']:.3f} "
        f"processes={report['worker_processes_observed']} "
        f"report={args.output}", flush=True,
    )


if __name__ == "__main__":
    main()
