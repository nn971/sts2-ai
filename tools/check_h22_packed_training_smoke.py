#!/usr/bin/env python3
"""Check one-round H22 packed shared-actor smoke receipts and model checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from sts2_ai.training.h16_joint_act_pilot import H22_SCHEMA, JointActPilotNet


def verify_smoke(
    root: Path, *, fixed_rows: int = 4, require_cuda: bool = False,
    expected_workers: int | None = None,
    expected_train: int | None = None,
    expected_heldout: int | None = None,
) -> dict:
    progress = json.loads((root / "progress.json").read_text())
    rounds = progress["rounds"]
    if len(rounds) != 1:
        raise ValueError(f"Expected exactly one trained H22 round, got {len(rounds)}")
    opts = progress["experiment"]["options"]
    if (progress["experiment"]["schema"] != H22_SCHEMA
            or opts.get("representation") != "h21"
            or opts.get("actor_inference") != "shared"
            or opts.get("actor_h21_collator") != "packed"
            or opts.get("actor_fixed_shape_batch_size") != fixed_rows
            or opts.get("actor_max_batch", 0) < fixed_rows):
        raise ValueError("H22 packed fixed-shape training options mismatch")
    if require_cuda and opts.get("actor_deterministic_cuda") is not True:
        raise ValueError("Missing deterministic CUDA training provenance")
    for label, value in (
        ("workers", expected_workers),
        ("train", expected_train),
        ("heldout", expected_heldout),
    ):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"Expected {label} must be a positive integer")
    for key, expected in (
        ("workers", expected_workers),
        ("runs_per_round", expected_train),
        ("heldout_runs", expected_heldout),
    ):
        if expected is not None and opts.get(key) != expected:
            raise ValueError(f"H22 smoke requested {key} does not match {expected}")
    record = rounds[0]
    if (record.get("round") != 1
            or record["train"]["completed"] < 1
            or record["train"]["censored"]
            or record["heldout"]["censored"]
            or record["train"]["searches"] < 1
            or record["train"]["search_transitions"] < 1
            or record["update"]["optimizer_steps"] < 1):
        raise ValueError("Missing completed train/eval, real PUCT, or optimizer update")
    if (expected_train is not None
            and record["train"]["completed"] != expected_train):
        raise ValueError("H22 smoke train count mismatch")
    if (expected_heldout is not None
            and record["heldout"]["completed"] != expected_heldout):
        raise ValueError("H22 smoke heldout count mismatch")
    for stage in ("train", "heldout"):
        actor = record[stage]["rollout_profile"]["actor_service"]
        if (not isinstance(actor, dict)
                or actor.get("p2_h21_collator") != "packed"
                or actor.get("p2_fixed_shape_batch_size") != fixed_rows
                or actor.get("p2_fixed_shape_bucket_forwards", 0) < 1
                or actor.get("p2_fixed_shape_physical_rows", 0) < actor.get("requests", 0)
                or actor.get("requests", 0) < 1):
            raise ValueError(f"Missing packed fixed-shape {stage} actor receipts")
        if require_cuda and (
            actor.get("device") != "cuda"
            or actor.get("p2_deterministic_cuda") is not True
            or actor.get("p2_cuda_cublas_workspace_config") != ":4096:8"
        ):
            raise ValueError(f"Missing deterministic CUDA {stage} actor receipts")
    checkpoint = root / "models" / "joint-r0001.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if (digest != record["checkpoint_sha256"]
            or not (root / "optimizer-state.pt").is_file()):
        raise ValueError("H22 checkpoint hash or optimizer state is missing")
    trained = JointActPilotNet.load(checkpoint, device="cpu")
    if trained.representation != "h21":
        raise ValueError("H22 trained checkpoint representation mismatch")
    tensors = trained.state_dict().values()
    if not all(torch.isfinite(t).all().item() for t in tensors if t.is_floating_point()):
        raise ValueError("H22 trained checkpoint has non-finite weights")
    report = {
        "result": "H22 one-round packed training smoke passed",
        "round": 1,
        "train_completed": record["train"]["completed"],
        "train_censored": record["train"]["censored"],
        "heldout_completed": record["heldout"]["completed"],
        "heldout_censored": record["heldout"]["censored"],
        "search_roots": record["train"]["searches"],
        "search_transitions": record["train"]["search_transitions"],
        "optimizer_steps": record["update"]["optimizer_steps"],
        "fixed_rows": fixed_rows,
        "actor_device": record["train"]["rollout_profile"]["actor_service"]["device"],
        "requested_workers": opts.get("workers"),
        "checkpoint_sha256": digest,
        "checkpoint": str(checkpoint),
    }
    (root / "smoke-verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fixed-rows", type=int, default=4)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--expected-workers", type=int)
    parser.add_argument("--expected-train", type=int)
    parser.add_argument("--expected-heldout", type=int)
    args = parser.parse_args()
    print(json.dumps(
        verify_smoke(
            args.output_root, fixed_rows=args.fixed_rows,
            require_cuda=args.require_cuda,
            expected_workers=args.expected_workers,
            expected_train=args.expected_train,
            expected_heldout=args.expected_heldout,
        ),
        indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
