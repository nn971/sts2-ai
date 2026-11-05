#!/usr/bin/env python3
"""Materialize a frozen, H21-untrained H22 benchmark checkpoint from H16.

The source H16 model supplies shape-compatible weights only; H21 entity
embeddings and relation layers are newly initialized. No optimization occurs.
The output is for architecture/actor benchmarking, not a trained H22 policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from sts2_ai.training.h16_joint_act_pilot import H22_SCHEMA, JointActPilotNet


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_checkpoint(source: Path, output: Path, *, seed: int = 20261105) -> dict:
    if not 0 <= seed < 2**63:
        raise ValueError("Seed must be a nonnegative 63-bit integer")
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing H16 source checkpoint: {source}. "
            "Select an existing H16 joint model from results/**/models/*.pt"
        )
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite frozen checkpoint: {output}"
        )
    base = JointActPilotNet.load(source, device="cpu")
    if base.representation != "h16":
        raise ValueError("Expected H16 source checkpoint for H22 warm-start")
    torch.manual_seed(seed)
    candidate = JointActPilotNet.warmstart_h21(base).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Staging checkpoint already exists: {staging}")
    try:
        candidate.save(staging)
        restored = JointActPilotNet.load(staging, device="cpu")
        if restored.representation != "h21":
            raise ValueError("H22 checkpoint reload did not preserve H21 encoding")
        if output.exists():
            raise FileExistsError(f"Output was created concurrently: {output}")
        staging.rename(output)
    finally:
        staging.unlink(missing_ok=True)
    return {
        "schema": "sts2-p2-h22-frozen-warmstart-provenance-v1",
        "checkpoint_schema": H22_SCHEMA,
        "representation": "h21",
        "source_h16": str(source.resolve()),
        "source_sha256": sha256(source),
        "output_h22": str(output.resolve()),
        "output_sha256": sha256(output),
        "seed": seed,
        "trained_h21_steps": 0,
        "note": "H16 shape-compatible warm-start only; H21 tokens/edges untrained",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initial-h16", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261105)
    a = p.parse_args()
    try:
        report = create_checkpoint(a.initial_h16, a.output, seed=a.seed)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        p.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
