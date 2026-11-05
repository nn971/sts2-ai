#!/usr/bin/env python3
"""Evaluate an H22 checkpoint once on an untouched, independent Act-1 seed cohort.

Read-only relative to the training run. Keep this audit for final validation,
not for selecting training hyperparameters or checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sts2_ai.training.h16_joint_act_pilot import H22_SCHEMA

try:
    from tools.train_h16_joint_act_pilot import collect, seed_namespace, sha256, summarize, write_json
except ModuleNotFoundError:
    from train_h16_joint_act_pilot import collect, seed_namespace, sha256, summarize, write_json


def audit_checkpoint(
    progress_path: Path, *, emulator_root: Path, audit_count: int = 32,
    audit_tag: str = "independent-final-v1", round_number: int | None = None,
    workers: int | None = None, output: Path | None = None,
    macro_model: Path | None = None, execute: bool = False,
) -> dict[str, Any]:
    if type(audit_count) is not int or audit_count < 1:
        raise ValueError("Independent audit count must be positive")
    if not audit_tag or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in audit_tag):
        raise ValueError("Audit tag must use lowercase letters, digits, '-' or '_'")
    progress = json.loads(progress_path.read_text())
    source = progress["experiment"]
    if source["schema"] != H22_SCHEMA:
        raise ValueError("Independent audit requires an H22/H21 training run")
    opts = source["options"]
    if (opts.get("representation") != "h21"
            or opts.get("actor_inference") != "shared"):
        raise ValueError("Audit requires the saved H21 shared-actor configuration")
    if (opts.get("actor_device") != "cuda"
            or opts.get("actor_h21_collator") != "packed"
            or opts.get("actor_deterministic_cuda") is not True
            or opts.get("actor_fixed_shape_batch_size") != 4):
        raise ValueError("Audit requires the packed deterministic CUDA run")
    if macro_model is not None and sha256(macro_model) != source.get("macro_sha256"):
        raise ValueError("Independent audit macro reference checksum mismatch")
    if opts.get("macro_prior_weight", 0) > 0 and macro_model is None:
        raise ValueError("Audit with nonzero macro prior requires its pinned model")
    if opts.get("macro_prior_weight", 0) == 0 and macro_model is not None:
        raise ValueError("Audit without macro prior must omit macro model")

    rows = progress["rounds"]
    if not rows or any(row["round"] != i for i, row in enumerate(rows, 1)):
        raise ValueError("No contiguous committed training rounds to audit")
    last = rows[-1]["round"]
    selected = last if round_number is None else round_number
    if type(selected) is not int or not 1 <= selected <= last:
        raise ValueError("Audit round must be an existing committed round")
    index = selected - 1
    row = rows[index]
    checkpoint = progress_path.parent / "models" / f"joint-r{selected:04d}.pt"
    if (not checkpoint.is_file()
            or sha256(checkpoint) != row["checkpoint_sha256"]):
        raise ValueError("Audit checkpoint checksum or file mismatch")
    if workers is None:
        workers = opts["workers"]
    if type(workers) is not int or workers <= 0:
        raise ValueError("Independent audit worker count must be positive")
    out = output or progress_path.parent / f"independent-audit-r{selected:04d}.json"
    if out.exists():
        raise ValueError("Independent audit output exists; choose a distinct tag/output")
    namespace = seed_namespace(opts["ascension"], opts["map_profile"])
    study_seed = opts["study_seed"]
    seeds = [f"{namespace}-audit-{study_seed}-{audit_tag}-{i}" for i in range(audit_count)]
    reserved = {
        record["seed"]
        for stage in [*(
            [r["train"], r["heldout"]]
            for r in rows
        )]
        for group in stage if group is not None
        for record in group.get("floor_progress", {}).get("runs", [])
    }
    baseline_path = progress_path.parent / "baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        reserved.update(
            record["seed"] for record in
            baseline.get("floor_progress", {}).get("runs", [])
        )
    if len(set(seeds)) != len(seeds) or not set(seeds).isdisjoint(reserved):
        raise ValueError("Independent audit seed overlaps train or heldout")
    evaluation_horizon = opts.get("evaluation_horizon_lambda")
    if evaluation_horizon is None:
        evaluation_horizon = row["horizon_lambda"]
    if not math.isfinite(evaluation_horizon) or not 0 <= evaluation_horizon < 1:
        raise ValueError("Invalid independent audit horizon")

    identity = {
        "schema": "sts2-h22-independent-checkpoint-audit-v1",
        "training_fingerprint": progress["fingerprint"],
        "emulator_revision": source["emulator_revision"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": row["checkpoint_sha256"],
        "round": selected,
        "evaluation_horizon_lambda": evaluation_horizon,
        "search": False,
        "audit_tag": audit_tag,
        "audit_count": audit_count,
        "workers": workers,
        "seeds": seeds,
    }
    if not execute:
        return {"execute": False, **identity}
    shared = SimpleNamespace(
        **{
            **opts,
            "ai_root": Path.cwd(),
            "emulator_root": emulator_root,
            "emulator_revision": source["emulator_revision"],
            "macro_model": macro_model,
            "workers": workers,
        },
    )
    episodes = collect(
        shared, checkpoint, seeds=seeds,
        horizon_lambda=evaluation_horizon, search=False,
    )
    observed = summarize(episodes)
    if (observed["played"] != audit_count
            or observed["censored"] != 0
            or observed["completed"] != audit_count
            or observed["searches"] != 0):
        raise ValueError("Incomplete, censored or searched independent audit")
    document = {**identity, "summary": observed}
    write_json(out, document)
    return {
        "result": "independent audit completed",
        "round": selected,
        "completed": observed["completed"],
        "cleared": observed["cleared"],
        "mean_max_floor": observed["floor_progress"]["mean_max_floor_reached"],
        "output": str(out),
        "checkpoint_sha256": row["checkpoint_sha256"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--progress", type=Path, required=True)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--audit-count", type=int, default=32)
    p.add_argument("--audit-tag", default="independent-final-v1")
    p.add_argument("--round", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--macro-model", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    try:
        result = audit_checkpoint(
            args.progress, emulator_root=args.emulator_root,
            audit_count=args.audit_count, audit_tag=args.audit_tag,
            round_number=args.round, workers=args.workers,
            output=args.output, macro_model=args.macro_model, execute=args.execute,
        )
    except (ValueError, FileNotFoundError, KeyError) as error:
        p.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
