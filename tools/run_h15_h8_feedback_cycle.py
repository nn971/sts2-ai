#!/usr/bin/env python3
"""One H15 ReferenceNature PUCT -> distillation -> renewed PUCT cycle.

This is a *fixed H8 fixture* under the emulator's declared Chance Laws.
It measures a feedback-loop integration, not native-game performance.
All stochastic outcomes and persistent cursors remain emulator-owned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from sts2_ai.emulator.h15_h8_reference import H8_DECLARED_LAW_MODEL
from sts2_ai.training.h15_search_supervision import decode_target

SCHEMA = "sts2-h15-reference-nature-feedback-cycle-v1"
REGIME = "fixed-h8-fixture-declared-chance-law"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_targets(path: Path, *, checkpoint_sha256: str, simulations: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: target must be an object")
            sample = decode_target(row)
            if (sample.search_model_sha256 != checkpoint_sha256
                    or row.get("chance_prior_id") != H8_DECLARED_LAW_MODEL
                    or row.get("experiment_regime") != REGIME
                    or row.get("search_rollout_mode") != "value"
                    or row.get("search_simulations") != simulations):
                raise ValueError(f"{path}:{line_number}: mismatched search provenance")
            rows.append(row)
    if not rows:
        raise ValueError(f"No real combat search targets saved in {path}")
    return rows


def compare_first_root(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compare the same public starting decision, even when later play differs."""
    for field in ("public_state", "legal_actions", "chance_prior_id",
                  "source_seed", "search_simulations"):
        if before[field] != after[field]:
            raise ValueError(f"Before/after search used different initial root: {field}")
    v0, v1 = before["visits"], after["visits"]
    total0, total1 = sum(v0), sum(v1)
    return {
        "root_action_ids": [a["action_id"] for a in before["legal_actions"]],
        "before_visits": v0,
        "after_visits": v1,
        "before_policy": [v / total0 for v in v0],
        "after_policy": [v / total1 for v in v1],
        "visit_policy_l1_change": sum(
            abs(a / total0 - b / total1) for a, b in zip(v0, v1)
        ),
    }


def verify_parameter_update(before: Path, after: Path) -> dict[str, Any]:
    """Assert an actual floating-point model weight changed, not just metadata."""
    import torch

    old = torch.load(before, weights_only=True, map_location="cpu")["state_dict"]
    new = torch.load(after, weights_only=True, map_location="cpu")["state_dict"]
    if old.keys() != new.keys():
        raise ValueError("Distilled model's state_dict keys changed unexpectedly")
    changed, squared_change = 0, 0.0
    for name in old:
        x, y = old[name], new[name]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"Distilled parameter shape/dtype changed: {name}")
        if not x.is_floating_point():
            if not torch.equal(x, y):
                raise ValueError(f"Non-floating model buffer changed: {name}")
            continue
        if not torch.equal(x, y):
            changed += 1
            delta = y.to(dtype=torch.float64) - x.to(dtype=torch.float64)
            squared_change += float(torch.sum(delta * delta))
    if changed == 0 or not math.isfinite(squared_change) or squared_change <= 0:
        raise RuntimeError("Distillation produced no finite model-parameter update")
    return {"changed_tensors": changed, "model_delta_l2": math.sqrt(squared_change)}



def verify_distillation_receipt(
    student: Path, *, initial_sha: str, source_sha: str,
    expected_steps: int,
) -> dict[str, Any]:
    """Ensure the optimizer receipt refers to exactly these search visits."""
    import torch

    checkpoint = torch.load(student, weights_only=True, map_location="cpu")
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, dict) or (
        provenance.get("base_checkpoint_sha256") != initial_sha
        or provenance.get("search_data_sha256") != source_sha
        or provenance.get("chance_prior") != H8_DECLARED_LAW_MODEL
        or provenance.get("optimizer_steps") != expected_steps
    ):
        raise ValueError("H15 student checkpoint has inconsistent optimizer provenance")
    values: dict[str, Any] = {}
    for key in ("in_sample_fit_before", "in_sample_fit_after"):
        row = provenance.get(key)
        if not isinstance(row, dict) or set(row) != {"policy_loss", "value_loss"}:
            raise ValueError("H15 student checkpoint lacks pre/post fit metrics")
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in row.values()):
            raise ValueError("Nonfinite H15 student fit metric")
        values[key] = row
    return {"optimizer_steps": expected_steps, **values}


def run_stage(name: str, command: list[str], *, root: Path) -> None:
    print(f"[h15-cycle-{name}] {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=root, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emulator-root", required=True, type=Path)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--emulator-dll", required=True, type=Path)
    p.add_argument("--initialize-from-model", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--scenario", default="mawler-a10")
    p.add_argument("--deck", default="silent-starter-v1")
    p.add_argument("--fixture-seed", default="h15-m3-fixed-fixture-v1")
    p.add_argument("--source-nature-seed", type=int, default=113)
    p.add_argument("--search-seed", type=int, default=20261029)
    p.add_argument("--simulations", type=int, default=16)
    p.add_argument("--max-searches", type=int, default=1)
    p.add_argument("--max-candidates", type=int, default=4096)
    p.add_argument("--max-decisions", type=int, default=512)
    p.add_argument("--tree-depth", type=int, default=8)
    p.add_argument("--rollout-depth", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()

    if (len(args.emulator_revision) != 40 or
            any(c not in "0123456789abcdef" for c in args.emulator_revision.lower())
            or min(args.simulations, args.max_searches, args.max_candidates,
                   args.max_decisions, args.tree_depth, args.rollout_depth,
                   args.epochs, args.batch_size) <= 0
            or min(args.source_nature_seed, args.search_seed) < 0
            or args.source_nature_seed >= (1 << 63)
            or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.value_weight) or args.value_weight < 0):
        p.error("Invalid H15 reference-cycle provenance or budget")
    if not args.emulator_dll.is_file() or not args.initialize_from_model.is_file():
        p.error("The pinned emulator DLL and initial checkpoint must exist")
    if not (args.emulator_root / "bindings/python").is_dir():
        p.error("Missing emulator Python bindings")

    root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    start_path = args.initialize_from_model.resolve()
    if output_root.exists():
        p.error("Output directory already exists; use a new directory for each cycle")
    original_sha = file_sha256(start_path)
    def collection_command(checkpoint: Path, destination: Path) -> list[str]:
        return [
            sys.executable, str(root / "tools/collect_h15_h8_reference.py"),
            "--emulator-root", str(args.emulator_root.resolve()),
            "--emulator-revision", args.emulator_revision,
            "--emulator-dll", str(args.emulator_dll.resolve()),
            "--h9-checkpoint", str(checkpoint),
            "--output-root", str(destination),
            "--scenario", args.scenario, "--deck", args.deck,
            "--fixture-seed", args.fixture_seed,
            "--source-nature-seed", str(args.source_nature_seed),
            "--search-seed", str(args.search_seed),
            "--simulations", str(args.simulations),
            "--max-searches", str(args.max_searches),
            "--max-candidates", str(args.max_candidates),
            "--max-decisions", str(args.max_decisions),
            "--tree-depth", str(args.tree_depth),
            "--rollout-depth", str(args.rollout_depth),
            "--leaf-mode", "value", "--use-declared-chance-laws", "--device", args.device,
            "--execute",
        ]
    before_dir, after_dir = output_root / "before", output_root / "after"
    updated_path = output_root / "student.pt"
    train_command = [
        sys.executable, str(root / "tools/train_h15_search_distill.py"),
        "--initialize-from-model", str(start_path),
        "--search-data", str(before_dir / "targets.jsonl"),
        "--accepted-chance-prior", H8_DECLARED_LAW_MODEL,
        "--output", str(updated_path),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--value-weight", str(args.value_weight),
        "--device", args.device, "--execute",
    ]

    print("[h15-cycle-preflight] " + json.dumps({
        "schema": SCHEMA, "emulator_revision": args.emulator_revision,
        "emulator_dll_sha256": file_sha256(args.emulator_dll),
        "initial_model_sha256": original_sha, "chance_prior": H8_DECLARED_LAW_MODEL,
        "regime": REGIME, "leaf_mode": "value",
        "simulations_per_root": args.simulations,
        "source_nature_seed": args.source_nature_seed,
        "search_seed": args.search_seed, "output_root": str(output_root),
    }, sort_keys=True), flush=True)
    if not args.execute:
        for name, command in [
            ("before", collection_command(start_path, before_dir)),
            ("fit", train_command),
            ("after", collection_command(updated_path, after_dir)),
        ]:
            print(f"[h15-cycle-dry-run:{name}] {shlex.join(command)}", flush=True)
        return

    run_stage("before", collection_command(start_path, before_dir), root=root)
    first = read_targets(before_dir / "targets.jsonl",
                         checkpoint_sha256=original_sha, simulations=args.simulations)
    run_stage("fit", train_command, root=root)
    update = verify_parameter_update(start_path, updated_path)
    fitted_steps = args.epochs * math.ceil(len(first) / args.batch_size)
    fitted = verify_distillation_receipt(
        updated_path, initial_sha=original_sha,
        source_sha=file_sha256(before_dir / "targets.jsonl"),
        expected_steps=fitted_steps,
    )
    updated_sha = file_sha256(updated_path)
    run_stage("after", collection_command(updated_path, after_dir), root=root)
    second = read_targets(after_dir / "targets.jsonl",
                          checkpoint_sha256=updated_sha, simulations=args.simulations)
    comparison = compare_first_root(first[0], second[0])
    result = {
        "schema": SCHEMA, "status": "integration_smoke",
        "emulator_revision": args.emulator_revision,
        "emulator_dll_sha256": file_sha256(args.emulator_dll),
        "chance_prior": H8_DECLARED_LAW_MODEL,
        "probability_claim": "declared-emulator-law-only",
        "training_source": REGIME, "leaf_mode": "value",
        "initial_teacher_sha256": original_sha,
        "updated_teacher_sha256": updated_sha,
        "before_targets": len(first), "after_targets": len(second),
        "before_targets_sha256": file_sha256(before_dir / "targets.jsonl"),
        "after_targets_sha256": file_sha256(after_dir / "targets.jsonl"),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "training_fit": fitted,
        "parameter_update": update, "first_root": comparison,
        "before_outcome": json.loads((before_dir / "report.json").read_text())["outcome"],
        "after_outcome": json.loads((after_dir / "report.json").read_text())["outcome"],
    }
    summary = output_root / "feedback-cycle.json"
    temporary = summary.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary)
    print("[h15-cycle-result] " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
