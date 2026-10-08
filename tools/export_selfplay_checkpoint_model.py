#!/usr/bin/env python3
"""Export learned policy/value weights from a completed-round self-play checkpoint.

This is **not** checkpoint resumption. It extracts *only* the public neural
parameters and starts no emulator, no optimizer, and no training trajectory.
Use when an emulator revision change invalidates exact resume; new training
should explicitly warm-start from the exported JSON with a new optimizer.
Only load checkpoints you created yourself.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

from sts2_ai.models.neural import NEURAL_FORMAT, NeuralPolicyValueModel
from sts2_ai.training.selfplay_checkpoint import CHECKPOINT_SCHEMA

_PARAMETER_NAMES = (
    "state_weight", "state_bias", "action_weight", "action_bias",
    "policy_weight", "policy_bias", "value_weight", "value_bias",
)


def export_checkpoint(checkpoint: Path, output: Path) -> dict[str, Any]:
    """Validate and export a portable model; leave checkpoint untouched."""
    torch = importlib.import_module("torch")
    # weights_only blocks arbitrary pickle code execution; always read locally
    # trusted checkpoints, since file contents are still untrusted data.
    record = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(record, dict) or record.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Unsupported self-play checkpoint schema")
    config = record.get("config")
    completed_rounds = record.get("completed_rounds")
    params = record.get("params")
    metrics = record.get("metrics")
    if not isinstance(config, dict) or not isinstance(params, dict):
        raise ValueError("Malformed checkpoint configuration or weights")
    if type(completed_rounds) is not int or completed_rounds <= 0:
        raise ValueError("Checkpoint must contain completed training rounds")
    if not isinstance(metrics, list) or len(metrics) != completed_rounds:
        raise ValueError("Incomplete or malformed checkpoint round history")
    if tuple(sorted(params)) != tuple(sorted(_PARAMETER_NAMES)):
        raise ValueError("Unexpected neural parameter keys")
    if not all(isinstance(params[name], torch.Tensor) for name in _PARAMETER_NAMES):
        raise ValueError("Checkpoint parameters must be Torch tensors")

    config_json = json.dumps(config, sort_keys=True)
    config_checksum = hashlib.sha256(config_json.encode()).hexdigest()
    if record.get("config_sha256") != config_checksum:
        raise ValueError("Checkpoint training configuration hash mismatch")
    dimension, hidden = config.get("dimension"), config.get("hidden")
    if type(dimension) is not int or type(hidden) is not int:
        raise ValueError("Checkpoint architecture is invalid")
    source_checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model_id = f"checkpoint-round{completed_rounds}-{source_checksum[:12]}"
    model = NeuralPolicyValueModel.from_dict({
        "format": NEURAL_FORMAT,
        "dimension": dimension,
        "hidden": hidden,
        "model_id": model_id,
        "value_head_trained": any(
            isinstance(round_data, dict) and
            isinstance(round_data.get("update_steps"), int) and
            round_data["update_steps"] > 0
            for round_data in metrics
        ),
        **{
            name: params[name].detach().cpu().tolist()
            for name in _PARAMETER_NAMES
        },
    })
    model.save(output)

    return {
        "input_checkpoint": str(checkpoint),
        "output_model": str(output),
        "completed_rounds": completed_rounds,
        "completed_episodes": sum(
            r.get("completed", 0) for r in metrics if isinstance(r, dict)
        ),
        "training_victories": sum(
            r.get("wins", 0) for r in metrics if isinstance(r, dict)
        ),
        "source_emulator_revision": record.get("emulator_revision"),
        "source_training_version": config.get("training_version"),
        "checkpoint_sha256": source_checksum,
        "model_id": model_id,
        "warning": (
            "Extracted model weights only. The optimizer, checkpoint history "
            "and old emulator state are NOT resumed. Run a new experiment "
            "with --initialize-from-model after updating the emulator."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        f"[checkpoint] validating {args.checkpoint} and exporting learned weights",
        flush=True,
    )
    result = export_checkpoint(args.checkpoint, args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
