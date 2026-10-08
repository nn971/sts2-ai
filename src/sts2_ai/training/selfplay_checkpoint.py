"""Versioned CPU-state checkpoints with atomic round boundaries.

Only locally created checkpoints should be loaded. torch.load(...,
weights_only=True) does not permit arbitrary object deserialization, and the
configuration/engine revision guards reject accidental cross-run resumption.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sts2_ai.models.neural import NeuralPolicyValueModel

if TYPE_CHECKING:
    from sts2_ai.training.selfplay import TrainingRound

CHECKPOINT_SCHEMA = "sts2-public-onpolicy-checkpoint-v1"


def _fingerprint(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def save_checkpoint(
    path: Path,
    *,
    torch: Any,
    params: dict[str, Any],
    optimizer: Any,
    config: dict[str, Any],
    revision: str,
    initial_model: NeuralPolicyValueModel,
    metrics: list[TrainingRound],
) -> None:
    """Atomically persist one *completed* round, optimizer state included."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": CHECKPOINT_SCHEMA,
        "config_sha256": _fingerprint(config),
        "config": config,
        "emulator_revision": revision,
        "completed_rounds": len(metrics),
        "metrics": [asdict(row) for row in metrics],
        "initial_model": initial_model.to_dict(),
        "params": {name: value.detach().cpu().clone() for name, value in params.items()},
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.",
            suffix=".pending", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            torch.save(record, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    *,
    torch: Any,
    params: dict[str, Any],
    optimizer: Any,
    config: dict[str, Any],
    revision: str,
) -> tuple[NeuralPolicyValueModel, list[TrainingRound]]:
    """Restore all training state or fail before taking a gradient step."""
    from sts2_ai.training.selfplay import TrainingRound

    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Incompatible neural training checkpoint schema")
    if (
        data.get("config_sha256") != _fingerprint(config)
        or data.get("config") != config
    ):
        raise ValueError("Checkpoint training configuration mismatch")
    if data.get("emulator_revision") != revision:
        raise ValueError("Checkpoint emulator revision mismatch")
    raw_params = data.get("params")
    if not isinstance(raw_params, dict) or raw_params.keys() != params.keys():
        raise ValueError("Checkpoint parameter layout mismatch")
    with torch.no_grad():
        for name, target in params.items():
            source = raw_params[name]
            if not isinstance(source, torch.Tensor) or source.shape != target.shape:
                raise ValueError(f"Checkpoint parameter shape mismatch: {name}")
            target.copy_(source)
    optimizer.load_state_dict(data["optimizer"])
    torch.set_rng_state(data["torch_rng_state"])
    initial = NeuralPolicyValueModel.from_dict(data["initial_model"])
    raw_metrics = data["metrics"]
    if not isinstance(raw_metrics, list) or any(
        not isinstance(item, dict) for item in raw_metrics
    ):
        raise ValueError("Invalid checkpoint round metrics")
    metrics = [TrainingRound(**item) for item in raw_metrics]
    if data.get("completed_rounds") != len(metrics) or any(
        row.round_index != i for i, row in enumerate(metrics)
    ):
        raise ValueError("Noncontiguous checkpoint round history")
    return initial, metrics
