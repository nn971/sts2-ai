"""Portable small two-head neural network: Torch training, dependency-free inference.

State and action vectors use the existing *fair-observation* feature tokenizer.
Exported JSON weights are ordinary floats so deploying a trained model to MCTS
does not require the PyTorch runtime. Weights are interpreted with a strict v1
format and checked for shape and finiteness before inference.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_ai.emulator import LegalAction, Observation

from .hashed_linear import policy_features, state_dict, state_features
from .protocol import PolicyValueEstimate

NEURAL_FORMAT = "sts2-neural-policy-value-v1"


def _vector(value: object, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError("Neural vector has incorrect shape")
    result = [float(x) for x in value]
    if not all(math.isfinite(x) for x in result):
        raise ValueError("Neural weights must be finite")
    return result


def _matrix(value: object, rows: int, columns: int) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError("Neural matrix has incorrect shape")
    return [_vector(row, columns) for row in value]


def _sparse_matvec(
    matrix: list[list[float]],
    features: dict[int, float],
    bias: list[float],
) -> list[float]:
    return [
        b + sum(row[index] * val for index, val in features.items())
        for row, b in zip(matrix, bias, strict=True)
    ]


def _relu(vector: list[float]) -> list[float]:
    return [max(0.0, v) for v in vector]


@dataclass(slots=True)
class NeuralPolicyValueModel:
    dimension: int
    hidden: int
    state_weight: list[list[float]]
    state_bias: list[float]
    action_weight: list[list[float]]
    action_bias: list[float]
    policy_weight: list[float]
    policy_bias: float
    value_weight: list[float]
    value_bias: float
    model_id: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NeuralPolicyValueModel:
        if raw.get("format") != NEURAL_FORMAT:
            raise ValueError("Unsupported neural model format")
        dimension, hidden = raw.get("dimension"), raw.get("hidden")
        if (
            type(dimension) is not int or type(hidden) is not int
            or dimension <= 0 or hidden <= 0
        ):
            raise ValueError("Invalid neural architecture")
        model_id = raw.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("Invalid neural model identifier")
        policy_bias = float(raw["policy_bias"])
        value_bias = float(raw["value_bias"])
        if not math.isfinite(policy_bias) or not math.isfinite(value_bias):
            raise ValueError("Nonfinite neural bias")
        return cls(
            dimension=dimension,
            hidden=hidden,
            state_weight=_matrix(raw["state_weight"], hidden, dimension),
            state_bias=_vector(raw["state_bias"], hidden),
            action_weight=_matrix(raw["action_weight"], hidden, dimension),
            action_bias=_vector(raw["action_bias"], hidden),
            policy_weight=_vector(raw["policy_weight"], hidden),
            policy_bias=policy_bias,
            value_weight=_vector(raw["value_weight"], hidden),
            value_bias=value_bias,
            model_id=model_id,
        )

    @classmethod
    def load(cls, path: Path) -> NeuralPolicyValueModel:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Neural weight file is not an object")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": NEURAL_FORMAT,
            "dimension": self.dimension,
            "hidden": self.hidden,
            "state_weight": self.state_weight,
            "state_bias": self.state_bias,
            "action_weight": self.action_weight,
            "action_bias": self.action_bias,
            "policy_weight": self.policy_weight,
            "policy_bias": self.policy_bias,
            "value_weight": self.value_weight,
            "value_bias": self.value_bias,
            "model_id": self.model_id,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def evaluate(
        self,
        observation: Observation,
        legal_actions: Sequence[LegalAction],
    ) -> PolicyValueEstimate:
        state = state_dict(observation.payload_json)
        features = state_features(state, self.dimension)
        hidden = _relu(_sparse_matvec(self.state_weight, features, self.state_bias))
        value_raw = self.value_bias + sum(
            a * b for a, b in zip(self.value_weight, hidden, strict=True)
        )
        logits = []
        for action in legal_actions:
            action_vector = policy_features(
                state, action.kind, action.payload_json, self.dimension
            )
            projected = _sparse_matvec(
                self.action_weight, action_vector, self.action_bias
            )
            logits.append(
                self.policy_bias
                + sum(
                    weight * max(0.0, h + projected[index])
                    for index, (weight, h) in enumerate(
                        zip(self.policy_weight, hidden, strict=True)
                    )
                )
            )
        return PolicyValueEstimate(action_logits=tuple(logits), value=math.tanh(value_raw))


def load_policy_value_model(path: Path) -> NeuralPolicyValueModel:
    return NeuralPolicyValueModel.load(path)
