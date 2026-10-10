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

from .hashed_linear import (
    neural_action_features,
    policy_features,
    state_dict,
    state_features,
    tactical_action_features,
    public_resources_tactical_action_features,
)
from .enemy_attention import ATTENTION_FIELDS, contextualize_python
from .enemy_instances import (
    action_target_instance, enemy_instance_vectors,
    instance_action_features, instance_global_features,
)
from .protocol import PolicyValueEstimate
from .tactical_state import (
    damage_tactical_state_features,
    public_resources_tactical_state_features,
    relational_tactical_state_features,
    tactical_state_features,
)

NEURAL_FORMAT = "sts2-neural-policy-value-v2-semantic-action"
TACTICAL_FORMAT = "sts2-neural-policy-value-v3-target-aware-combat"
TACTICAL_STRUCTURED_FORMAT = "sts2-neural-policy-value-v4-structured-tactical"
TACTICAL_RELATIONAL_FORMAT = "sts2-neural-policy-value-v5-relational-tactical"
TACTICAL_DAMAGE_FORMAT = "sts2-neural-policy-value-v6-relational-damage-tactical"
TACTICAL_RESOURCES_FORMAT = "sts2-neural-policy-value-v7-public-resources-tactical"
TACTICAL_INSTANCES_FORMAT = "sts2-neural-policy-value-v8-instance-species-target"
TACTICAL_ATTENTION_FORMAT = "sts2-neural-policy-value-v13-enemy-relational-attention"
LEGACY_NEURAL_FORMAT = "sts2-neural-policy-value-v1"


def _trained_flag(raw: dict[str, Any]) -> bool:
    value = raw.get("value_head_trained", True)
    if type(value) is not bool:
        raise ValueError("Neural value-head training flag must be boolean")
    return value


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
    format_id: str = NEURAL_FORMAT
    value_head_trained: bool = True
    enemy_weight: list[list[float]] | None = None
    enemy_bias: list[float] | None = None
    enemy_context_weight: list[float] | None = None
    enemy_target_weight: list[float] | None = None
    attn_query: list[list[float]] | None = None
    attn_key: list[list[float]] | None = None
    attn_value: list[list[float]] | None = None
    attn_output: list[list[float]] | None = None
    attn_gate: list[float] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NeuralPolicyValueModel:
        if raw.get("format") not in {
            NEURAL_FORMAT, TACTICAL_FORMAT, TACTICAL_STRUCTURED_FORMAT,
            TACTICAL_RELATIONAL_FORMAT, TACTICAL_DAMAGE_FORMAT,
            TACTICAL_RESOURCES_FORMAT, TACTICAL_INSTANCES_FORMAT,
            TACTICAL_ATTENTION_FORMAT,
            LEGACY_NEURAL_FORMAT,
        }:
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
        if raw["format"] == TACTICAL_ATTENTION_FORMAT and hidden % 4:
            raise ValueError("Attention hidden size must be divisible by four")
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
            format_id=str(raw["format"]),
            value_head_trained=_trained_flag(raw),
            enemy_weight=(
                _matrix(raw["enemy_weight"], hidden, dimension)
                if raw["format"] in (TACTICAL_INSTANCES_FORMAT, TACTICAL_ATTENTION_FORMAT) else None
            ),
            enemy_bias=(
                _vector(raw["enemy_bias"], hidden)
                if raw["format"] in (TACTICAL_INSTANCES_FORMAT, TACTICAL_ATTENTION_FORMAT) else None
            ),
            enemy_context_weight=(
                _vector(raw["enemy_context_weight"], hidden)
                if raw["format"] in (TACTICAL_INSTANCES_FORMAT, TACTICAL_ATTENTION_FORMAT) else None
            ),
            enemy_target_weight=(
                _vector(raw["enemy_target_weight"], hidden)
                if raw["format"] in (TACTICAL_INSTANCES_FORMAT, TACTICAL_ATTENTION_FORMAT) else None
            ),
            attn_query=_matrix(raw["attn_query"], hidden, hidden)
            if raw["format"] == TACTICAL_ATTENTION_FORMAT else None,
            attn_key=_matrix(raw["attn_key"], hidden, hidden)
            if raw["format"] == TACTICAL_ATTENTION_FORMAT else None,
            attn_value=_matrix(raw["attn_value"], hidden, hidden)
            if raw["format"] == TACTICAL_ATTENTION_FORMAT else None,
            attn_output=_matrix(raw["attn_output"], hidden, hidden)
            if raw["format"] == TACTICAL_ATTENTION_FORMAT else None,
            attn_gate=_vector(raw["attn_gate"], hidden)
            if raw["format"] == TACTICAL_ATTENTION_FORMAT else None,
            
        )

    @classmethod
    def load(cls, path: Path) -> NeuralPolicyValueModel:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Neural weight file is not an object")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "format": self.format_id,
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
            "value_head_trained": self.value_head_trained,
        }
        if self.format_id in (TACTICAL_INSTANCES_FORMAT, TACTICAL_ATTENTION_FORMAT):
            result.update(
                enemy_weight=self.enemy_weight,
                enemy_bias=self.enemy_bias,
                enemy_context_weight=self.enemy_context_weight,
                enemy_target_weight=self.enemy_target_weight,
            )
        if self.format_id == TACTICAL_ATTENTION_FORMAT:
            result.update({name: getattr(self, name) for name in ATTENTION_FIELDS})
        return result

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
        if self.format_id in (TACTICAL_INSTANCES_FORMAT, TACTICAL_ATTENTION_FORMAT):
            return self._evaluate_instances(state, legal_actions)
        features = (
            public_resources_tactical_state_features(state, self.dimension)
            if self.format_id == TACTICAL_RESOURCES_FORMAT
            else damage_tactical_state_features(state, self.dimension)
            if self.format_id == TACTICAL_DAMAGE_FORMAT
            else relational_tactical_state_features(state, self.dimension)
            if self.format_id == TACTICAL_RELATIONAL_FORMAT
            else tactical_state_features(state, self.dimension)
            if self.format_id == TACTICAL_STRUCTURED_FORMAT
            else state_features(state, self.dimension)
        )
        hidden = _relu(_sparse_matvec(self.state_weight, features, self.state_bias))
        value_raw = self.value_bias + sum(
            a * b for a, b in zip(self.value_weight, hidden, strict=True)
        )
        logits = []
        for action in legal_actions:
            if self.format_id == TACTICAL_RESOURCES_FORMAT:
                feature_fn = public_resources_tactical_action_features
            elif self.format_id in (TACTICAL_FORMAT, TACTICAL_STRUCTURED_FORMAT,
                                  TACTICAL_RELATIONAL_FORMAT, TACTICAL_DAMAGE_FORMAT,
                                  TACTICAL_RESOURCES_FORMAT):
                feature_fn = tactical_action_features
            elif self.format_id == NEURAL_FORMAT:
                feature_fn = neural_action_features
            else:
                feature_fn = policy_features
            action_vector = feature_fn(
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


    def _evaluate_instances(
        self, state: dict[str, Any], legal_actions: Sequence[LegalAction],
    ) -> PolicyValueEstimate:
        """Inference mirrors the Torch instance-aware forward path exactly."""
        if any(value is None for value in (
            self.enemy_weight, self.enemy_bias,
            self.enemy_context_weight, self.enemy_target_weight,
        )):
            raise ValueError("v8 model is missing learned enemy parameters")
        assert self.enemy_weight is not None
        assert self.enemy_bias is not None
        assert self.enemy_context_weight is not None
        assert self.enemy_target_weight is not None
        instances = enemy_instance_vectors(state, self.dimension)
        individual = {
            enemy_id: _relu(_sparse_matvec(
                self.enemy_weight, features, self.enemy_bias,
            ))
            for enemy_id, features in instances.items()
        }
        if self.format_id == TACTICAL_ATTENTION_FORMAT:
            attention = {name: getattr(self, name) for name in ATTENTION_FIELDS}
            if any(value is None for value in attention.values()):
                raise ValueError("v13 model is missing attention parameters")
            individual = contextualize_python(individual, attention)
        pooled = [
            sum(embedding[i] for embedding in individual.values()) / len(individual)
            if individual else 0.0
            for i in range(self.hidden)
        ]
        global_features = instance_global_features(state, self.dimension)
        projected_state = _sparse_matvec(
            self.state_weight, global_features, self.state_bias,
        )
        hidden = _relu([
            projected_state[i] + self.enemy_context_weight[i] * pooled[i]
            for i in range(self.hidden)
        ])
        value = math.tanh(self.value_bias + sum(
            x * y for x, y in zip(self.value_weight, hidden, strict=True)
        ))
        logits = []
        for action in legal_actions:
            target_id = action_target_instance(action.payload_json, instances)
            target = individual.get(target_id, [0.0] * self.hidden)
            action_features = instance_action_features(
                state, action.kind, action.payload_json, self.dimension,
            )
            projected = _sparse_matvec(
                self.action_weight, action_features, self.action_bias,
            )
            logits.append(self.policy_bias + sum(
                self.policy_weight[i] * max(
                    0.0, hidden[i] + projected[i] +
                    self.enemy_target_weight[i] * target[i],
                )
                for i in range(self.hidden)
            ))
        return PolicyValueEstimate(action_logits=tuple(logits), value=value)


def load_policy_value_model(path: Path) -> NeuralPolicyValueModel:
    return NeuralPolicyValueModel.load(path)
