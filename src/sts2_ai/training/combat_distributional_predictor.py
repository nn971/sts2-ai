"""Distributional PUBLIC combat outcome critic, separate from tactical policy.

Predicts survival probability and an HP histogram *conditional on survival*,
with explicit defeat mass. This is an observational, on-policy outcome
predictor, not a counterfactual action-value model; potion valuation remains
the responsibility of the strategic continuation critic.

HP is represented as a fraction of the observed maximum HP at exit.
Discrete bins deliberately retain an approximate distribution instead of
only E[HP]; exact HP outcomes remain available in the source JSONL.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_ai.models.hashed_linear import state_dict, state_features

DISTRIBUTIONAL_FORMAT = "sts2-combat-outcome-distribution-v1"
HP_BINS = 10


def hp_bin(hp_fraction: float) -> int:
    """A positive-survival HP fraction in one of ten equal-width bins."""
    if not 0.0 <= hp_fraction <= 1.0 or not math.isfinite(hp_fraction):
        raise ValueError("HP fraction must be finite and in [0, 1]")
    return min(HP_BINS - 1, int(hp_fraction * HP_BINS))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def _softmax(values: list[float]) -> tuple[float, ...]:
    high = max(values)
    exp = [math.exp(value - high) for value in values]
    total = sum(exp)
    return tuple(value / total for value in exp)


@dataclass(frozen=True)
class DistributionalCombatPrediction:
    """Joint loss/death and coarse post-combat HP outcomes.

    joint_hp_probabilities has 11 entries: first = defeat (HP 0), and
    remaining 10 = survived with HP fraction in corresponding positive bin.
    Defeats and zero-HP victories remain distinct in the training targets,
    even though a zero-HP victory is biologically unlikely in this game.
    """

    survival_probability: float
    conditional_hp_probabilities: tuple[float, ...]
    joint_hp_probabilities: tuple[float, ...]
    expected_exit_hp_fraction: float
    normalized_hp_variance: float

    def quantile(self, q: float) -> float:
        """Coarse (discrete) unconditional exit-HP quantile in [0,1]."""
        if not 0.0 <= q <= 1.0:
            raise ValueError("Quantile level must be in [0, 1]")
        support = (0.0,) + tuple(
            (i + 0.5) / HP_BINS for i in range(HP_BINS)
        )
        cumulative = 0.0
        for value, p in zip(support, self.joint_hp_probabilities, strict=True):
            cumulative += p
            if q <= cumulative + 1e-12:
                return value
        return support[-1]


@dataclass(slots=True)
class DistributionalCombatOutcomePredictor:
    dimension: int
    hidden: int
    state_weight: list[list[float]]
    state_bias: list[float]
    survival_weight: list[float]
    survival_bias: float
    hp_bin_weight: list[list[float]]
    hp_bin_bias: list[float]
    model_id: str

    def __post_init__(self) -> None:
        if self.dimension <= 0 or self.hidden <= 0 or not self.model_id:
            raise ValueError("Invalid distributional predictor architecture")
        if len(self.state_weight) != self.hidden or any(
            len(row) != self.dimension for row in self.state_weight
        ):
            raise ValueError("Wrong state projection shape")
        if len(self.state_bias) != self.hidden or (
            len(self.survival_weight) != self.hidden
        ):
            raise ValueError("Wrong survival projection shape")
        if len(self.hp_bin_weight) != HP_BINS or (
            len(self.hp_bin_bias) != HP_BINS
        ) or any(len(row) != self.hidden for row in self.hp_bin_weight):
            raise ValueError("Wrong exit-HP histogram shape")
        values = [
            self.survival_bias, *self.state_bias, *self.survival_weight,
            *self.hp_bin_bias,
            *(x for row in self.state_weight for x in row),
            *(x for row in self.hp_bin_weight for x in row),
        ]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Nonfinite distributional model weight")

    def predict(self, public_entry_json: str) -> DistributionalCombatPrediction:
        state = state_dict(public_entry_json)
        if not isinstance(state.get("combat"), dict):
            raise ValueError("Expected public combat-entry observation")
        features = state_features(state, self.dimension)
        representation = [
            max(0.0, self.state_bias[j] + sum(
                self.state_weight[j][i] * x for i, x in features.items()
            )) for j in range(self.hidden)
        ]
        survival = _sigmoid(
            self.survival_bias + sum(
                x * y for x, y in zip(
                    representation, self.survival_weight, strict=True
                )
            )
        )
        conditional = _softmax([
            bias + sum(x * y for x, y in zip(
                representation, weights, strict=True
            ))
            for weights, bias in zip(
                self.hp_bin_weight, self.hp_bin_bias, strict=True
            )
        ])
        joint = (1.0 - survival,) + tuple(survival * p for p in conditional)
        support = (0.0,) + tuple(
            (i + 0.5) / HP_BINS for i in range(HP_BINS)
        )
        mean = sum(p * value for p, value in zip(joint, support, strict=True))
        variance = sum(
            p * (value - mean) ** 2
            for p, value in zip(joint, support, strict=True)
        )
        return DistributionalCombatPrediction(
            survival, conditional, joint, mean, max(0.0, variance),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": DISTRIBUTIONAL_FORMAT,
            "dimension": self.dimension,
            "hidden": self.hidden,
            "state_weight": self.state_weight,
            "state_bias": self.state_bias,
            "survival_weight": self.survival_weight,
            "survival_bias": self.survival_bias,
            "hp_bin_weight": self.hp_bin_weight,
            "hp_bin_bias": self.hp_bin_bias,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DistributionalCombatOutcomePredictor:
        if raw.get("format") != DISTRIBUTIONAL_FORMAT:
            raise ValueError("Incompatible combat distribution model format")
        return cls(
            dimension=int(raw["dimension"]),
            hidden=int(raw["hidden"]),
            state_weight=[[float(v) for v in row] for row in raw["state_weight"]],
            state_bias=[float(x) for x in raw["state_bias"]],
            survival_weight=[float(x) for x in raw["survival_weight"]],
            survival_bias=float(raw["survival_bias"]),
            hp_bin_weight=[
                [float(x) for x in row] for row in raw["hp_bin_weight"]
            ],
            hp_bin_bias=[float(x) for x in raw["hp_bin_bias"]],
            model_id=str(raw["model_id"]),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), sort_keys=True) + "\n",
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> DistributionalCombatOutcomePredictor:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Invalid distributional model file")
        return cls.from_dict(raw)
