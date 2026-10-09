"""Short-horizon public combat-outcome predictor.

The model predicts combat survival and normalized exit HP from a PUBLIC
combat-entry observation. It is trained on individual outcomes, not terminal
run rewards or teacher moves. The expected exit-HP output is a diagnostic,
NOT a fixed exchange rate for potions or a final tactical objective.

Samples retain their full exit resource vectors; later models may estimate
normalized variance, conditional quantiles, or joint outcome distributions.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_ai.models.hashed_linear import state_features

OUTCOME_FORMAT = "sts2-combat-outcome-predictor-v1"


def _dense(features: dict[int, float], dimension: int) -> list[float]:
    return [features.get(i, 0.0) for i in range(dimension)]


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    value = math.exp(x)
    return value / (1.0 + value)


@dataclass(frozen=True, slots=True)
class CombatPrediction:
    survival_probability: float
    expected_exit_hp_ratio: float


@dataclass(slots=True)
class CombatOutcomePredictor:
    dimension: int
    hidden: int
    state_weight: list[list[float]]
    state_bias: list[float]
    survival_weight: list[float]
    survival_bias: float
    hp_weight: list[float]
    hp_bias: float
    model_id: str

    def __post_init__(self) -> None:
        if self.dimension <= 0 or self.hidden <= 0 or not self.model_id:
            raise ValueError("Invalid combat outcome model architecture")
        if len(self.state_weight) != self.hidden or any(
            len(row) != self.dimension for row in self.state_weight
        ):
            raise ValueError("Invalid state projection shape")
        if any(len(row) != self.hidden for row in (
            self.state_bias, self.survival_weight, self.hp_weight
        )):
            raise ValueError("Invalid outcome projection shape")
        flat = [
            value for row in self.state_weight for value in row
        ] + self.state_bias + self.survival_weight + self.hp_weight + [
            self.survival_bias, self.hp_bias
        ]
        if not all(math.isfinite(x) for x in flat):
            raise ValueError("Nonfinite model parameter")

    def predict(self, public_entry_json: str) -> CombatPrediction:
        raw = json.loads(public_entry_json)
        if not isinstance(raw, dict) or not isinstance(raw.get("combat"), dict):
            raise ValueError("Prediction requires public combat entry observation")
        features = state_features(raw, self.dimension)
        hidden = [
            max(0.0, self.state_bias[k] + sum(
                self.state_weight[k][i] * x for i, x in features.items()
            ))
            for k in range(self.hidden)
        ]
        return CombatPrediction(
            survival_probability=_sigmoid(
                self.survival_bias + sum(
                    x * y for x, y in zip(hidden, self.survival_weight, strict=True)
                )
            ),
            expected_exit_hp_ratio=_sigmoid(
                self.hp_bias + sum(
                    x * y for x, y in zip(hidden, self.hp_weight, strict=True)
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": OUTCOME_FORMAT,
            "dimension": self.dimension,
            "hidden": self.hidden,
            "model_id": self.model_id,
            "state_weight": self.state_weight,
            "state_bias": self.state_bias,
            "survival_weight": self.survival_weight,
            "survival_bias": self.survival_bias,
            "hp_weight": self.hp_weight,
            "hp_bias": self.hp_bias,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CombatOutcomePredictor:
        if raw.get("format") != OUTCOME_FORMAT:
            raise ValueError("Unsupported combat outcome predictor format")
        dimension = raw.get("dimension")
        hidden = raw.get("hidden")
        if type(dimension) is not int or type(hidden) is not int:
            raise ValueError("Invalid predictor dimensions")
        return cls(
            dimension, hidden,
            [[float(x) for x in row] for row in raw["state_weight"]],
            [float(x) for x in raw["state_bias"]],
            [float(x) for x in raw["survival_weight"]],
            float(raw["survival_bias"]),
            [float(x) for x in raw["hp_weight"]],
            float(raw["hp_bias"]),
            str(raw["model_id"]),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> CombatOutcomePredictor:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("Invalid outcome predictor file")
        return cls.from_dict(raw)


@dataclass(frozen=True, slots=True)
class OutcomeExample:
    seed: str
    entry_public_json: str
    survived: float
    exit_hp_ratio: float


def load_outcome_samples(directory: Path) -> tuple[OutcomeExample, ...]:
    """Load only combat-entry observations with known, resolved outcome.

    Older sample schemas have no entry public observation; fail explicitly,
    rather than silently learning from leaked exit state.
    """
    results: list[OutcomeExample] = []
    paths = sorted(directory.glob("round-*.jsonl"))
    if not paths:
        raise ValueError("No per-round combat sample files found")
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if not isinstance(sample, dict) or sample.get("schema") != (
                "sts2-public-combat-sample-v1"
            ):
                raise ValueError(f"Unsupported combat sample: {path}:{number}")
            outcome = sample.get("outcome")
            if not isinstance(outcome, dict):
                raise ValueError("Malformed combat sample")
            entry_json = outcome.get("entry_public_json")
            exit_resources = outcome.get("exit")
            if not isinstance(entry_json, str) or not entry_json:
                raise ValueError(
                    "Combat sample lacks entry_public_json; recollect with the new "
                    "combat-outcome recorder (older samples cannot be upgraded)"
                )
            if not isinstance(exit_resources, dict):
                raise ValueError("Malformed public exit resources")
            hp = exit_resources.get("hp")
            max_hp = exit_resources.get("max_hp")
            if type(hp) is not int or type(max_hp) is not int or max_hp <= 0:
                raise ValueError("Malformed observed exit HP")
            result = outcome.get("result")
            if result not in ("victory", "defeat"):
                raise ValueError("Unresolved or censored combats cannot be labeled")
            seed = sample.get("seed")
            if not isinstance(seed, str) or not seed:
                raise ValueError("Missing sample seed")
            results.append(OutcomeExample(
                seed, entry_json, float(result == "victory"),
                max(0.0, min(1.0, hp / max_hp)),
            ))
    if not results:
        raise ValueError("No completed combat outcomes")
    return tuple(results)
