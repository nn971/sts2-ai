"""Public-only two-policy model: combat tactics and out-of-combat strategy.

Each policy has independent weights. The phase gate inspects only the fair
public combat field. All mechanics and legal-action filtering remain with the
emulator. This model is a portable wrapper, not a joint training algorithm.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.neural import (
    NEURAL_FORMAT,
    TACTICAL_FORMAT,
    TACTICAL_STRUCTURED_FORMAT,
    NeuralPolicyValueModel,
)
from sts2_ai.models.protocol import PolicyValueEstimate

PHASE_SPLIT_FORMAT = "sts2-phase-split-policy-value-v1"


def observation_phase(observation: Observation) -> str:
    state = json.loads(observation.payload_json)
    if not isinstance(state, dict):
        raise ValueError("Public observation must be an object")
    return "combat" if isinstance(state.get("combat"), dict) else "strategy"


@dataclass(slots=True)
class PhaseSplitNeuralModel:
    strategy: NeuralPolicyValueModel
    combat: NeuralPolicyValueModel
    model_id: str
    combat_value_objective: str = "continuation"

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("Phase-split model requires a model identifier")
        if self.combat_value_objective not in ("continuation", "hp_first"):
            raise ValueError("Unsupported combat value objective")
        if self.strategy.format_id != NEURAL_FORMAT:
            raise ValueError("Strategy head must use the semantic action format")
        if self.combat.format_id not in (TACTICAL_FORMAT, TACTICAL_STRUCTURED_FORMAT):
            raise ValueError("Combat head must use target-aware tactical format")
        if (self.strategy.dimension, self.strategy.hidden) != (
            self.combat.dimension, self.combat.hidden
        ):
            raise ValueError("Phase-split heads must agree on their architecture")

    def evaluate(
        self, observation: Observation, legal_actions: Sequence[LegalAction]
    ) -> PolicyValueEstimate:
        model = self.combat if observation_phase(observation) == "combat" else self.strategy
        return model.evaluate(observation, legal_actions)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "format": PHASE_SPLIT_FORMAT,
            "model_id": self.model_id,
            "strategy": self.strategy.to_dict(),
            "combat": self.combat.to_dict(),
        }
        # Preserve the exact original serialization for old checkpoints.
        if self.combat_value_objective != "continuation":
            payload["combat_value_objective"] = self.combat_value_objective
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PhaseSplitNeuralModel:
        if raw.get("format") != PHASE_SPLIT_FORMAT:
            raise ValueError("Unsupported phase-split model format")
        if not isinstance(raw.get("strategy"), dict) or not isinstance(raw.get("combat"), dict):
            raise ValueError("Both policy heads are required")
        model_id = raw.get("model_id")
        if not isinstance(model_id, str):
            raise ValueError("Invalid phase-split model id")
        return cls(
            NeuralPolicyValueModel.from_dict(raw["strategy"]),
            NeuralPolicyValueModel.from_dict(raw["combat"]),
            model_id,
            combat_value_objective=str(raw.get("combat_value_objective", "continuation")),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> PhaseSplitNeuralModel:
        obj = json.loads(path.read_text())
        if not isinstance(obj, dict):
            raise ValueError("Invalid model file")
        return cls.from_dict(obj)


def load_public_model(payload: dict[str, Any]) -> NeuralPolicyValueModel | PhaseSplitNeuralModel:
    """Dispatch only on an explicit versioned model format."""
    if payload.get("format") == PHASE_SPLIT_FORMAT:
        return PhaseSplitNeuralModel.from_dict(payload)
    return NeuralPolicyValueModel.from_dict(payload)
