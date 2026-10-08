"""Public-only categorical neural policy with explicitly controlled temperature.

This diagnostic policy changes inference action sampling only. It does not
modify the on-policy REINFORCE training distribution or checkpoint contract.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence

from sts2_ai.agents.base import Decision
from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.neural import NeuralPolicyValueModel


def temperature_probabilities(
    logits: Sequence[float], temperature: float
) -> tuple[float, ...]:
    """Stable categorical softmax of logits / T, T strictly positive."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Sampling temperature must be finite and positive")
    if not logits or not all(math.isfinite(x) for x in logits):
        raise ValueError("Expected nonempty finite action logits")
    peak = max(logits)
    # Subtract the peak before dividing to avoid overflow, including at tiny T.
    weights = [math.exp((value - peak) / temperature) for value in logits]
    normalizer = sum(weights)
    return tuple(weight / normalizer for weight in weights)


class NeuralTemperatureAgent:
    """Draw actions only from legal public choices using softmax(logits / T)."""

    def __init__(
        self, model: NeuralPolicyValueModel, *, temperature: float, rng: random.Random
    ) -> None:
        # Validate temperature immediately, including one-action edge cases.
        temperature_probabilities((0.0,), temperature)
        self._model = model
        self.temperature = temperature
        self._rng = rng
        self.policy_id = (
            f"neural-temperature-v1-t{temperature:g}-{model.model_id}"
        )

    def choose(
        self,
        observation: Observation,
        legal_actions: Sequence[LegalAction],
    ) -> Decision:
        if not legal_actions:
            raise ValueError("Cannot choose from zero legal actions")
        logits = self._model.evaluate(observation, legal_actions).action_logits
        probabilities = temperature_probabilities(logits, self.temperature)
        threshold = self._rng.random()
        cumulative = 0.0
        chosen_index = len(probabilities) - 1
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if threshold < cumulative:
                chosen_index = index
                break

        entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        max_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
        return Decision(
            action=legal_actions[chosen_index],
            policy_name=self.policy_id,
            metadata_json=json.dumps({
                "temperature": self.temperature,
                "legal_action_count": len(probabilities),
                "chosen_probability": probabilities[chosen_index],
                "max_probability": max(probabilities),
                "entropy": entropy,
                "normalized_entropy": entropy / max_entropy if max_entropy else 0.0,
            }, sort_keys=True),
        )
