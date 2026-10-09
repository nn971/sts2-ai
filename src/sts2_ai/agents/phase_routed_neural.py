"""Two-policy inference router with public-only phase detection.

The strategy model owns noncombat decisions; the tactical model owns combat
decisions. Both score only the emulator's mechanically legal actions. This
does NOT itself claim either policy has been trained with combat-local rewards.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from collections.abc import Sequence

from sts2_ai.agents.base import Decision
from sts2_ai.agents.neural_temperature import temperature_probabilities
from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.neural import NeuralPolicyValueModel

PHASE_ROUTER_VERSION = "sts2-public-phase-router-v1"


def public_phase(observation: Observation) -> str:
    data = json.loads(observation.payload_json)
    if not isinstance(data, dict):
        raise ValueError("Public observation is not a JSON object")
    return "combat" if isinstance(data.get("combat"), dict) else "strategy"


class PhaseRoutedNeuralAgent:
    """Deploy distinct portable neural networks, without hidden state access."""

    def __init__(
        self,
        *,
        strategy: NeuralPolicyValueModel,
        tactical: NeuralPolicyValueModel,
        temperature: float | None = None,
        seed: int = 0,
    ) -> None:
        if temperature is not None:
            temperature_probabilities((0.0,), temperature)
        self.strategy = strategy
        self.tactical = tactical
        self.temperature = temperature
        self._rng = random.Random(seed)
        self.policy_id = (
            f"{PHASE_ROUTER_VERSION}-strategy-{strategy.model_id}"
            f"-combat-{tactical.model_id}"
        )

    @classmethod
    def load(
        cls, strategy_path: Path, tactical_path: Path,
        *, temperature: float | None = None, seed: int = 0
    ) -> PhaseRoutedNeuralAgent:
        return cls(
            strategy=NeuralPolicyValueModel.load(strategy_path),
            tactical=NeuralPolicyValueModel.load(tactical_path),
            temperature=temperature, seed=seed,
        )

    def choose(
        self, observation: Observation, legal_actions: Sequence[LegalAction]
    ) -> Decision:
        if not legal_actions:
            raise ValueError("Cannot choose from empty legal action menu")
        phase = public_phase(observation)
        policy = self.tactical if phase == "combat" else self.strategy
        logits = policy.evaluate(observation, legal_actions).action_logits
        if self.temperature is None:
            index = max(
                range(len(logits)),
                key=lambda i: (logits[i], legal_actions[i].action_id),
            )
        else:
            probs = temperature_probabilities(logits, self.temperature)
            needle = self._rng.random()
            accumulated = 0.0
            index = len(probs) - 1
            for i, probability in enumerate(probs):
                accumulated += probability
                if needle < accumulated:
                    index = i
                    break
        return Decision(
            action=legal_actions[index],
            policy_name=self.policy_id,
            metadata_json=json.dumps({
                "phase": phase,
                "selected_model_id": policy.model_id,
                "temperature": self.temperature,
            }, sort_keys=True),
        )
