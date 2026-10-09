"""Greedy public-observation agent with independent tactical/strategic heads."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from sts2_ai.agents.base import Decision
from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.phase_split import PhaseSplitNeuralModel


class PhaseSplitGreedyAgent:
    def __init__(self, model: PhaseSplitNeuralModel) -> None:
        self._model = model
        self.policy_id = "phase-split-greedy-v1-" + model.model_id

    @classmethod
    def load(cls, path: Path) -> PhaseSplitGreedyAgent:
        return cls(PhaseSplitNeuralModel.load(path))

    def choose(
        self, observation: Observation, legal_actions: Sequence[LegalAction]
    ) -> Decision:
        if not legal_actions:
            raise ValueError("No legal actions")
        logits = self._model.evaluate(observation, legal_actions).action_logits
        index = max(
            range(len(logits)), key=lambda i: (
                logits[i], legal_actions[i].action_id
            ),
        )
        return Decision(legal_actions[index], policy_name=self.policy_id)
