"""Observation-only learned rollout policy for opt-in MCTS experiments."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from sts2_ai.agents.base import Decision
from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.neural import NeuralPolicyValueModel


class NeuralGreedyAgent:
    def __init__(self, model: NeuralPolicyValueModel, *, content_hash: str) -> None:
        self._model = model
        self.policy_id = f"neural-greedy-v1-sha256-{content_hash}"

    @classmethod
    def load(cls, path: Path) -> NeuralGreedyAgent:
        payload = path.read_bytes()
        return cls(
            NeuralPolicyValueModel.load(path),
            content_hash=hashlib.sha256(payload).hexdigest(),
        )

    def choose(
        self,
        observation: Observation,
        legal_actions: Sequence[LegalAction],
    ) -> Decision:
        if not legal_actions:
            raise ValueError("Cannot select from zero legal actions")
        logits = self._model.evaluate(observation, legal_actions).action_logits
        index = max(range(len(logits)), key=lambda i: (logits[i], legal_actions[i].action_id))
        return Decision(
            action=legal_actions[index],
            policy_name=self.policy_id,
        )
