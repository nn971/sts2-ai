"""Opt-in H10 fair public hybrid: frozen H9 combat + existing macro decisions.

Both policies consume only the currently published observation and legal menu.
No Torch checkpoint or old phase-split model is converted to another format.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from sts2_ai.agents.base import Decision
from sts2_ai.agents.phase_split_agent import PhaseSplitGreedyAgent
from sts2_ai.emulator import FAIR_POLICY_ID, LegalAction, Observation
from sts2_ai.models.h7_public_policy import H7PlayerSpec, TorchH7PublicPolicy


class H10FrozenHybridAgent:
    """Greedy macro + H9 combat. Experimental native-game compatibility gate."""

    def __init__(
        self, macro: PhaseSplitGreedyAgent, combat: TorchH7PublicPolicy,
        *, combat_sha256: str,
    ) -> None:
        if len(combat_sha256) != 64:
            raise ValueError("H10 combat model must have pinned SHA-256 identity")
        self.macro = macro
        self.combat = combat
        self.policy_id = (
            f"h10-frozen-hybrid-{macro.policy_id}-{combat_sha256[:16]}"
        )

    @classmethod
    def load(
        cls, *, macro_model: Path, combat_checkpoint: Path,
        combat_device: str = "cpu",
    ) -> H10FrozenHybridAgent:
        macro = PhaseSplitGreedyAgent.load(macro_model)
        spec = H7PlayerSpec.from_file(combat_checkpoint, seed=31337, greedy=True)
        combat = spec.load_pinned()
        combat.model.to(combat_device).eval()
        return cls(macro, combat, combat_sha256=spec.model_sha256)

    def choose(
        self, observation: Observation, legal_actions: Sequence[LegalAction],
    ) -> Decision:
        if observation.policy_id != FAIR_POLICY_ID or not legal_actions:
            raise ValueError("H10 agent requires a fair public frame and legal actions")
        state = json.loads(observation.payload_json)
        if not isinstance(state, dict):
            raise ValueError("Malformed public run state")
        if not isinstance(state.get("combat"), dict):
            result = self.macro.choose(observation, legal_actions)
            return Decision(result.action, policy_name=self.policy_id)
        logits = self.combat.evaluate(observation, legal_actions).action_logits
        if len(logits) != len(legal_actions):
            raise ValueError("H9 combat returned wrong legal-action count")
        index = max(
            range(len(legal_actions)),
            key=lambda i: (logits[i], legal_actions[i].action_id),
        )
        return Decision(legal_actions[index], policy_name=self.policy_id)
