"""H10 hybrid dispatch must be fair, phase-correct and action-menu preserving."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from sts2_ai.agents.base import Decision
from sts2_ai.agents.h10_hybrid import H10FrozenHybridAgent
from sts2_ai.emulator import FAIR_POLICY_ID, LegalAction, Observation
from sts2_ai.models.protocol import PolicyValueEstimate


class FakeMacro:
    policy_id = "macro-test"

    def __init__(self) -> None:
        self.calls = 0

    def choose(self, observation, legal_actions):
        self.calls += 1
        return Decision(legal_actions[1], self.policy_id)


class FakeCombat:
    model_id = "h9-test"

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, observation, legal_actions):
        self.calls += 1
        return PolicyValueEstimate((2.0, -1.0), 0.5)


def public(state: dict) -> Observation:
    return Observation(FAIR_POLICY_ID, json.dumps(state), "hash")


def test_hybrid_routes_macro_and_combat_without_changing_legal_action() -> None:
    macro, combat = FakeMacro(), FakeCombat()
    agent = H10FrozenHybridAgent(macro, combat, combat_sha256="a" * 64)
    actions = (LegalAction("a", "play_card"), LegalAction("b", "end_turn"))

    picked_macro = agent.choose(public({"hp": 60}), actions)
    assert picked_macro.action is actions[1]
    assert macro.calls == 1 and combat.calls == 0

    picked_combat = agent.choose(public({"hp": 60, "combat": {"turn": 1}}), actions)
    assert picked_combat.action is actions[0]
    assert macro.calls == 1 and combat.calls == 1
    assert picked_combat.policy_name == picked_macro.policy_name == agent.policy_id


def test_hybrid_rejects_private_or_bad_inputs() -> None:
    agent = H10FrozenHybridAgent(
        FakeMacro(), FakeCombat(), combat_sha256="a" * 64,
    )
    actions = (LegalAction("a", "end_turn"),)
    with pytest.raises(ValueError, match="fair public"):
        agent.choose(Observation("private", "{}", "hash"), actions)
    with pytest.raises(ValueError, match="legal actions"):
        agent.choose(public({"combat": {}}), ())
    with pytest.raises(ValueError, match="Malformed"):
        agent.choose(public([]), actions)
    with pytest.raises(ValueError, match="wrong legal-action count"):
        agent.choose(public({"combat": {}}), actions)
