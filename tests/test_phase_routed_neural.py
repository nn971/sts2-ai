"""Inference phase routing never uses hidden state and obeys legal menus."""
from __future__ import annotations

import json

from sts2_ai.agents.phase_routed_neural import PhaseRoutedNeuralAgent, public_phase
from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.neural import NeuralPolicyValueModel


def model(name: str, score: float) -> NeuralPolicyValueModel:
    return NeuralPolicyValueModel.from_dict({
        "format": "sts2-neural-policy-value-v2-semantic-action",
        "dimension": 16, "hidden": 1, "model_id": name,
        "value_head_trained": False,
        "state_weight": [[0.0] * 16], "state_bias": [0.0],
        "action_weight": [[0.0] * 16], "action_bias": [0.0],
        "policy_weight": [0.0], "policy_bias": score,
        "value_weight": [0.0], "value_bias": 0.0,
    })


def frame(combat: bool) -> Observation:
    data = {"hp": 20, "combat": {"turn": 1, "enemies": []} if combat else None}
    return Observation("prototype-fair-v0", json.dumps(data), "test")


def test_phase_detection_is_public_only() -> None:
    assert public_phase(frame(True)) == "combat"
    assert public_phase(frame(False)) == "strategy"


def test_routed_decision_identifies_selected_policy_and_is_legal() -> None:
    agent = PhaseRoutedNeuralAgent(
        strategy=model("strategy-test", 1.0),
        tactical=model("combat-test", -1.0),
    )
    actions = (LegalAction("a", "end_turn"), LegalAction("b", "play_card"))
    decision = agent.choose(frame(True), actions)
    assert decision.action in actions
    assert json.loads(decision.metadata_json)["selected_model_id"] == "combat-test"
    decision = agent.choose(frame(False), actions)
    assert decision.action in actions
    assert json.loads(decision.metadata_json)["selected_model_id"] == "strategy-test"
