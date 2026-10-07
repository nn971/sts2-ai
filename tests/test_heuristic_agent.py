import json

from sts2_ai.agents import HeuristicAgent
from sts2_ai.emulator import LegalAction, Observation


def _observation(payload: dict[str, object]) -> Observation:
    return Observation(
        policy_id="test",
        payload_json=json.dumps(payload),
        observation_hash="hash",
    )


def test_heuristic_heals_when_low() -> None:
    agent = HeuristicAgent()
    observation = _observation({"hp": 20, "max_hp": 70})
    actions = (
        LegalAction("heal", "rest_heal"),
        LegalAction("upgrade", "rest_upgrade"),
    )
    assert agent.choose(observation, actions).action.kind == "rest_heal"


def test_heuristic_upgrades_when_healthy() -> None:
    agent = HeuristicAgent()
    observation = _observation({"hp": 65, "max_hp": 70})
    actions = (
        LegalAction("heal", "rest_heal"),
        LegalAction("upgrade", "rest_upgrade"),
    )
    assert agent.choose(observation, actions).action.kind == "rest_upgrade"
