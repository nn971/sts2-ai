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


def test_heuristic_reads_pascal_case_map_payload() -> None:
    agent = HeuristicAgent()
    observation = _observation(
        {
            "hp": 20,
            "max_hp": 70,
            "map": [
                {"node_id": "fight", "room_type": 0},
                {"node_id": "rest", "room_type": 4},
            ],
        }
    )
    actions = (
        LegalAction(
            "choose_map_node:fight",
            "choose_map_node",
            payload_json='{"NodeId":"fight"}',
        ),
        LegalAction(
            "choose_map_node:rest",
            "choose_map_node",
            payload_json='{"NodeId":"rest"}',
        ),
    )

    assert agent.choose(observation, actions).action.action_id == "choose_map_node:rest"


def test_heuristic_reads_pascal_case_combat_payload() -> None:
    agent = HeuristicAgent()
    observation = _observation(
        {
            "hp": 60,
            "max_hp": 70,
            "combat": {
                "energy": 3,
                "hand": [
                    {"instance_id": 10, "card_id": "Strike"},
                    {"instance_id": 11, "card_id": "Defend"},
                ],
                "enemies": [
                    {"instance_id": 2, "enemy_id": "test_enemy", "hp": 6},
                ],
            },
        }
    )
    actions = (
        LegalAction(
            "play_card:strike",
            "play_card",
            payload_json='{"CardInstanceId":10,"TargetEnemyId":2}',
        ),
        LegalAction(
            "play_card:defend",
            "play_card",
            payload_json='{"CardInstanceId":11,"TargetEnemyId":null}',
        ),
    )

    assert agent.choose(observation, actions).action.action_id == "play_card:strike"


def test_heuristic_prefers_defense_when_hp_is_low() -> None:
    agent = HeuristicAgent()
    observation = _observation(
        {
            "hp": 28,
            "max_hp": 70,
            "combat": {
                "energy": 3,
                "player_block": 0,
                "hand": [
                    {"instance_id": 10, "card_id": "Strike"},
                    {"instance_id": 11, "card_id": "Defend"},
                ],
                "enemies": [
                    {"instance_id": 2, "enemy_id": "test_enemy", "hp": 40},
                ],
            },
        }
    )
    actions = (
        LegalAction(
            "play_card:strike",
            "play_card",
            payload_json='{"CardInstanceId":10,"TargetEnemyId":2}',
        ),
        LegalAction(
            "play_card:defend",
            "play_card",
            payload_json='{"CardInstanceId":11,"TargetEnemyId":null}',
        ),
    )

    assert agent.choose(observation, actions).action.action_id == "play_card:defend"


def test_existing_block_reduces_defense_urgency() -> None:
    agent = HeuristicAgent()
    observation = _observation(
        {
            "hp": 28,
            "max_hp": 70,
            "combat": {
                "energy": 3,
                "player_block": 15,
                "hand": [
                    {"instance_id": 10, "card_id": "Strike"},
                    {"instance_id": 11, "card_id": "Defend"},
                ],
                "enemies": [
                    {"instance_id": 2, "enemy_id": "test_enemy", "hp": 40},
                ],
            },
        }
    )
    actions = (
        LegalAction(
            "play_card:strike",
            "play_card",
            payload_json='{"CardInstanceId":10,"TargetEnemyId":2}',
        ),
        LegalAction(
            "play_card:defend",
            "play_card",
            payload_json='{"CardInstanceId":11,"TargetEnemyId":null}',
        ),
    )

    assert agent.choose(observation, actions).action.action_id == "play_card:strike"
