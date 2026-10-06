from __future__ import annotations

import json

from sts2_ai.emulator import Observation
from sts2_ai.evaluation import PrototypeHeuristicEvaluator


def test_prototype_heuristic_reads_canonical_snake_case_fields() -> None:
    evaluator = PrototypeHeuristicEvaluator()

    base = Observation(
        policy_id="prototype-fair-v0",
        payload_json=json.dumps(
            {
                "act": 1,
                "floor": 2,
                "hp": 35,
                "max_hp": 70,
                "gold": 100,
                "deck": [
                    {"upgrade_level": 0},
                    {"upgrade_level": 1},
                ],
                "relics": [{"relic_id": "r"}],
                "potions": [{"slot": 0, "potion_id": "p"}],
                "terminal_outcome": None,
            }
        ),
        observation_hash="h",
    )
    healthier = Observation(
        policy_id=base.policy_id,
        payload_json=json.dumps(
            {
                **json.loads(base.payload_json),
                "hp": 70,
            }
        ),
        observation_hash="h2",
    )

    assert evaluator.evaluate(healthier, terminal=False) > evaluator.evaluate(
        base,
        terminal=False,
    )


def test_prototype_heuristic_terminal_values_dominate() -> None:
    evaluator = PrototypeHeuristicEvaluator()
    victory = Observation(
        "prototype-fair-v0",
        '{"terminal_outcome":"victory"}',
        "v",
    )
    defeat = Observation(
        "prototype-fair-v0",
        '{"terminal_outcome":"defeat"}',
        "d",
    )

    assert evaluator.evaluate(victory, terminal=True) == 1_000_000.0
    assert evaluator.evaluate(defeat, terminal=True) == -1_000_000.0
