from __future__ import annotations

import json

from sts2_ai.emulator import LegalAction
from sts2_ai.evaluation import PrototypeSearchDecision
from sts2_ai.scenarios import (
    ARCHIVE_SCHEMA_ID,
    mine_close_search_scenarios,
    write_replay_scenario_archive,
)
from sts2_ai.search import ActionEvaluation


def _evaluation(action_id: str, value: float, uncertainty: float | None) -> ActionEvaluation:
    return ActionEvaluation(
        action=LegalAction(action_id=action_id, kind="choose", payload_json="{}"),
        value=value,
        visits=8,
        uncertainty=uncertainty,
    )


def test_close_decision_miner_ranks_by_margin_then_uncertainty() -> None:
    decisions = (
        PrototypeSearchDecision(
            seed="a",
            decision_index=3,
            state_hash="hash-a",
            action_history=("start", "left"),
            evaluations=(
                _evaluation("a1", 1.0, 0.2),
                _evaluation("a2", 0.9, 0.3),
            ),
        ),
        PrototypeSearchDecision(
            seed="b",
            decision_index=7,
            state_hash="hash-b",
            action_history=("start", "right"),
            evaluations=(
                _evaluation("b1", 1.0, 0.8),
                _evaluation("b2", 0.9, 0.9),
            ),
        ),
        PrototypeSearchDecision(
            seed="c",
            decision_index=9,
            state_hash="hash-c",
            action_history=("start", "middle"),
            evaluations=(
                _evaluation("c1", 1.0, 0.9),
                _evaluation("c2", 0.7, 0.9),
            ),
        ),
    )

    scenarios = mine_close_search_scenarios(
        decisions,
        limit=2,
        emulator_revision="emu-1",
        information_policy="fair-v0",
    )

    assert [scenario.run_seed for scenario in scenarios] == ["b", "a"]
    assert scenarios[0].value_margin == scenarios[1].value_margin == 0.1
    assert scenarios[0].max_uncertainty == 0.9
    assert scenarios[0].action_history == ("start", "right")


def test_scenario_identity_is_deterministic_and_archive_is_json(tmp_path) -> None:
    decision = PrototypeSearchDecision(
        seed="seed",
        decision_index=5,
        state_hash="state-hash",
        action_history=("one", "two"),
        evaluations=(
            _evaluation("left", 10.0, None),
            _evaluation("right", 9.5, None),
        ),
    )

    first = mine_close_search_scenarios(
        (decision,),
        limit=1,
        emulator_revision="emu-2",
        information_policy="fair-v0",
    )
    second = mine_close_search_scenarios(
        (decision,),
        limit=1,
        emulator_revision="emu-2",
        information_policy="fair-v0",
    )
    assert first == second

    output = tmp_path / "scenarios.json"
    write_replay_scenario_archive(output, first)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_id"] == ARCHIVE_SCHEMA_ID
    assert payload["count"] == 1
    assert payload["scenarios"][0]["run_seed"] == "seed"
    assert payload["scenarios"][0]["action_history"] == ["one", "two"]
    assert payload["scenarios"][0]["exact_state_hash"] == "state-hash"
