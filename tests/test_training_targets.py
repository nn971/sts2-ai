import json
from pathlib import Path

import pytest

from sts2_ai.emulator import LegalAction
from sts2_ai.search import ActionEvaluation, SearchResult
from sts2_ai.strategy_db import (
    SQLiteStrategyStore,
    SearchActionEvidence,
    SearchObservationEvidence,
    SearchRootEvidence,
    record_search_result,
)
from sts2_ai.training import (
    build_training_example,
    build_training_examples,
    write_training_jsonl,
)


def test_build_training_example_normalizes_visits_and_uses_best_visited_value() -> None:
    root = SearchRootEvidence(
        state_hash="state",
        observation_hash="obs",
        information_policy="prototype-fair-v0",
        search_regime="oracle-exact",
        legal_action_ids=("a", "b"),
        chosen_action_id="b",
        search_budget=32,
        expanded_nodes=10,
        transitions=100,
        transposition_hits=2,
        search_version="search-v1",
        model_id=None,
        emulator_revision="emu-1",
        game_build="build-1",
    )
    actions = (
        SearchActionEvidence(
            state_hash="state",
            information_policy="prototype-fair-v0",
            search_regime="oracle-exact",
            action_id="a",
            action_kind="play_card",
            action_payload_json='{"CardInstanceId":1}',
            value=0.4,
            visits=24,
            uncertainty=0.1,
            search_budget=32,
            search_version="search-v1",
            model_id=None,
            emulator_revision="emu-1",
            game_build="build-1",
        ),
        SearchActionEvidence(
            state_hash="state",
            information_policy="prototype-fair-v0",
            search_regime="oracle-exact",
            action_id="b",
            action_kind="end_turn",
            action_payload_json="{}",
            value=0.6,
            visits=8,
            uncertainty=0.2,
            search_budget=32,
            search_version="search-v1",
            model_id=None,
            emulator_revision="emu-1",
            game_build="build-1",
        ),
    )
    observation = SearchObservationEvidence(
        observation_hash="obs",
        information_policy="prototype-fair-v0",
        payload_json='{"phase":3}',
    )

    example = build_training_example(root, actions, observation)

    assert [target.action_id for target in example.policy_targets] == ["a", "b"]
    assert [target.probability for target in example.policy_targets] == pytest.approx(
        [0.75, 0.25]
    )
    assert example.policy_targets[0].action_kind == "play_card"
    assert example.value_target == pytest.approx(0.6)
    assert example.observation_json == '{"phase":3}'
    assert example.search_budget == 32
    assert len(example.source_search_id) == 64


def _search_result(
    *,
    budget: int,
    first_visits: int,
    second_visits: int,
    first_value: float,
    second_value: float,
) -> SearchResult:
    first = LegalAction("a", "play_card", '{"CardInstanceId":1}')
    second = LegalAction("b", "end_turn", "{}")
    return SearchResult(
        root_state_hash="shared-state",
        root_observation_hash="shared-observation",
        root_observation_json='{"phase":3,"hp":50,"max_hp":70}',
        evaluations=(
            ActionEvaluation(first, value=first_value, visits=first_visits),
            ActionEvaluation(second, value=second_value, visits=second_visits),
        ),
        expanded_nodes=budget,
        transitions=budget * 4,
        transposition_hits=0,
        search_version="search-v1",
    )


def test_training_export_uses_strongest_budget_per_state_by_default(
    tmp_path: Path,
) -> None:
    database = tmp_path / "strategy.sqlite"
    with SQLiteStrategyStore(database) as store:
        low = _search_result(
            budget=8,
            first_visits=6,
            second_visits=2,
            first_value=0.2,
            second_value=0.1,
        )
        high = _search_result(
            budget=32,
            first_visits=8,
            second_visits=24,
            first_value=0.3,
            second_value=0.5,
        )
        record_search_result(
            store,
            low,
            low.evaluations[0].action,
            information_policy="prototype-fair-v0",
            search_regime="oracle-exact",
            search_budget=8,
            emulator_revision="emu-1",
            game_build="build-1",
        )
        record_search_result(
            store,
            high,
            high.evaluations[1].action,
            information_policy="prototype-fair-v0",
            search_regime="oracle-exact",
            search_budget=32,
            emulator_revision="emu-1",
            game_build="build-1",
        )

        examples = build_training_examples(
            store,
            "prototype-fair-v0",
            search_version="search-v1",
        )

    assert len(examples) == 1
    assert examples[0].search_budget == 32
    assert [target.probability for target in examples[0].policy_targets] == pytest.approx(
        [0.25, 0.75]
    )

    output = tmp_path / "targets.jsonl"
    assert write_training_jsonl(examples, output) == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["search_budget"] == 32
    assert payload["source_state_hash"] == "shared-state"
