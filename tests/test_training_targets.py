import json
from dataclasses import replace
from pathlib import Path

import pytest

from sts2_ai.emulator import LegalAction
from sts2_ai.search import ActionEvaluation, SearchResult
from sts2_ai.strategy_db import (
    SearchActionEvidence,
    SearchObservationEvidence,
    SearchRootEvidence,
    SQLiteStrategyStore,
    record_search_result,
)
from sts2_ai.training import (
    PolicyTarget,
    TrainingExample,
    build_training_example,
    build_training_examples,
    split_training_examples,
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


def test_grouped_split_keeps_repeated_search_states_together() -> None:
    base = TrainingExample(
        observation_hash="obs-a",
        information_policy="fair",
        policy_targets=(PolicyTarget("a", 1.0),),
        value_target=0.2,
        source_search_id="search-a",
        emulator_revision="emu",
        source_state_hash="state-a",
    )
    examples = (base,) + tuple(
        replace(base, source_state_hash=f"state-{i}", observation_hash=f"obs-{i}")
        for i in range(8)
    ) + (replace(base, source_search_id="other-budget"),)
    train, validation = split_training_examples(examples, seed=7)
    train_states = {e.source_state_hash for e in train}
    validation_states = {e.source_state_hash for e in validation}
    assert train_states.isdisjoint(validation_states)
    assert len(train) + len(validation) == len(examples)
    reverse_train, reverse_validation = split_training_examples(tuple(reversed(examples)), seed=7)
    assert train == reverse_train
    assert validation == reverse_validation
    same_state_train = {
        e.source_search_id for e in train if e.source_state_hash == "state-a"
    }
    same_state_validation = {
        e.source_search_id for e in validation if e.source_state_hash == "state-a"
    }
    both_budgets = {"search-a", "other-budget"}
    assert (same_state_train, same_state_validation) in (
        (both_budgets, set()),
        (set(), both_budgets),
    )


def test_grouped_split_rejects_single_state() -> None:
    example = TrainingExample(
        observation_hash="obs",
        information_policy="fair",
        policy_targets=(PolicyTarget("a", 1.0),),
        value_target=0.0,
        source_search_id="search",
        emulator_revision="emu",
        source_state_hash="one-state",
    )
    with pytest.raises(ValueError, match="distinct states"):
        split_training_examples((example, example))



def test_split_groups_hidden_states_with_same_fair_observation() -> None:
    one = TrainingExample(
        observation_hash="same-fair-observation",
        information_policy="fair",
        policy_targets=(PolicyTarget("a", 1.0),),
        value_target=0.3,
        source_search_id="sample-a",
        emulator_revision="emu",
        source_state_hash="hidden-state-1",
    )
    two = replace(
        one,
        source_search_id="sample-b",
        source_state_hash="hidden-state-2",
    )
    other = replace(
        one,
        observation_hash="different-observation",
        source_state_hash="hidden-state-3",
        source_search_id="sample-c",
    )
    train, validation = split_training_examples((one, two, other), seed=11)
    assert (one in train) == (two in train)
    assert (one in validation) == (two in validation)
    assert len(train) + len(validation) == 3


def test_split_groups_transitive_observation_overlap() -> None:
    one = TrainingExample(
        observation_hash="fair-a",
        information_policy="fair",
        policy_targets=(PolicyTarget("a", 1.0),),
        value_target=0.0,
        source_search_id="sample-a",
        emulator_revision="emu",
        source_state_hash="hidden-state-1",
    )
    connected = (
        one,
        replace(one, observation_hash="fair-b", source_search_id="sample-b"),
        replace(
            one,
            observation_hash="fair-b",
            source_state_hash="hidden-state-2",
            source_search_id="sample-c",
        ),
    )
    separate = replace(
        one,
        observation_hash="other",
        source_state_hash="hidden-state-3",
        source_search_id="sample-d",
    )
    train, validation = split_training_examples((*connected, separate), seed=2)
    assert all(e in train for e in connected) or all(e in validation for e in connected)
