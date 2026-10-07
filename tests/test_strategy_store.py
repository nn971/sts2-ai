from pathlib import Path

import pytest

from sts2_ai.emulator import LegalAction
from sts2_ai.search import ActionEvaluation, SearchResult
from sts2_ai.strategy_db import (
    SQLiteStrategyStore,
    StrategicEvidence,
    diagnose_budget_disagreements,
    record_search_result,
)


def test_roundtrip_strategy_evidence(tmp_path: Path) -> None:
    record = StrategicEvidence(
        state_hash="state-1",
        information_policy="fair-v1",
        action_id="take-card-a",
        value=0.61,
        visits=100,
        uncertainty=0.03,
        search_version="search-v1",
        model_id="model-7",
        emulator_revision="emu-abc",
        game_build="build-1",
    )

    with SQLiteStrategyStore(tmp_path / "strategy.sqlite") as store:
        store.upsert(record)
        assert store.for_state("state-1", "fair-v1") == (record,)


def test_roundtrip_search_root_and_action_statistics(tmp_path: Path) -> None:
    actions = (
        LegalAction("a", "take_reward_card"),
        LegalAction("b", "skip_reward_card"),
    )
    result = SearchResult(
        root_state_hash="state-search",
        root_observation_hash="obs-search",
        root_observation_json='{"phase":5,"act":1,"floor":2,"hp":42,"max_hp":70}',
        evaluations=(
            ActionEvaluation(actions[0], value=0.7, visits=20, uncertainty=0.1),
            ActionEvaluation(actions[1], value=0.2, visits=12, uncertainty=0.2),
        ),
        expanded_nodes=15,
        transitions=200,
        transposition_hits=4,
        search_version="oracle-exact-uct-v1",
    )

    with SQLiteStrategyStore(tmp_path / "nested" / "strategy.sqlite") as store:
        record_search_result(
            store,
            result,
            actions[0],
            information_policy="prototype-fair-v0",
            search_regime="oracle-exact",
            search_budget=32,
            emulator_revision="emu-1",
            game_build="build-1",
        )
        records = store.search_for_state(
            "state-search",
            "prototype-fair-v0",
        )

    assert len(records) == 1
    root, stored_actions = records[0]
    assert root.observation_hash == "obs-search"
    assert root.legal_action_ids == ("a", "b")
    assert root.chosen_action_id == "a"
    assert root.search_budget == 32
    assert root.transitions == 200
    assert [item.action_id for item in stored_actions] == ["a", "b"]
    assert [item.action_kind for item in stored_actions] == [
        "take_reward_card",
        "skip_reward_card",
    ]
    assert [item.action_payload_json for item in stored_actions] == ["{}", "{}"]
    assert [item.visits for item in stored_actions] == [20, 12]


def test_budget_disagreement_diagnostic_separates_value_flip_from_visit_choice(
    tmp_path: Path,
) -> None:
    first = LegalAction("play_card:a", "play_card")
    second = LegalAction("end_turn:b", "end_turn")
    observation = (
        '{"phase":3,"act":1,"floor":6,"hp":31,"max_hp":70,'
        '"combat":{"turn":4}}'
    )
    low = SearchResult(
        root_state_hash="shared-state",
        root_observation_hash="shared-observation",
        root_observation_json=observation,
        evaluations=(
            ActionEvaluation(first, value=0.30, visits=5, uncertainty=0.1),
            ActionEvaluation(second, value=0.25, visits=3, uncertainty=0.1),
        ),
        expanded_nodes=4,
        transitions=10,
        transposition_hits=0,
        search_version="search-v1",
    )
    high = SearchResult(
        root_state_hash="shared-state",
        root_observation_hash="shared-observation",
        root_observation_json=observation,
        evaluations=(
            ActionEvaluation(first, value=0.20, visits=12, uncertainty=0.05),
            ActionEvaluation(second, value=0.40, visits=20, uncertainty=0.05),
        ),
        expanded_nodes=8,
        transitions=30,
        transposition_hits=1,
        search_version="search-v1",
    )

    with SQLiteStrategyStore(tmp_path / "strategy.sqlite") as store:
        record_search_result(
            store,
            low,
            first,
            information_policy="prototype-fair-v0",
            search_regime="oracle-exact",
            search_budget=8,
            emulator_revision="emu-1",
            game_build="build-1",
        )
        record_search_result(
            store,
            high,
            second,
            information_policy="prototype-fair-v0",
            search_regime="oracle-exact",
            search_budget=32,
            emulator_revision="emu-1",
            game_build="build-1",
        )

        report = diagnose_budget_disagreements(
            store,
            "prototype-fair-v0",
        )

    assert report.total_roots == 1
    root = report.roots[0]
    assert root.phase == "Combat"
    assert root.classification == "value-ranking-flip"
    assert root.value_ranking_flip
    assert not root.has_visit_selection_mismatch
    assert root.low_budget_pair_delta == pytest.approx(0.05)
    assert root.high_budget_pair_delta == pytest.approx(-0.20)
    assert report.by_phase[0].value_ranking_flips == 1


def test_budget_disagreement_diagnostic_detects_visit_selection_regret(
    tmp_path: Path,
) -> None:
    first = LegalAction("choose_map_node:a", "choose_map_node")
    second = LegalAction("choose_map_node:b", "choose_map_node")
    observation = '{"phase":2,"act":1,"floor":3,"hp":60,"max_hp":70}'
    low = SearchResult(
        root_state_hash="shared-state",
        root_observation_hash="shared-observation",
        root_observation_json=observation,
        evaluations=(
            ActionEvaluation(first, value=0.40, visits=7, uncertainty=0.1),
            ActionEvaluation(second, value=0.35, visits=1, uncertainty=0.1),
        ),
        expanded_nodes=4,
        transitions=10,
        transposition_hits=0,
        search_version="search-v1",
    )
    high = SearchResult(
        root_state_hash="shared-state",
        root_observation_hash="shared-observation",
        root_observation_json=observation,
        evaluations=(
            ActionEvaluation(first, value=0.41, visits=14, uncertainty=0.05),
            ActionEvaluation(second, value=0.39, visits=18, uncertainty=0.05),
        ),
        expanded_nodes=8,
        transitions=30,
        transposition_hits=1,
        search_version="search-v1",
    )

    with SQLiteStrategyStore(tmp_path / "strategy.sqlite") as store:
        record_search_result(
            store,
            low,
            first,
            information_policy="prototype-fair-v0",
            search_regime="oracle-exact",
            search_budget=8,
            emulator_revision="emu-1",
            game_build="build-1",
        )
        record_search_result(
            store,
            high,
            second,
            information_policy="prototype-fair-v0",
            search_regime="oracle-exact",
            search_budget=32,
            emulator_revision="emu-1",
            game_build="build-1",
        )
        report = diagnose_budget_disagreements(
            store,
            "prototype-fair-v0",
        )

    root = report.roots[0]
    assert root.phase == "MapChoice"
    assert root.classification == "visit-selection"
    assert not root.value_ranking_flip
    assert root.has_visit_selection_mismatch
    assert root.decisions[-1].selection_regret == pytest.approx(0.02)


def test_budget_disagreement_diagnostic_flags_unvisited_selection(
    tmp_path: Path,
) -> None:
    visited = LegalAction("play_card:visited", "play_card")
    unvisited = LegalAction("play_card:unvisited", "play_card")
    observation = '{"phase":3,"act":1,"floor":1,"hp":70,"max_hp":70}'
    low = SearchResult(
        root_state_hash="shared-state",
        root_observation_hash="shared-observation",
        root_observation_json=observation,
        evaluations=(
            ActionEvaluation(visited, value=-0.4, visits=8, uncertainty=0.1),
            ActionEvaluation(unvisited, value=0.0, visits=0, uncertainty=None),
        ),
        expanded_nodes=4,
        transitions=10,
        transposition_hits=0,
        search_version="search-v1",
    )
    high = SearchResult(
        root_state_hash="shared-state",
        root_observation_hash="shared-observation",
        root_observation_json=observation,
        evaluations=(
            ActionEvaluation(visited, value=-0.3, visits=10, uncertainty=0.1),
            ActionEvaluation(unvisited, value=-0.2, visits=12, uncertainty=0.1),
        ),
        expanded_nodes=8,
        transitions=30,
        transposition_hits=0,
        search_version="search-v1",
    )

    with SQLiteStrategyStore(tmp_path / "strategy.sqlite") as store:
        record_search_result(
            store,
            low,
            unvisited,
            information_policy="prototype-fair-v0",
            search_regime="oracle-exact",
            search_budget=8,
            emulator_revision="emu-1",
            game_build="build-1",
        )
        record_search_result(
            store,
            high,
            visited,
            information_policy="prototype-fair-v0",
            search_regime="oracle-exact",
            search_budget=32,
            emulator_revision="emu-1",
            game_build="build-1",
        )
        report = diagnose_budget_disagreements(
            store,
            "prototype-fair-v0",
        )
        health = store.root_selection_visit_counts("prototype-fair-v0")

    assert report.roots[0].classification == "unvisited-selection"
    assert report.by_phase[0].unvisited_selection == 1
    assert health == ((8, 1, 1), (32, 1, 0))


def test_diagnostic_best_mean_ignores_unvisited_placeholder(tmp_path: Path) -> None:
    visited = LegalAction("play_card:visited", "play_card")
    unvisited = LegalAction("play_card:unvisited", "play_card")
    other = LegalAction("play_card:other", "play_card")
    observation = '{"phase":3,"act":1,"floor":1,"hp":70,"max_hp":70}'
    low = SearchResult(
        root_state_hash="shared-state-diagnostic",
        root_observation_hash="shared-observation-diagnostic",
        root_observation_json=observation,
        evaluations=(
            ActionEvaluation(visited, value=-0.40, visits=8, uncertainty=0.1),
            ActionEvaluation(unvisited, value=0.0, visits=0, uncertainty=None),
            ActionEvaluation(other, value=-0.45, visits=1, uncertainty=0.2),
        ),
        expanded_nodes=4,
        transitions=10,
        transposition_hits=0,
        search_version="search-v1",
    )
    high = SearchResult(
        root_state_hash="shared-state-diagnostic",
        root_observation_hash="shared-observation-diagnostic",
        root_observation_json=observation,
        evaluations=(
            ActionEvaluation(visited, value=-0.35, visits=10, uncertainty=0.1),
            ActionEvaluation(unvisited, value=-0.30, visits=12, uncertainty=0.1),
            ActionEvaluation(other, value=-0.50, visits=10, uncertainty=0.1),
        ),
        expanded_nodes=8,
        transitions=30,
        transposition_hits=0,
        search_version="search-v1",
    )

    with SQLiteStrategyStore(tmp_path / "strategy.sqlite") as store:
        record_search_result(
            store,
            low,
            visited,
            information_policy="prototype-fair-v0",
            search_regime="oracle-exact",
            search_budget=8,
            emulator_revision="emu-1",
            game_build="build-1",
        )
        record_search_result(
            store,
            high,
            unvisited,
            information_policy="prototype-fair-v0",
            search_regime="oracle-exact",
            search_budget=32,
            emulator_revision="emu-1",
            game_build="build-1",
        )
        report = diagnose_budget_disagreements(store, "prototype-fair-v0")

    low_decision = report.roots[0].decisions[0]
    assert low_decision.best_mean_action_id == visited.action_id
    assert low_decision.selection_regret == 0.0
