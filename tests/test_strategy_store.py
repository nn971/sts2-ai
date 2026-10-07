from pathlib import Path

from sts2_ai.emulator import LegalAction
from sts2_ai.search import ActionEvaluation, SearchResult
from sts2_ai.strategy_db import (
    SQLiteStrategyStore,
    StrategicEvidence,
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
        root_observation_json='{"phase":"reward"}',
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
        observation = store.observation(
            "obs-search",
            "prototype-fair-v0",
        )

    assert observation is not None
    assert observation.payload_json == '{"phase":"reward"}'
    assert len(records) == 1
    root, stored_actions = records[0]
    assert root.observation_hash == "obs-search"
    assert root.legal_action_ids == ("a", "b")
    assert root.chosen_action_id == "a"
    assert root.search_budget == 32
    assert root.transitions == 200
    assert [item.action_id for item in stored_actions] == ["a", "b"]
    assert [item.visits for item in stored_actions] == [20, 12]
