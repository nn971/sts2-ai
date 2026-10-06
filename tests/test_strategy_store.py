from pathlib import Path

from sts2_ai.strategy_db import (
    CachedActionEvaluation,
    SQLiteStrategyStore,
    StrategicEvidence,
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


def test_roundtrip_search_cache_isolated_by_configuration(tmp_path: Path) -> None:
    first = CachedActionEvaluation(
        state_hash="state-1",
        information_policy="prototype-fair-v0",
        action_id="action-a",
        value=1.25,
        visits=32,
        uncertainty=0.1,
        search_version="search-v2",
        search_config_id="nodes=32;depth=4;seed=7",
        model_id=None,
        emulator_revision="emu-1",
        game_build="prototype-unbound",
    )
    second = CachedActionEvaluation(
        state_hash="state-1",
        information_policy="prototype-fair-v0",
        action_id="action-b",
        value=0.75,
        visits=32,
        uncertainty=0.2,
        search_version="search-v2",
        search_config_id="nodes=32;depth=4;seed=7",
        model_id=None,
        emulator_revision="emu-1",
        game_build="prototype-unbound",
    )

    with SQLiteStrategyStore(tmp_path / "strategy.sqlite") as store:
        store.cache_action_evaluations((first, second))

        assert store.cached_action_evaluations(
            state_hash="state-1",
            information_policy="prototype-fair-v0",
            search_version="search-v2",
            search_config_id="nodes=32;depth=4;seed=7",
            emulator_revision="emu-1",
            game_build="prototype-unbound",
        ) == (first, second)

        assert store.cached_action_evaluations(
            state_hash="state-1",
            information_policy="prototype-fair-v0",
            search_version="search-v2",
            search_config_id="nodes=64;depth=4;seed=7",
            emulator_revision="emu-1",
            game_build="prototype-unbound",
        ) == ()
