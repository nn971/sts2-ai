from pathlib import Path

from sts2_ai.strategy_db import SQLiteStrategyStore, StrategicEvidence


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
