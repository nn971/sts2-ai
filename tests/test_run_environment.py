"""Native-structure training must never quietly fall back to legacy maps."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2_ai.emulator import JsonlBridgeError, Observation
from sts2_ai.emulator.run_environment import (
    LEGACY,
    NATIVE_MAP_PROFILE,
    NATIVE_OVERGROWTH,
    NATIVE_RESET_SCHEMA,
    cumulative_floor_progress,
    require_environment,
    reset_training_run,
)
from sts2_ai.training.selfplay import Episode, _normalized_progress, train_selfplay


class FakeRunBackend:
    native_overgrowth_reset_schema: str | None = None

    def __init__(self, *, valid: bool = True) -> None:
        self.native_calls = 0
        self.legacy_calls = 0
        self.released: list[str] = []
        self.valid = valid

    def reset(self, seed: str, ascension: int = 0) -> str:
        self.legacy_calls += 1
        return "legacy-state"

    def reset_native_overgrowth(self, seed: str, ascension: int = 0) -> str:
        self.native_calls += 1
        return "native-state"

    def observe(self, state: str, policy: object) -> Observation:
        payload = {
            "map_generation_profile_id": NATIVE_MAP_PROFILE if self.valid else "legacy",
            "act": 1,
            "floor": 0,
            "event_id": "proto.native.event.neow",
            "map": [
                {"node_id": "1:1:0", "floor": 1},
                {"node_id": "1:16:3", "floor": 16},
            ],
            "deck": [{"card_id": "proto.silent.strike"}] * 12
            + [{"card_id": "proto.common.restlessness"}],
        }
        return Observation(
            "prototype-fair-v0", json.dumps(payload), "mock-public-hash"
        )

    def release_many(self, states: list[str]) -> int:
        self.released.extend(states)
        return len(states)


def test_pinned_bridge_absence_fails_without_legacy_reset() -> None:
    backend = FakeRunBackend()
    with pytest.raises(JsonlBridgeError, match="does not advertise"):
        reset_training_run(backend, "seed-1", NATIVE_OVERGROWTH)
    assert backend.legacy_calls == backend.native_calls == 0


def test_declared_native_capability_validates_visible_map_starter_and_neow() -> None:
    backend = FakeRunBackend()
    backend.native_overgrowth_reset_schema = NATIVE_RESET_SCHEMA
    assert reset_training_run(backend, "seed-1", NATIVE_OVERGROWTH) == "native-state"
    assert backend.native_calls == 1
    assert backend.legacy_calls == 0


def test_mislabeled_native_reset_fails_and_releases_state() -> None:
    backend = FakeRunBackend(valid=False)
    backend.native_overgrowth_reset_schema = NATIVE_RESET_SCHEMA
    with pytest.raises(JsonlBridgeError, match="failed public checks"):
        reset_training_run(backend, "seed-1", NATIVE_OVERGROWTH)
    assert backend.released == ["native-state"]


def test_legacy_mode_remains_explicitly_selectable() -> None:
    backend = FakeRunBackend()
    assert reset_training_run(backend, "seed-1", LEGACY) == "legacy-state"
    assert backend.legacy_calls == 1
    assert backend.native_calls == 0


def test_progress_normalization_respects_native_act_one_16_floors() -> None:
    assert cumulative_floor_progress(1, 16, LEGACY) == (6.0, 18.0)
    assert cumulative_floor_progress(1, 16, NATIVE_OVERGROWTH) == (16.0, 28.0)
    assert cumulative_floor_progress(2, 3, NATIVE_OVERGROWTH) == (19.0, 28.0)
    assert cumulative_floor_progress(3, 6, NATIVE_OVERGROWTH) == (28.0, 28.0)
    native = Episode(
        "s", (), "defeat", 1, 16, 0.0, True, NATIVE_OVERGROWTH
    )
    assert _normalized_progress(native) == pytest.approx(16 / 28)


def test_training_refuses_native_mode_before_torch_or_checkpoint(
    tmp_path: Path,
) -> None:
    backend = FakeRunBackend()
    checkpoint = tmp_path / "checkpoint.pt"
    with pytest.raises(JsonlBridgeError, match="does not advertise"):
        train_selfplay(
            backend, environment=NATIVE_OVERGROWTH,
            checkpoint_path=checkpoint, rounds=1, episodes_per_round=1,
        )
    assert not checkpoint.exists()
    assert backend.legacy_calls == backend.native_calls == 0


def test_mode_argument_validation() -> None:
    backend = FakeRunBackend()
    with pytest.raises(ValueError, match="Unknown training environment"):
        require_environment(backend, "invented-mode")
