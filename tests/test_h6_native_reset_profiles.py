"""Native Overgrowth reset permits only audited profile IDs, not prototypes."""
import json

import pytest

from sts2_ai.emulator import Observation
from sts2_ai.emulator.jsonl_backend import JsonlBridgeError
from sts2_ai.emulator.run_environment import (
    NATIVE_MAP_PROFILE,
    NATIVE_MAP_PROFILE_CONTEXT,
    validate_native_start,
)


def _native_frame(profile: str, ascension: int = 0) -> Observation:
    curse = ("proto.curse.ascenders_bane" if ascension >= 5
             else "proto.common.restlessness")
    return Observation(
        "prototype-fair-v0",
        json.dumps({
            "map_generation_profile_id": profile,
            "act": 1, "floor": 0,
            "event_id": "proto.native.event.neow",
            "map": [{"floor": 1}, {"floor": 9}, {"floor": 16}],
            "deck": ([{"card_id": curse}]
                     + [{"card_id": "proto.silent.strike"}] * 12),
        }),
        "",
    )


@pytest.mark.parametrize("profile", [
    NATIVE_MAP_PROFILE,
    NATIVE_MAP_PROFILE_CONTEXT,
])
def test_accepts_explicit_overgrowth_native_profiles(profile):
    validate_native_start(_native_frame(profile))


@pytest.mark.parametrize("profile", [
    "legacy-prototype-six-floor-v1",
    "unknown-native-looking-name",
    "",
])
def test_rejects_unrecognized_map_profiles(profile):
    with pytest.raises(JsonlBridgeError, match="Native Overgrowth reset failed"):
        validate_native_start(_native_frame(profile))


@pytest.mark.parametrize("ascension", [0, 4, 5, 10])
def test_native_ascension_starter_is_verified(ascension):
    frame = _native_frame(NATIVE_MAP_PROFILE_CONTEXT, ascension)
    validate_native_start(frame, ascension=ascension)
    with pytest.raises(JsonlBridgeError, match="13-card Silent deck"):
        validate_native_start(frame, ascension=(0 if ascension >= 5 else 10))
