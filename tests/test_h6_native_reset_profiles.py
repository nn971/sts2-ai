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


def _native_frame(profile: str) -> Observation:
    return Observation(
        "prototype-fair-v0",
        json.dumps({
            "map_generation_profile_id": profile,
            "act": 1, "floor": 0,
            "event_id": "proto.native.event.neow",
            "map": [{"floor": 1}, {"floor": 9}, {"floor": 16}],
            "deck": ([{"card_id": "proto.common.restlessness"}]
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
