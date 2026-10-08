"""Versioned training environments, with native Overgrowth strictly fail-closed.

The native-structure factory exists inside sts2-emulator, but its pinned JSONL
server does NOT expose it. No call to the legacy reset is a valid substitute.
This module describes the *requested* bridge contract without implementing or
altering any emulator operation.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

from .protocol import EmulatorBackend, InformationPolicy, Observation, StateHandle

LEGACY = "legacy-prototype"
NATIVE_OVERGROWTH = "native-overgrowth"
ENVIRONMENTS = (NATIVE_OVERGROWTH, LEGACY)

# Proposed AI/bridge handshake contract; not advertised by the pinned emulator.
NATIVE_RESET_SCHEMA = "prototype-native-overgrowth-reset-v1"
NATIVE_MAP_PROFILE = "native-overgrowth-map-structure-v0.111.0-v1"


def require_environment(backend: EmulatorBackend, environment: str) -> None:
    from .jsonl_backend import JsonlBridgeError

    if environment not in ENVIRONMENTS:
        raise ValueError(f"Unknown training environment: {environment!r}")
    if (
        environment == NATIVE_OVERGROWTH
        and getattr(backend, "native_overgrowth_reset_schema", None) != NATIVE_RESET_SCHEMA
    ):
        raise JsonlBridgeError(
            "Native-structure Overgrowth training is unavailable: the pinned "
            "emulator JSONL bridge does not advertise nativeOvergrowthResetId="
            f"{NATIVE_RESET_SCHEMA!r}. Its normal reset still generates the "
            "six-floor legacy prototype. No fallback or emulator modification "
            "is permitted. Request an explicit native-Overgrowth reset "
            "operation in sts2-emulator first."
        )


def validate_native_start(observation: Observation) -> None:
    """Verify visible evidence of real native-shaped initialization."""
    from .jsonl_backend import JsonlBridgeError

    raw = json.loads(observation.payload_json)
    if not isinstance(raw, dict):
        raise JsonlBridgeError("Native Overgrowth reset gave malformed public observation")
    map_nodes = raw.get("map")
    deck = raw.get("deck")
    if (
        raw.get("map_generation_profile_id") != NATIVE_MAP_PROFILE
        or raw.get("act") != 1
        or raw.get("floor") != 0
        or raw.get("event_id") != "proto.native.event.neow"
        or not isinstance(map_nodes, list)
        or 16 not in [node.get("floor") for node in map_nodes if isinstance(node, dict)]
        or not isinstance(deck, list)
        or sum(
            card.get("card_id") == "proto.common.restlessness"
            for card in deck if isinstance(card, dict)
        ) != 1
        or len(deck) != 13
    ):
        raise JsonlBridgeError(
            "Native Overgrowth reset failed public checks: expected Act 1 "
            "16-floor map, Neow opening event and 13-card Silent deck with "
            "Restlessness. Refusing to train on a substituted prototype."
        )


def reset_training_run(
    backend: RunEnvironmentBackend,
    seed: str,
    environment: str,
    ascension: int = 0,
) -> StateHandle:
    require_environment(backend, environment)
    if environment == LEGACY:
        return backend.reset(seed, ascension)
    native_reset = cast(
        Callable[[str, int], StateHandle], getattr(backend, "reset_native_overgrowth")
    )
    state = native_reset(seed, ascension)
    try:
        validate_native_start(
            backend.observe(state, InformationPolicy("prototype-fair-v0"))
        )
    except BaseException:
        backend.release_many([state])
        raise
    return state


def cumulative_floor_progress(
    act: int | None, floor: int | None, environment: str,
) -> tuple[float, float]:
    """Progress proxy and run maximum, in the appropriate act geometry.

    Native-structure starts on a 16-floor Overgrowth act. Later acts still
    use the current six-floor prototype; do not imply native Act 2/3 geometry.
    """
    if environment not in ENVIRONMENTS:
        raise ValueError(f"Unknown training environment: {environment!r}")
    a = max(1, min(3, act if isinstance(act, int) else 1))
    f = max(0, floor if isinstance(floor, int) else 0)
    if environment == LEGACY:
        return float((a - 1) * 6 + min(6, f)), 18.0
    first = 16
    if a == 1:
        return float(min(first, f)), 28.0
    return float(first + (a - 2) * 6 + min(6, f)), 28.0
