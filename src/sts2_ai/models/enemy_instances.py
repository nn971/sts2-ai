"""Player-visible, instance-keyed enemy features for the v8 tactical actor.

IDs are lookup keys, never numeric neural inputs. Every enemy retains its
species, position, announced intent and owned effects before any aggregation.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .tactical_state import (
    DAMAGE_NUMERIC_COORDINATES,
    damage_tactical_state_features,
    public_resources_tactical_state_features,
)

ENEMY_INSTANCE_SCHEMA = "sts2-enemy-instance-public-v8"
_ENEMY_NUMERIC = 16


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value) if math.isfinite(float(value)) else 0.0


def _hashed(features: dict[int, float], token: str, weight: float, dimension: int) -> None:
    slot = _ENEMY_NUMERIC + (
        int.from_bytes(hashlib.sha256(("enemy-instance-v8:" + token).encode()).digest()[:8], "big")
        % (dimension - _ENEMY_NUMERIC)
    )
    features[slot] = max(-8.0, min(8.0, features.get(slot, 0.0) + weight))


def enemy_instance_vectors(
    state: dict[str, Any], dimension: int,
) -> dict[int, dict[int, float]]:
    """One vector per unique living enemy; preserves species and individual state.

    Formation and last_move are visible. Hidden AI automaton state and RNG are
    deliberately excluded. Requires a v3 emulator public observation.
    """
    if dimension <= DAMAGE_NUMERIC_COORDINATES:
        raise ValueError("Instance encoder requires dimension > 40")
    combat = state.get("combat")
    if not isinstance(combat, dict) or not isinstance(combat.get("enemies"), list):
        raise ValueError("Instance encoder requires combat enemies")
    output: dict[int, dict[int, float]] = {}
    for enemy in combat["enemies"]:
        if not isinstance(enemy, dict):
            raise ValueError("Malformed enemy record")
        instance_id, species, position = (
            enemy.get("instance_id"), enemy.get("enemy_id"),
            enemy.get("formation_position"),
        )
        if (type(instance_id) is not int or instance_id <= 0
                or not isinstance(species, str) or not species
                or type(position) is not int):
            raise ValueError("Enemy lacks public instance, species or position")
        if instance_id in output:
            raise ValueError("Duplicate enemy instance ID")
        hp = _finite(enemy.get("hp"))
        block = _finite(enemy.get("block"))
        incoming = _finite(enemy.get("intent_damage"))
        hits = _finite(enemy.get("intent_hits"))
        base = _finite(enemy.get("intent_base_damage"))
        features: dict[int, float] = {
            0: 1.0, 1: hp / 200, 2: block / 100, 3: position / 10,
            4: incoming / 100, 5: hits / 10, 6: incoming * hits / 200,
            7: base / 100, 8: (incoming - base) * hits / 200,
            9: float(hp <= 0),
        }
        _hashed(features, "species=" + species, 1.0, dimension)
        for field in ("move_id", "last_move_id"):
            value = enemy.get(field)
            if isinstance(value, str) and value:
                _hashed(features, f"{field}={value}", 1.0, dimension)
        for power in enemy.get("powers") or ():
            if not isinstance(power, dict) or not isinstance(power.get("power_id"), str):
                raise ValueError("Malformed enemy power")
            count = _finite(power.get("stacks"))
            prefix = "power=" + power["power_id"]
            _hashed(features, prefix, 1.0, dimension)
            _hashed(features, prefix + ":amount", max(-8.0, min(8.0, count / 5)), dimension)
            _hashed(features, prefix + f":exact={count}", 1.0, dimension)
        statuses = enemy.get("statuses")
        if not isinstance(statuses, dict):
            raise ValueError("Malformed enemy statuses")
        for status, count_raw in sorted(statuses.items()):
            if not isinstance(status, str):
                raise ValueError("Malformed enemy status identity")
            count = _finite(count_raw)
            prefix = "status=" + status
            _hashed(features, prefix, 1.0, dimension)
            _hashed(features, prefix + ":amount", max(-8.0, min(8.0, count / 5)), dimension)
            _hashed(features, prefix + f":exact={count}", 1.0, dimension)
        output[instance_id] = features
    return output


def instance_global_features(state: dict[str, Any], dimension: int) -> dict[int, float]:
    """Only player/deck resources enter global hashed categorical slots.

    Keep v6 numerical battlefield aggregates; per-enemy categorical details
    flow exclusively through the separate learned enemy encoder.
    """
    combat = state.get("combat")
    if not isinstance(combat, dict):
        raise ValueError("Instance global features require a combat frame")
    stripped = dict(state)
    stripped["combat"] = {**combat, "enemies": []}
    global_features = public_resources_tactical_state_features(stripped, dimension)
    aggregate = damage_tactical_state_features(state, dimension)
    for index in range(DAMAGE_NUMERIC_COORDINATES):
        if index in aggregate:
            global_features[index] = aggregate[index]
        else:
            global_features.pop(index, None)
    return global_features


def action_target_instance(action_payload_json: str, instances: dict[int, Any]) -> int | None:
    """Decode a legal target reference; reject dangling references, never hash IDs."""
    try:
        raw = json.loads(action_payload_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed action payload") from exc
    if not isinstance(raw, dict):
        raise ValueError("Malformed action payload")
    target = raw.get("TargetEnemyId", raw.get("target_enemy_id"))
    if target is None:
        return None
    if type(target) is not int or target not in instances:
        raise ValueError("Action target does not reference a visible enemy instance")
    return target


def instance_action_features(
    state: dict[str, Any], action_kind: str, payload_json: str, dimension: int,
) -> dict[int, float]:
    """Encode action/card identity only; the target is supplied as its own node."""
    from .hashed_linear import tactical_action_features
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed action payload") from exc
    if not isinstance(payload, dict):
        raise ValueError("Malformed action payload")
    untargeted = {
        key: value for key, value in payload.items()
        if key not in ("TargetEnemyId", "target_enemy_id")
    }
    return tactical_action_features(
        state, action_kind, json.dumps(untargeted, sort_keys=True), dimension,
    )
