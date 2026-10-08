from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from sts2_ai.emulator import LegalAction, Observation

from .protocol import PolicyValueEstimate

_MODEL_FORMAT = "hashed-linear-policy-value-v2-unit-features"
_SKIP_STATE_KEYS = {
    "canonical_state_hash",
    "current_map_node_id",
    "decision_index",
    "instance_id",
    "node_id",
    "ruleset_id",
}


@dataclass(slots=True)
class HashedLinearPolicyValueModel:
    """Tiny dependency-free policy/value baseline with feature hashing.

    This is intentionally a first learning baseline rather than the intended final
    architecture. It provides genuine state/action generalization while keeping the
    training and inference path transparent and cheap enough for CI fixtures.
    """

    dimension: int
    policy_weights: list[float]
    value_weights: list[float]
    model_id: str = "hashed-linear-v2"

    @classmethod
    def zeros(
        cls,
        dimension: int = 4096,
        *,
        model_id: str | None = None,
    ) -> HashedLinearPolicyValueModel:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        return cls(
            dimension=dimension,
            policy_weights=[0.0] * dimension,
            value_weights=[0.0] * dimension,
            model_id=model_id or f"hashed-linear-v2-d{dimension}",
        )

    def evaluate(
        self,
        observation: Observation,
        legal_actions: Sequence[LegalAction],
    ) -> PolicyValueEstimate:
        state = _state_dict(observation.payload_json)
        logits = tuple(
            _dot(
                self.policy_weights,
                policy_features(
                    state,
                    action.kind,
                    action.payload_json,
                    self.dimension,
                ),
            )
            for action in legal_actions
        )
        value_raw = _dot(
            self.value_weights,
            state_features(state, self.dimension),
        )
        return PolicyValueEstimate(
            action_logits=logits,
            value=math.tanh(value_raw),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": _MODEL_FORMAT,
            "dimension": self.dimension,
            "model_id": self.model_id,
            "policy_weights": self.policy_weights,
            "value_weights": self.value_weights,
        }
        path.write_text(
            json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> HashedLinearPolicyValueModel:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("format") != _MODEL_FORMAT:
            raise ValueError("Unsupported hashed-linear model format")
        dimension = raw.get("dimension")
        policy_weights = raw.get("policy_weights")
        value_weights = raw.get("value_weights")
        model_id = raw.get("model_id")
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Model dimension is malformed")
        if not isinstance(policy_weights, list) or not isinstance(value_weights, list):
            raise ValueError("Model weights are malformed")
        if len(policy_weights) != dimension or len(value_weights) != dimension:
            raise ValueError("Model weight dimensions do not match")
        if not isinstance(model_id, str):
            raise ValueError("Model id is malformed")
        return cls(
            dimension=dimension,
            policy_weights=[float(value) for value in policy_weights],
            value_weights=[float(value) for value in value_weights],
            model_id=model_id,
        )


def state_features(
    state: dict[str, Any],
    dimension: int,
) -> dict[int, float]:
    tokens: list[tuple[str, float]] = [("state:bias", 1.0)]
    _flatten_state(tokens, "", state)
    return _hash_features(tokens, dimension)


def policy_features(
    state: dict[str, Any],
    action_kind: str,
    action_payload_json: str,
    dimension: int,
) -> dict[int, float]:
    state_tokens: list[tuple[str, float]] = [("state:bias", 1.0)]
    _flatten_state(state_tokens, "", state)
    semantic = _semantic_action_label(
        state,
        action_kind,
        action_payload_json,
    )
    tokens: list[tuple[str, float]] = [
        ("policy:bias", 1.0),
        (f"action-kind:{action_kind}", 1.0),
        (f"action-semantic:{semantic}", 1.0),
    ]
    for token, value in state_tokens:
        tokens.append((f"kind-cross:{action_kind}|{token}", value))
        tokens.append((f"semantic-cross:{semantic}|{token}", value))
    return _hash_features(tokens, dimension)


def state_dict(payload_json: str) -> dict[str, Any]:
    return _state_dict(payload_json)


def dot(weights: list[float], features: dict[int, float]) -> float:
    return _dot(weights, features)


def _state_dict(payload_json: str) -> dict[str, Any]:
    raw = json.loads(payload_json)
    return cast(dict[str, Any], raw if isinstance(raw, dict) else {})


def _flatten_state(
    output: list[tuple[str, float]],
    path: str,
    value: object,
) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            if key in _SKIP_STATE_KEYS:
                continue
            child_path = f"{path}.{key}" if path else str(key)
            _flatten_state(output, child_path, value[key])
        return

    if isinstance(value, list):
        output.append((f"len:{path}", min(4.0, len(value) / 5.0)))
        for item in value:
            _flatten_state(output, f"{path}[]", item)
        return

    if isinstance(value, bool):
        output.append((f"cat:{path}={str(value).lower()}", 1.0))
        return

    if isinstance(value, str):
        output.append((f"cat:{path}={value}", 1.0))
        return

    if isinstance(value, int | float):
        numeric = float(value)
        if abs(numeric) <= 12 and numeric.is_integer():
            output.append((f"cat:{path}={int(numeric)}", 1.0))
        else:
            bucket = int(math.floor(numeric / 5.0))
            output.append((f"bucket5:{path}={bucket}", 1.0))
        output.append((f"num:{path}", math.tanh(numeric / 50.0)))
        return

    if value is None:
        output.append((f"cat:{path}=null", 1.0))


def _semantic_action_label(
    state: dict[str, Any],
    action_kind: str,
    payload_json: str,
) -> str:
    payload = _action_payload(payload_json)
    card_by_instance = _card_ids_by_instance(state)
    enemy_by_instance = _enemy_ids_by_instance(state)

    card_instance = _int_field(
        payload,
        "CardInstanceId",
        "card_instance_id",
    )
    target_enemy = _int_field(
        payload,
        "TargetEnemyId",
        "target_enemy_id",
    )

    if action_kind in {"play_card", "rest_upgrade", "remove_card"}:
        card_id = (
            card_by_instance.get(card_instance, "unknown-card")
            if card_instance is not None
            else "unknown-card"
        )
        if action_kind == "play_card" and target_enemy is not None:
            enemy_id = enemy_by_instance.get(target_enemy, "unknown-enemy")
            return f"{action_kind}:{card_id}->{enemy_id}"
        return f"{action_kind}:{card_id}"

    if action_kind == "select_cards":
        raw_ids = payload.get(
            "CardInstanceIds",
            payload.get("card_instance_ids"),
        )
        if isinstance(raw_ids, list):
            cards = sorted(
                card_by_instance.get(item, "unknown-card")
                for item in raw_ids
                if isinstance(item, int)
            )
            return f"{action_kind}:{','.join(cards)}"

    if action_kind == "take_reward_card":
        index = _int_field(payload, "Index", "index")
        reward = state.get("reward")
        if isinstance(reward, dict):
            options = reward.get("card_options")
            if (
                isinstance(options, list)
                and index is not None
                and 0 <= index < len(options)
                and isinstance(options[index], str)
            ):
                return f"{action_kind}:{options[index]}"

    if action_kind == "choose_map_node":
        node_id = payload.get("NodeId", payload.get("node_id"))
        if isinstance(node_id, str):
            for node in _dict_items(state.get("map")):
                if node.get("node_id") == node_id:
                    return f"{action_kind}:room={node.get('room_type')}"

    return action_kind


def _action_payload(payload_json: str) -> dict[str, Any]:
    try:
        raw = json.loads(payload_json)
    except json.JSONDecodeError:
        return {}
    return cast(dict[str, Any], raw if isinstance(raw, dict) else {})


def _int_field(
    payload: dict[str, Any],
    *keys: str,
) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int):
            return value
    return None


def _card_ids_by_instance(state: dict[str, Any]) -> dict[int, str]:
    result: dict[int, str] = {}

    def consume(value: object) -> None:
        for card in _dict_items(value):
            instance_id = card.get("instance_id")
            card_id = card.get("card_id")
            if isinstance(instance_id, int) and isinstance(card_id, str):
                result[instance_id] = card_id

    consume(state.get("deck"))
    combat = state.get("combat")
    if isinstance(combat, dict):
        consume(combat.get("hand"))
        consume(combat.get("discard_pile"))
        consume(combat.get("exhaust_pile"))
    return result


def _enemy_ids_by_instance(state: dict[str, Any]) -> dict[int, str]:
    result: dict[int, str] = {}
    combat = state.get("combat")
    if not isinstance(combat, dict):
        return result
    for enemy in _dict_items(combat.get("enemies")):
        instance_id = enemy.get("instance_id")
        enemy_id = enemy.get("enemy_id")
        if isinstance(instance_id, int) and isinstance(enemy_id, str):
            result[instance_id] = enemy_id
    return result


def _dict_items(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        cast(dict[str, Any], item)
        for item in value
        if isinstance(item, dict)
    )


def _hash_features(
    tokens: list[tuple[str, float]],
    dimension: int,
) -> dict[int, float]:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    result: dict[int, float] = {}
    for token, value in tokens:
        digest = hashlib.blake2b(
            token.encode("utf-8"),
            digest_size=8,
            person=b"sts2-ai",
        ).digest()
        index = int.from_bytes(digest, "little") % dimension
        result[index] = result.get(index, 0.0) + value
    # The old unnormalized vector made SGD updates scale with the size of a
    # visible map/deck; a single long observation could drive enormous logits.
    # Unit L2 norm makes the step size independent of observation length.
    norm = math.sqrt(sum(value * value for value in result.values()))
    if norm > 0.0:
        return {index: value / norm for index, value in result.items()}
    return result


def _dot(weights: list[float], features: dict[int, float]) -> float:
    return sum(weights[index] * value for index, value in features.items())
