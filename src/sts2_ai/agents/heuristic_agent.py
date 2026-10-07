from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, cast

from sts2_ai.agents.base import Decision
from sts2_ai.emulator import LegalAction, Observation


class HeuristicAgent:
    """Small, deliberately legible rollout policy.

    The policy uses only player-facing observation JSON and legal actions. It is meant
    to be a useful rollout baseline, not a hand-authored expert.
    """

    def choose(
        self,
        observation: Observation,
        legal_actions: Sequence[LegalAction],
    ) -> Decision:
        if not legal_actions:
            raise ValueError("Cannot choose from an empty legal-action set")

        raw = json.loads(observation.payload_json)
        state = cast(dict[str, Any], raw if isinstance(raw, dict) else {})
        scored = [
            (self._score(state, action), action.action_id, action)
            for action in legal_actions
        ]
        _, _, action = max(scored)
        return Decision(action=action, policy_name="heuristic-v2-payload-aware")

    def _score(self, state: dict[str, Any], action: LegalAction) -> float:
        kind = action.kind
        payload = self._payload(action)
        hp = self._number(state.get("hp"), 1.0)
        max_hp = max(1.0, self._number(state.get("max_hp"), 1.0))
        hp_ratio = hp / max_hp

        if kind in {"start_run", "continue_act", "leave_reward"}:
            return 100.0

        if kind == "choose_map_node":
            node_id = payload.get("node_id")
            room_type = None
            for node in self._list_of_dicts(state.get("map")):
                if node.get("node_id") == node_id:
                    room_type = node.get("room_type")
                    break
            return self._map_score(room_type, hp_ratio)

        if kind == "rest_heal":
            return 12.0 if hp_ratio < 0.60 else 1.0
        if kind == "rest_upgrade":
            return 8.0 if hp_ratio >= 0.45 else 2.0
        if kind == "rest_train":
            return 5.0 if hp_ratio >= 0.70 else 2.5

        if kind == "take_reward_relic":
            return 15.0
        if kind == "take_reward_potion":
            return 7.0
        if kind == "skip_reward_potion":
            return 1.0
        if kind == "take_reward_card":
            deck = state.get("deck")
            deck_size = len(deck) if isinstance(deck, list) else 0
            return 6.0 if deck_size < 24 else 2.0
        if kind == "skip_reward_card":
            deck = state.get("deck")
            deck_size = len(deck) if isinstance(deck, list) else 0
            return 4.0 if deck_size >= 24 else 1.0

        if kind == "buy_relic":
            return 10.0
        if kind == "remove_card":
            return 8.0
        if kind == "buy_card":
            return 6.0
        if kind == "buy_potion":
            return 5.0 if hp_ratio < 0.65 else 3.0
        if kind == "leave_shop":
            return 0.0

        if kind == "event_choice":
            return 1.0

        if kind == "select_cards":
            return 2.0

        if kind == "use_potion":
            return 7.0 if hp_ratio < 0.45 else 1.5

        if kind == "play_card":
            return self._combat_card_score(state, payload)

        if kind == "end_turn":
            combat = state.get("combat")
            if isinstance(combat, dict):
                energy = self._number(combat.get("energy"), 0.0)
                return -4.0 if energy > 0 else 0.0
            return -1.0

        return 0.0

    def _combat_card_score(
        self,
        state: dict[str, Any],
        payload: dict[str, Any],
    ) -> float:
        combat = state.get("combat")
        if not isinstance(combat, dict):
            return 3.0

        card_id = ""
        instance_id = payload.get("card_instance_id")
        for card in self._list_of_dicts(combat.get("hand")):
            if card.get("instance_id") == instance_id:
                value = card.get("card_id")
                if isinstance(value, str):
                    card_id = value.lower()
                break

        score = 4.0
        attack_words = (
            "strike",
            "stab",
            "dagger",
            "blade",
            "poison",
            "neutralize",
            "skewer",
            "finisher",
            "eviscerate",
        )
        defense_words = (
            "defend",
            "survivor",
            "backflip",
            "blur",
            "dodge",
            "cloak",
            "footwork",
        )
        if any(word in card_id for word in attack_words):
            score += 2.0
        if any(word in card_id for word in defense_words):
            score += 1.5

        target_id = payload.get("target_enemy_id")
        if target_id is not None:
            for enemy in self._list_of_dicts(combat.get("enemies")):
                if enemy.get("instance_id") == target_id:
                    enemy_hp = self._number(enemy.get("hp"), 999.0)
                    if enemy_hp <= 8:
                        score += 3.0
                    elif enemy_hp <= 16:
                        score += 1.0
                    break
        return score

    @staticmethod
    def _map_score(room_type: object, hp_ratio: float) -> float:
        # PrototypeRoomType: Combat=0, Elite=1, Event=2, Shop=3, Rest=4, Boss=5.
        if not isinstance(room_type, int):
            return 0.0
        if hp_ratio < 0.45:
            low_hp_scores: dict[int, float] = {
                4: 10.0,
                3: 7.0,
                2: 6.0,
                0: 3.0,
                1: 0.0,
                5: -1.0,
            }
            return low_hp_scores.get(room_type, 0.0)
        healthy_scores: dict[int, float] = {
            1: 8.0,
            4: 7.0,
            2: 6.0,
            3: 5.0,
            0: 4.0,
            5: 3.0,
        }
        return healthy_scores.get(room_type, 0.0)

    @staticmethod
    def _payload(action: LegalAction) -> dict[str, Any]:
        raw = json.loads(action.payload_json)
        if not isinstance(raw, dict):
            return {}
        return {
            _snake_case(str(key)): value
            for key, value in cast(dict[str, Any], raw).items()
        }

    @staticmethod
    def _list_of_dicts(value: object) -> tuple[dict[str, Any], ...]:
        if not isinstance(value, list):
            return ()
        return tuple(cast(dict[str, Any], item) for item in value if isinstance(item, dict))

    @staticmethod
    def _number(value: object, default: float) -> float:
        if isinstance(value, int | float):
            return float(value)
        return default


def _snake_case(name: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()
