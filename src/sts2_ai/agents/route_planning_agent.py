from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, cast

from sts2_ai.agents.base import Decision
from sts2_ai.agents.heuristic_agent import HeuristicAgent
from sts2_ai.emulator import LegalAction, Observation

# The numerical room types are part of the versioned prototype-ai-v0 observation.
_COMBAT = 0
_ELITE = 1
_EVENT = 2
_SHOP = 3
_REST = 4
_BOSS = 5


@dataclass(frozen=True, slots=True)
class RouteEstimate:
    """Best visible path from a candidate map node under a disclosed proxy."""

    score: float
    node_ids: tuple[str, ...]


class RoutePlanner:
    """Deterministic lookahead on the *visible* room DAG; no emulator forks.

    This is a planning baseline, not a predictor of random encounters, events,
    hidden RNG, or eventual win probability. Node values are intentionally
    simple resource-sensitive proxies and should be measured, not trusted.
    """

    def __init__(self, *, horizon: int = 6, discount: float = 0.8) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must lie in (0, 1]")
        self.horizon = horizon
        self.discount = discount

    def evaluate(
        self, observation: Mapping[str, Any], candidate_node_ids: Sequence[str]
    ) -> dict[str, RouteEstimate]:
        raw_map = observation.get("map")
        if not isinstance(raw_map, list):
            return {}

        nodes: dict[str, tuple[int, int, tuple[str, ...]]] = {}
        for entry in raw_map:
            if not isinstance(entry, dict):
                continue
            node_id = entry.get("node_id")
            room_type = entry.get("room_type")
            floor = entry.get("floor")
            edges = entry.get("next_node_ids")
            if (
                not isinstance(node_id, str)
                or type(room_type) is not int
                or type(floor) is not int
                or not isinstance(edges, list)
                or not all(isinstance(child, str) for child in edges)
            ):
                continue
            nodes[node_id] = (room_type, floor, tuple(edges))

        hp = _numeric(observation.get("hp"), 1.0)
        max_hp = max(1.0, _numeric(observation.get("max_hp"), 1.0))
        hp_fraction = max(0.0, min(1.0, hp / max_hp))
        gold = _numeric(observation.get("gold"), 0.0)
        act = observation.get("act")
        history = observation.get("completed_rooms")
        current_act_rooms: list[int] = []
        if isinstance(history, list):
            for item in history:
                if (
                    isinstance(item, dict)
                    and item.get("act") == act
                    and type(item.get("room_type")) is int
                ):
                    current_act_rooms.append(item["room_type"])

        ordinary_combat_count = current_act_rooms.count(_COMBAT)
        streak = 0
        for room_type in reversed(current_act_rooms):
            if room_type != _COMBAT:
                break
            streak += 1

        relics = observation.get("relics")
        has_trail_ledger = (
            isinstance(relics, list)
            and any(
                isinstance(relic, dict)
                and relic.get("relic_id") == "proto.relic.trail_ledger"
                for relic in relics
            )
        )

        @cache
        def best(
            node_id: str, remaining: int, combat_count: int, combat_streak: int
        ) -> RouteEstimate | None:
            if remaining <= 0 or node_id not in nodes:
                return None
            room_type, floor, next_ids = nodes[node_id]
            local = _room_value(room_type, hp_fraction, gold)
            next_count = combat_count + int(room_type == _COMBAT)
            next_streak = combat_streak + 1 if room_type == _COMBAT else 0
            if has_trail_ledger and room_type == _COMBAT and next_count % 2 == 0:
                # A small, explicitly approximate marginal value of +20 gold.
                local += 1.0

            result = RouteEstimate(local, (node_id,))
            for child_id in next_ids:
                child = nodes.get(child_id)
                # Only reason across verified forward map edges. This prevents
                # malformed observations or cycles from contaminating search.
                if child is None or child[1] <= floor:
                    continue
                suffix = best(child_id, remaining - 1, next_count, next_streak)
                if suffix is None:
                    continue
                combined = RouteEstimate(
                    local + self.discount * suffix.score,
                    (node_id, *suffix.node_ids),
                )
                if (combined.score, combined.node_ids) > (
                    result.score, result.node_ids
                ):
                    result = combined
            return result

        return {
            candidate: estimate
            for candidate in candidate_node_ids
            if (estimate := best(
                candidate, self.horizon, ordinary_combat_count, streak
            )) is not None
        }


class RoutePlanningAgent:
    """Public-information route lookahead; unchanged v2 heuristic elsewhere.

    It is intentionally separate from HeuristicAgent so comparisons preserve
    the previous rollout baseline and can measure the map planner independently.
    """

    policy_id = "route-planner-v1-visible-dag"

    def __init__(
        self, *, horizon: int = 6, discount: float = 0.8
    ) -> None:
        self._planner = RoutePlanner(horizon=horizon, discount=discount)
        self._fallback = HeuristicAgent()
        self.policy_id = (
            f"{type(self).policy_id}"
            f"|horizon={horizon}|discount={discount:.6g}"
            f"|fallback={self._fallback.policy_id}"
        )

    def choose(
        self, observation: Observation, legal_actions: Sequence[LegalAction]
    ) -> Decision:
        if not legal_actions:
            raise ValueError("Cannot choose from an empty legal-action set")
        fallback = self._fallback.choose(observation, legal_actions)
        if not all(action.kind == "choose_map_node" for action in legal_actions):
            return Decision(fallback.action, self.policy_id)

        try:
            loaded = json.loads(observation.payload_json)
        except json.JSONDecodeError:
            return Decision(fallback.action, self.policy_id)
        if not isinstance(loaded, dict):
            return Decision(fallback.action, self.policy_id)
        state = cast(dict[str, Any], loaded)

        candidates: dict[str, LegalAction] = {}
        for action in legal_actions:
            try:
                payload = json.loads(action.payload_json)
            except json.JSONDecodeError:
                return Decision(fallback.action, self.policy_id)
            if not isinstance(payload, dict):
                return Decision(fallback.action, self.policy_id)
            node_id = payload.get("NodeId", payload.get("node_id"))
            if not isinstance(node_id, str) or node_id in candidates:
                return Decision(fallback.action, self.policy_id)
            candidates[node_id] = action

        estimates = self._planner.evaluate(state, tuple(candidates))
        if len(estimates) != len(candidates):
            return Decision(fallback.action, self.policy_id)

        # Deterministic tie-breaking by semantic action ID for reproducibility.
        chosen_id = max(
            candidates,
            key=lambda node_id: (
                estimates[node_id].score,
                candidates[node_id].action_id,
            ),
        )
        chosen = candidates[chosen_id]
        metadata = {
            "route_score": estimates[chosen_id].score,
            "route_node_ids": estimates[chosen_id].node_ids,
            "route_horizon": self._planner.horizon,
        }
        return Decision(
            chosen, self.policy_id,
            metadata_json=json.dumps(metadata, separators=(",", ":"), sort_keys=True)
        )


def _numeric(value: object, default: float) -> float:
    return float(value) if type(value) in (int, float) else default


def _room_value(room_type: int, hp_fraction: float, gold: float) -> float:
    if room_type == _COMBAT:
        return 0.5 if hp_fraction >= 0.4 else -0.7
    if room_type == _ELITE:
        if hp_fraction < 0.45:
            return -3.0
        return 2.2 if hp_fraction >= 0.75 else -0.4
    if room_type == _EVENT:
        return 1.4
    if room_type == _SHOP:
        return 1.5 if gold >= 100 else 0.5 if gold >= 50 else 0.1
    if room_type == _REST:
        return 0.2 + 4.0 * (1.0 - hp_fraction)
    if room_type == _BOSS:
        return 0.0
    return 0.0
