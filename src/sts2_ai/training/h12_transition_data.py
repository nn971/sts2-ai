"""H12 public-only run traces for policy-conditioned combat transitions.

A combat begins at its first *published* combat frame and ends at the first
published noncombat frame, or an observed terminal defeat. All persistent
resources come from fair-player observations. Censored transitions are never
given survival/death labels.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sts2_ai.training.h13_public_transitions import (
    CONTEXT_VERSION,
    public_choice_context,
)

SCHEMA = "sts2-h12-public-joint-transition-v1"


def _id_list(state: dict[str, Any], field: str, key: str) -> list[str]:
    values = state.get(field)
    if not isinstance(values, list):
        return []
    return sorted(
        value if isinstance(value, str) else value.get(key)
        for value in values
        if isinstance(value, str) or (
            isinstance(value, dict) and isinstance(value.get(key), str)
        )
    )


def _potions(state: dict[str, Any]) -> list[dict[str, Any]]:
    raw = state.get("potions")
    if not isinstance(raw, list):
        return []
    potions = []
    for slot_index, item in enumerate(raw):
        if not isinstance(item, dict) or not isinstance(item.get("potion_id"), str):
            continue
        slot = item.get("slot", slot_index)
        if type(slot) is not int:
            continue
        potions.append({"slot": slot, "potion_id": item["potion_id"]})
    return sorted(potions, key=lambda p: p["slot"])


def _deck(state: dict[str, Any]) -> list[dict[str, Any]]:
    raw = state.get("deck")
    if not isinstance(raw, list):
        return []
    deck = []
    for card in raw:
        if not isinstance(card, dict) or not isinstance(card.get("card_id"), str):
            continue
        level = card.get("upgrade_level", card.get("upgrades", 0))
        deck.append({
            "card_id": card["card_id"],
            "upgrade_level": level if type(level) is int else 0,
        })
    return sorted(deck, key=lambda c: (c["card_id"], c["upgrade_level"]))


def public_resources(state: dict[str, Any], *, include_combat: bool = False
                     ) -> dict[str, Any]:
    """Allowlist public features; exclude run seed, RNG, hashes and internal IDs."""
    result = {
        key: state.get(key) for key in ("act", "floor", "hp", "max_hp", "gold")
    }
    result["deck"] = _deck(state)
    result["relics"] = _id_list(state, "relics", "relic_id")
    result["potions"] = _potions(state)
    # Four future public map layers. Aggregate types, not ephemeral node IDs;
    # hidden room outcomes remain hidden.
    nodes = state.get("map")
    if isinstance(nodes, list) and type(state.get("floor")) is int:
        floor = state["floor"]
        projected: dict[str, int] = {}
        for node in nodes:
            if not isinstance(node, dict) or type(node.get("floor")) is not int:
                continue
            depth = node["floor"] - floor
            if 1 <= depth <= 4:
                key = f"d{depth}:{node.get('room_type', 'unknown')}"
                projected[key] = projected.get(key, 0) + 1
        result["public_map_future_room_counts"] = projected
    if include_combat and isinstance(state.get("combat"), dict):
        combat = state["combat"]
        enemies = []
        for enemy in combat.get("enemies", []):
            if not isinstance(enemy, dict):
                continue
            allowed = {
                name: enemy.get(name) for name in (
                    "enemy_id", "hp", "block", "move_id",
                    "intent_damage", "intent_hits", "intent_base_damage",
                )
            }
            statuses = enemy.get("statuses")
            if isinstance(statuses, dict):
                allowed["statuses"] = {
                    str(k): int(v) for k, v in statuses.items()
                    if isinstance(k, str) and type(v) is int
                }
            powers = enemy.get("powers")
            if isinstance(powers, list):
                allowed["powers"] = [
                    {"power_id": p.get("power_id"), "stacks": p.get("stacks")}
                    for p in powers if isinstance(p, dict)
                ]
            enemies.append(allowed)
        result["combat"] = {
            "turn": combat.get("turn"),
            "energy": combat.get("energy"),
            "player_block": combat.get("player_block"),
            "draw_pile_count": combat.get("draw_pile_count"),
            "enemies": enemies,
            "hand": _deck({"deck": combat.get("hand")}),
            "player_powers": [
                {"power_id": p.get("power_id"), "stacks": p.get("stacks")}
                for p in (combat.get("player_powers") or [])
                if isinstance(p, dict)
            ],
        }
    return result


def _named_potion_transition(
    before: list[dict[str, Any]], after: list[dict[str, Any]],
) -> dict[str, Any]:
    # Slots can be replaced or shifted, so compare *named multisets*, not
    # ephemeral instances. Retained repeated potions remain multiplicity-aware.
    b = Counter(p["potion_id"] for p in before)
    a = Counter(p["potion_id"] for p in after)
    return {
        "retained": sorted((b & a).elements()),
        "consumed": sorted((b - a).elements()),
        "gained": sorted((a - b).elements()),
    }


@dataclass
class PublicJointRunTrace:
    source_seed: str
    combats: list[dict[str, Any]] = field(default_factory=list)
    macro_states: list[dict[str, Any]] = field(default_factory=list)
    _pending: dict[str, Any] | None = field(default=None, repr=False)
    _last: dict[str, Any] | None = field(default=None, repr=False)
    _frame: int = 0

    def _close(self, state: dict[str, Any], *, survived: bool) -> None:
        entry = self._pending
        if entry is None:
            raise ValueError("H12 combat close without a public entry")
        exit_frame = public_resources(state)
        before = entry["entry"]
        self.combats.append({
            "combat_index": len(self.combats),
            "entry_observation_index": entry["observation_index"],
            "exit_observation_index": self._frame,
            "entry": before,
            "exit": exit_frame,
            "survived": survived,
            "potion_transition": _named_potion_transition(
                before["potions"], exit_frame["potions"],
            ),
            "hp_delta": exit_frame["hp"] - before["hp"],
            "gold_delta": exit_frame["gold"] - before["gold"],
            "deck_delta": {
                "before": before["deck"], "after": exit_frame["deck"],
            },
            "relic_delta": {
                "before": before["relics"], "after": exit_frame["relics"],
            },
        })
        self._pending = None

    def observe(self, state: dict[str, Any]) -> None:
        """Passive, synchronous observer; invoked for *every* public frame."""
        if not isinstance(state, dict):
            raise ValueError("H12 requires fair public dictionary")
        self._last = state
        in_combat = isinstance(state.get("combat"), dict)
        if in_combat and self._pending is None:
            self._pending = {
                "observation_index": self._frame,
                "entry": public_resources(state, include_combat=True),
            }
        elif not in_combat and self._pending is not None:
            self._close(state, survived=type(state.get("hp")) is int
                        and state["hp"] > 0)
        if (in_combat and self._pending is not None
                and state.get("terminal_outcome") == "defeat"):
            self._close(state, survived=False)
        if not in_combat and state.get("terminal_outcome") is None:
            self.macro_states.append({
                "observation_index": self._frame,
                "state": public_resources(state),
            })
        self._frame += 1

    def record_decision(self, observation, legal_actions, decision) -> None:
        """Record public macro choices for subsequent trajectory diagnostics."""
        if not self.macro_states:
            return
        index = self._frame - 1
        last = self.macro_states[-1]
        if last["observation_index"] == index:
            kind = decision.action.kind
            payload = json.loads(decision.action.payload_json)
            selected = {
                "kind": kind,
                "payload": payload,
                "legal_action_kinds": sorted(set(a.kind for a in legal_actions)),
            }
            if kind == "choose_map_node" or "reward" in kind:
                raw = json.loads(observation.payload_json)
                selected.update(public_choice_context(raw, kind, payload))
            last["selected_action"] = selected

    def finish(self, *, outcome: str, censored: bool,
               act1_clear: bool, frontier_progress: float,
               decisions: int, policy_identity: dict[str, str]
               ) -> dict[str, Any]:
        if self._pending is not None:
            if not censored and outcome == "defeat" and self._last is not None:
                self._close(self._last, survived=False)
            else:
                # Never invent the exit of a truncated/suspended combat.
                self._pending = None
        if not censored and outcome not in ("defeat", "victory"):
            raise ValueError("Unexpected completed H12 episode status")
        for entry in self.combats:
            entry["source_run_seed"] = self.source_seed
            entry["policy_identity"] = policy_identity
        target = (1.0 if act1_clear else
                  0.1 * max(0.0, min(1.0, frontier_progress / 16.0)))
        return {
            "schema": SCHEMA,
            "public_choice_context_version": CONTEXT_VERSION,
            "source_run_seed": self.source_seed,
            "policy_identity": policy_identity,
            "outcome": outcome,
            "censored": censored,
            "act1_clear": act1_clear,
            "frontier_progress": frontier_progress,
            "decisions": decisions,
            "combats": self.combats,
            "macro_states": [] if censored else [
                {**item, "continuation_target": target}
                for item in self.macro_states
            ],
        }
