"""H14 public-state projection and observed combat-to-macro transitions.

Only published fair-player observations and committed player actions enter this
module. Run seeds and policy identities are provenance, never model features.
The first noncombat observation closes combat BEFORE any reward selection.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

STATE_SCHEMA = "sts2-h14-public-persistent-state-v1"
TRANSITION_SCHEMA = "sts2-h14-observed-combat-boundary-v1"

# Audited against PrototypeNativeSharedRelics and PrototypeCombat on 2026-10-29.
# An absent persistent property initializes to zero in CreateRelicTriggerCounts.
PERSISTENT_COUNTERS: dict[str, tuple[str, int]] = {
    "proto.relic.pendulum": ("TurnsSeen", 3),
    "proto.relic.pen_nib": ("AttacksPlayed", 10),
}


def _integer(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"Missing or invalid public {field}")
    return value


def _objects(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(x, dict) for x in value):
        raise ValueError(f"Missing or invalid public {field}")
    return value


def _identity(item: Mapping[str, Any], key: str) -> str:
    identity = item.get(key)
    if not isinstance(identity, str) or not identity:
        raise ValueError(f"Missing public {key}")
    return identity


def _visible_rewards(reward: Mapping[str, Any]) -> dict[str, Any]:
    """Published offers only: no hidden offers, generation seed or pickup."""
    result: dict[str, Any] = {}
    for key in ("card_options", "relic_options", "potion_options"):
        value = reward.get(key)
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            result[key] = list(value)
    for key in ("relic_option", "potion_option", "gold_option"):
        value = reward.get(key)
        if (isinstance(value, str) if key != "gold_option"
                else type(value) is int):
            result[key] = value
    for key, field in (
        ("pending_card_groups", "card_options"),
        ("pending_extra_relic_rewards", "relic_id"),
        ("pending_extra_potion_rewards", "potion_id"),
        ("pending_extra_gold_rewards", "amount"),
    ):
        raw = reward.get(key)
        if not isinstance(raw, list):
            continue
        groups: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict) or type(item.get("group_index")) is not int:
                continue
            value = item.get(field)
            valid = (isinstance(value, list)
                     and all(isinstance(v, str) for v in value)) if field == "card_options" else (
                         type(value) is int if field == "amount"
                         else isinstance(value, str))
            if valid:
                groups.append({"group_index": item["group_index"], field: value})
        if groups:
            result[key] = groups
    extra = reward.get("extra_card_reward_groups_remaining")
    if type(extra) is int and extra >= 0:
        result["extra_card_reward_groups_remaining"] = extra
    return result


def public_persistent_state(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Versioned feature projection from an emulator fair-player observation.

    Opaque IDs and arbitrary relic.state properties are deliberately discarded.
    Known persistent counter residues remain separate, including copy index.
    """
    if not isinstance(observation, Mapping):
        raise ValueError("Expected fair-player observation mapping")
    result: dict[str, Any] = {"schema": STATE_SCHEMA}
    for name in ("act", "floor"):
        value = observation.get(name)
        if value is not None:
            result[name] = _integer(value, name)
    for name in ("hp", "max_hp", "gold"):
        result[name] = _integer(observation.get(name), name)
    for name in ("ruleset_id", "character_id", "act_one_region",
                 "act_one_boss_encounter_id"):
        value = observation.get(name)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            result[name] = value

    deck = []
    for item in _objects(observation.get("deck"), "deck"):
        deck.append({
            "card_id": _identity(item, "card_id"),
            "upgrade_level": _integer(item.get("upgrade_level"), "upgrade_level"),
        })
    result["deck"] = sorted(deck, key=lambda c: (c["card_id"], c["upgrade_level"]))

    relics = []
    for index, item in enumerate(_objects(observation.get("relics"), "relics")):
        relic_id = _identity(item, "relic_id")
        relic: dict[str, Any] = {
            "index": index, "relic_id": relic_id,
            "is_wax": item.get("is_wax") is True,
            "is_melted": item.get("is_melted") is True,
        }
        spec = PERSISTENT_COUNTERS.get(relic_id)
        if spec is not None:
            property_name, modulus = spec
            raw = item.get("state")
            state = raw if isinstance(raw, dict) else {}
            count = state.get(property_name, 0)
            value = _integer(count, f"{relic_id}.{property_name}")
            if value < 0:
                raise ValueError("Negative persistent trigger count")
            relic["counter"] = {
                "name": property_name, "residue": value % modulus,
                "modulus": modulus,
            }
        relics.append(relic)
    result["relics"] = relics

    potions = []
    for item in _objects(observation.get("potions"), "potions"):
        potions.append({
            "slot": _integer(item.get("slot"), "potion slot"),
            "potion_id": _identity(item, "potion_id"),
        })
    slots = [p["slot"] for p in potions]
    if len(slots) != len(set(slots)) or any(x < 0 for x in slots):
        raise ValueError("Invalid public potion slots")
    result["potions"] = sorted(potions, key=lambda p: p["slot"])

    nodes = observation.get("map")
    if isinstance(nodes, list):
        counts: Counter[tuple[int, str]] = Counter()
        for node in nodes:
            if not isinstance(node, dict):
                continue
            floor, room = node.get("floor"), node.get("room_type")
            if type(floor) is int and isinstance(room, (str, int)) and not isinstance(room, bool):
                counts[(floor, str(room))] += 1
        result["visible_map_room_counts"] = [
            {"floor": floor, "room_type": room, "count": n}
            for (floor, room), n in sorted(counts.items())
        ]
    reward = observation.get("reward")
    if isinstance(reward, dict):
        result["reward_offers"] = _visible_rewards(reward)
    return result


def _inventory_delta(before: list[dict[str, Any]], after: list[dict[str, Any]],
                     *, ignore: tuple[str, ...] = ()) -> dict[str, list[dict[str, Any]]]:
    """Multiplicity-aware: item slot/index is irrelevant to consumption."""
    def count(items: list[dict[str, Any]]) -> Counter[str]:
        return Counter(
            json.dumps({k: v for k, v in item.items() if k not in ignore},
                       sort_keys=True)
            for item in items
        )
    b, a = count(before), count(after)
    return {
        "removed": [json.loads(x) for x in sorted((b - a).elements())],
        "added": [json.loads(x) for x in sorted((a - b).elements())],
    }


def _bonus_receipts(exit_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    reward = exit_state.get("reward_offers")
    if not isinstance(reward, dict):
        return []
    extra = reward.get("extra_card_reward_groups_remaining")
    if type(extra) is int and extra > 0:
        # A published bonus is observable; its causal trigger may not be.
        # Do not attribute this to The Hunt without a public event receipt.
        return [{"kind": "extra_card_reward_groups_visible",
                 "count": extra, "source": "unattributed_public_reward"}]
    return []


class H14CombatBoundaryTrace:
    """Synchronous, passive observer; record committed actions after successful step.

    A truncated combat never receives an invented survival or resource label.
    """

    def __init__(self, *, source_run_seed: str, combat_policy_id: str):
        if not source_run_seed or not combat_policy_id:
            raise ValueError("Source run seed and combat policy identity are required")
        self.source_run_seed = source_run_seed
        self.combat_policy_id = combat_policy_id
        self.combats: list[dict[str, Any]] = []
        self._pending: dict[str, Any] | None = None
        self._frame = 0

    def observe(self, observation: Mapping[str, Any]) -> None:
        state = public_persistent_state(observation)
        in_combat = isinstance(observation.get("combat"), dict)
        outcome = observation.get("terminal_outcome")
        if in_combat and self._pending is None and outcome != "defeat":
            self._pending = {
                "entry_observation_index": self._frame,
                "entry": state,
                "potion_uses": [],
                "combat_decisions": 0,
            }
        elif self._pending is not None and (not in_combat or outcome == "defeat"):
            pending = self._pending
            survived = outcome != "defeat" and state["hp"] > 0
            if not survived and outcome not in ("defeat", None):
                raise ValueError("Contradictory terminal combat observation")
            transition = {
                "schema": TRANSITION_SCHEMA,
                "source_run_seed": self.source_run_seed,
                "combat_policy_id": self.combat_policy_id,
                "entry_observation_index": pending["entry_observation_index"],
                "exit_observation_index": self._frame,
                "combat_decisions": pending["combat_decisions"],
                "survived": survived,
                "entry": pending["entry"],
                "exit": state,
                "hp_delta": state["hp"] - pending["entry"]["hp"],
                "gold_delta_observed": state["gold"] - pending["entry"]["gold"],
                "potion_uses": pending["potion_uses"],
                "potion_inventory_delta": _inventory_delta(
                    pending["entry"]["potions"], state["potions"], ignore=("slot",)),
                "deck_delta": _inventory_delta(
                    pending["entry"]["deck"], state["deck"]),
                "relic_state_before": pending["entry"]["relics"],
                "relic_state_after": state["relics"],
                "effect_receipts": _bonus_receipts(state),
                "boundary": ("defeat" if not survived
                             else "reward_decision" if "reward_offers" in state
                             else "macro_decision"),
            }
            self.combats.append(transition)
            self._pending = None
        self._frame += 1

    def record_committed_action(
        self, observation: Mapping[str, Any], *, kind: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Call after a successful emulator step, before observing its child."""
        if not isinstance(observation.get("combat"), dict):
            return
        pending = self._pending
        if pending is None:
            raise ValueError("Committed combat action without entry")
        pending["combat_decisions"] += 1
        if kind != "use_potion":
            return
        slot = payload.get("Slot", payload.get("slot"))
        if type(slot) is not int:
            raise ValueError("Potion-use action lacks public slot")
        potions = public_persistent_state(observation)["potions"]
        matches = [item for item in potions if item["slot"] == slot]
        if len(matches) != 1:
            raise ValueError("Committed potion use lacks visible held potion")
        pending["potion_uses"].append({
            "slot": slot, "potion_id": matches[0]["potion_id"],
        })

    def finish(self, *, censored: bool) -> list[dict[str, Any]]:
        if self._pending is not None and not censored:
            raise ValueError("Unclosed combat in a completed trace")
        if self._pending is not None:
            self._pending = None  # Do not assign labels to partial combat.
        return list(self.combats)
