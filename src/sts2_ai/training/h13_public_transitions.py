"""H13: public-only combat/reward separation and empirical encounter law.

The data source is *observed* full native runs under pinned H6-C/H9.
A room choice exposes its destination type and floor, not the enemies or
opening hand. The empirical catalog therefore contains enemy-composition
outcomes, including no-combat outcomes, but is NOT a combat-entry generator.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from sts2_ai.training.h12_reward_settlement import settled_public_resources

CONTEXT_VERSION = "sts2-h13-visible-map-reward-choice-v1"
ENCOUNTER_LAW_SCHEMA = "sts2-h13-observed-room-encounters-v1"

_REWARD_KINDS = (
    "reward", "skip_reward", "sacrifice_reward",
)
_REWARD_LISTS = (
    "card_options", "relic_options", "potion_options",
)
_REWARD_SINGLES = ("relic_option", "potion_option")
_REWARD_GROUPS = (
    ("pending_card_groups", "card_options"),
    ("pending_extra_relic_rewards", "relic_id"),
    ("pending_extra_potion_rewards", "potion_id"),
    ("pending_extra_gold_rewards", "amount"),
)


def _visible_reward_offers(reward: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist published offer identities; discard all other reward fields."""
    offers: dict[str, Any] = {}
    for key in _REWARD_LISTS:
        value = reward.get(key)
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            offers[key] = list(value)
    for key in _REWARD_SINGLES:
        value = reward.get(key)
        if isinstance(value, str):
            offers[key] = value
    for key, field in _REWARD_GROUPS:
        entries = reward.get(key)
        if not isinstance(entries, list):
            continue
        projected = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            value = item.get(field)
            if field == "card_options":
                if not isinstance(value, list) or not all(
                    isinstance(x, str) for x in value
                ):
                    continue
            elif field == "amount":
                if type(value) is not int:
                    continue
            elif not isinstance(value, str):
                continue
            group_index = item.get("group_index")
            if type(group_index) is not int:
                continue
            projected.append({"group_index": group_index, field: value})
        if projected:
            offers[key] = projected
    return offers


def public_choice_context(
    state: Mapping[str, Any], kind: str, payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture *only* information visible when a legal choice is made.

    Map node IDs are used for the lookup, then discarded. Reward choices
    retain explicit published offers; future reward RNG is not consulted.
    """
    if kind == "choose_map_node":
        node_id = payload.get("NodeId", payload.get("node_id"))
        nodes = state.get("map")
        if not isinstance(node_id, str) or not isinstance(nodes, list):
            raise ValueError("Map choice lacks its published map")
        matching = [
            n for n in nodes if isinstance(n, dict)
            and n.get("node_id") == node_id
        ]
        if len(matching) != 1:
            raise ValueError("Chosen map node is absent or ambiguous")
        node = matching[0]
        if (type(node.get("floor")) is not int
                or type(state.get("act")) is not int
                or type(node.get("room_type")) not in (int, str)):
            raise ValueError("Chosen map node lacks public type/floor")
        return {"map_target": {
            "act": state["act"], "floor": node["floor"],
            "room_type": node["room_type"],
        }}
    if any(marker in kind for marker in _REWARD_KINDS):
        reward = state.get("reward")
        if isinstance(reward, dict):
            offers = _visible_reward_offers(reward)
            if offers:
                return {"reward_offers": offers}
    return {}


def _ids(resources: Mapping[str, Any], key: str, name: str) -> Counter[str]:
    raw = resources.get(key, [])
    if not isinstance(raw, list):
        raise ValueError(f"Invalid public {key}")
    result: Counter[str] = Counter()
    for item in raw:
        identity = item if isinstance(item, str) else (
            item.get(name) if isinstance(item, dict) else None
        )
        if not isinstance(identity, str):
            raise ValueError(f"Invalid public {key} identity")
        result[identity] += 1
    return result


def _as_names(counter: Counter[str]) -> list[str]:
    return sorted(counter.elements())


def observed_combat_boundary(
    run: Mapping[str, Any], index: int,
) -> dict[str, Any]:
    """Return immediate combat consequences and separate later choices.

    postcombat_snapshot is diagnostic and H6-C-policy-conditioned; it may
    include non-reward same-floor actions. It is *not* a combat transition.
    """
    combats = run["combats"]
    combat = combats[index]
    before, after = combat["entry"], combat["exit"]
    survived = combat["survived"]
    if (type(survived) is not bool
            or type(after.get("hp")) is not int
            or survived != (after["hp"] > 0)):
        raise ValueError("Invalid observed immediate combat exit")
    if not all(type(x.get("gold")) is int for x in (before, after)):
        raise ValueError("Combat boundary has no public gold")
    pot_before = _ids(before, "potions", "potion_id")
    pot_after = _ids(after, "potions", "potion_id")
    deck_before = _ids(before, "deck", "card_id")
    deck_after = _ids(after, "deck", "card_id")
    relic_before = _ids(before, "relics", "relic_id")
    relic_after = _ids(after, "relics", "relic_id")
    immediate = {
        "survived": survived,
        "entry_hp": before["hp"], "exit_hp": after["hp"],
        "max_hp_at_entry": before["max_hp"],
        "hp_delta": after["hp"] - before["hp"],
        "gold_delta": after["gold"] - before["gold"],
        "potions_consumed": _as_names(pot_before - pot_after),
        "potions_gained_in_combat": _as_names(pot_after - pot_before),
        "cards_added_in_combat": _as_names(deck_after - deck_before),
        "cards_removed_in_combat": _as_names(deck_before - deck_after),
        "relics_added_in_combat": _as_names(relic_after - relic_before),
        "relics_removed_in_combat": _as_names(relic_before - relic_after),
    }
    # The full structured immediate exit is authoritative for upgrades and
    # transformations (which cannot be encoded by card ID counts alone).
    stop = (combats[index + 1]["entry_observation_index"]
            if index + 1 < len(combats) else float("inf"))
    reward_decisions = []
    floor, act = after.get("floor"), after.get("act")
    for frame in run["macro_states"]:
        at = frame["observation_index"]
        if at < combat["exit_observation_index"]:
            continue
        if at >= stop:
            break
        state = frame["state"]
        if state.get("floor") != floor or state.get("act") != act:
            break
        selected = frame.get("selected_action")
        if isinstance(selected, dict) and "reward_offers" in selected:
            reward_decisions.append({
                "observation_index": at,
                "offers": selected["reward_offers"],
                "selected_kind": selected["kind"],
                "selected_payload": selected["payload"],
            })
    settled, status = settled_public_resources(dict(run), index)
    return {
        "combat_immediate": immediate,
        "immediate_exit": after,
        "reward_decisions": reward_decisions,
        "postcombat_snapshot": settled,
        "postcombat_snapshot_status": status,
        "postcombat_snapshot_is_policy_conditioned": True,
    }


def observed_room_encounters(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Join selected visible rooms with their subsequently realized fights.

    A room with no ensuing fight is retained as an explicit empty encounter.
    Old H12 journals lack choice metadata and must be recollected for this
    empirical law. No future information is supplied to the actor.
    """
    if run.get("public_choice_context_version") != CONTEXT_VERSION:
        raise ValueError("Run lacks H13 choice context; old H12 journal")
    if run.get("censored"):
        return []
    choices = [
        frame for frame in run["macro_states"]
        if isinstance(frame.get("selected_action"), dict)
        and isinstance(frame["selected_action"].get("map_target"), dict)
    ]
    combats = run["combats"]
    result = []
    for i, frame in enumerate(choices):
        room = frame["selected_action"]["map_target"]
        start = frame["observation_index"]
        stop = (choices[i + 1]["observation_index"]
                if i + 1 < len(choices) else float("inf"))
        fights = []
        for combat in combats:
            at = combat["entry_observation_index"]
            if start < at < stop:
                entry = combat["entry"]
                if (entry.get("act") != room["act"]
                        or entry.get("floor") != room["floor"]):
                    continue
                enemies = entry.get("combat", {}).get("enemies", [])
                if not isinstance(enemies, list):
                    raise ValueError("Combat lacks a public enemy list")
                names = []
                for enemy in enemies:
                    if not isinstance(enemy, dict) or not isinstance(
                        enemy.get("enemy_id"), str
                    ):
                        raise ValueError("Combat lacks a public enemy identity")
                    names.append(enemy["enemy_id"])
                fights.append(sorted(names))
        result.append({
            "act": room["act"], "floor": room["floor"],
            "room_type": room["room_type"],
            "enemy_groups": fights,
            "source_run_seed": run["source_run_seed"],
        })
    return result


def _floor_band(floor: int) -> int:
    return floor // 5


@dataclass(frozen=True)
class EncounterDistribution:
    """Observable room-conditioned law, including probability of no fight."""

    count: int
    # Each outcome lists enemy identity groups for all fights on that floor.
    outcomes: tuple[tuple[tuple[tuple[str, ...], ...], float], ...]


class EmpiricalEncounterLaw:
    """Empirical observations, not counterfactual combat-entry states.

    The next step is an independent legal opening-hand / combat-entry sampler
    using the interchangeable stochastic provider. H12.predict must only
    receive *complete* generated public combat entries from that sampler.
    """

    def __init__(self, *, min_count: int = 30) -> None:
        if min_count < 1:
            raise ValueError("min_count must be positive")
        self.min_count = min_count
        self.counts: dict[
            tuple[int, int | str, int],
            Counter[tuple[tuple[str, ...], ...]],
        ] = {}

    def ingest(self, runs: Iterable[Mapping[str, Any]]) -> None:
        for run in runs:
            for row in observed_room_encounters(run):
                key = (
                    row["act"], row["room_type"], _floor_band(row["floor"]),
                )
                groups = tuple(tuple(group) for group in row["enemy_groups"])
                self.counts.setdefault(key, Counter())[groups] += 1

    def distribution(
        self, *, act: int, room_type: int | str, floor: int,
    ) -> EncounterDistribution | None:
        hist = self.counts.get((act, room_type, _floor_band(floor)))
        if hist is None:
            return None
        total = sum(hist.values())
        if total < self.min_count:
            return None
        return EncounterDistribution(
            count=total,
            outcomes=tuple(
                (groups, count / total)
                for groups, count in sorted(hist.items())
            ),
        )
