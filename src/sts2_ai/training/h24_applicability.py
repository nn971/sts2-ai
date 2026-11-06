"""H24 resource-dominance predicate registry and fail-closed pair auditor.

A passing structural check is *not* proof of gameplay dominance. Approval and
revision-pinned mechanics review are independent evidence gates. Neither is
populated by default. This module does not alter H22 actor or critic.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

PREDICATES = {
    "G1": "Identical observable non-resource context",
    "G2": "Chance laws admit a dominance-preserving coupling",
    "G3": "Only selected resource and necessary bookkeeping differ",
    "G4": "No forced harmful trigger from resource increase",
    "G5": "Greater-resource continuation simulates lesser-resource actions",
    "G6": "States valid and future mechanics audited at pinned emulator revision",
    "H1": "Both HP positive and bounded by identical max HP",
    "H2": "HP thresholds, powers, cards, relics and events safe",
    "H3": "HP gain cannot reduce terminal clear prospects",
    "D1": "Nonnegative gold strictly increases",
    "D2": "Affordable actions under smaller balance remain feasible",
    "D3": "Gold-based costs, triggers and events cannot harm extra gold",
    "P1": "Exactly one potion inserted in a previously vacant slot",
    "P2": "Exact potion identity and persistent state are approved and audited",
    "P3": "No forced, automatic or adverse potion-use effects",
    "P4": "Empty-slot triggers, capacity and replacement remain safe",
    "P5": "Old continuation can ignore, use or safely replace extra potion",
    "R1": "Exactly one relic appended, previous relics unchanged",
    "R2": "Added relic starts in audited valid persistent state",
    "R3": "Acquisition and ongoing effects cannot be detrimental",
    "R4": "Existing legal outcomes are simulatable with added relic",
    "R5": "Persistent counters, randomized hooks and other interactions audited",
}

REQUIRED = {
    "hp": ("G2", "G4", "G5", "G6", "H2", "H3"),
    "gold": ("G2", "G4", "G5", "G6", "D2", "D3"),
    "potion": ("G2", "G4", "G5", "G6", "P2", "P3", "P4", "P5"),
    "relic": ("G2", "G4", "G5", "G6", "R3", "R4", "R5"),
}


@dataclass(frozen=True)
class ResourceAuthority:
    """Separate human verdict from mechanics proof; defaults to none.

    The caller must supply a source-revision-matched mechanics review. The
    fields here are evidence *assertions*, not an automatic emulator proof.
    """

    emulator_revision: str = ""
    user_approved: frozenset[str] = frozenset()
    mechanics_verified: frozenset[str] = frozenset()
    user_potions: frozenset[str] = frozenset()
    verified_potions: frozenset[str] = frozenset()
    user_relics: frozenset[str] = frozenset()
    verified_relics: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        unknown = (self.user_approved | self.mechanics_verified) - PREDICATES.keys()
        if unknown:
            raise ValueError(f"Unknown H24 predicates: {sorted(unknown)}")
        for values in (self.user_potions, self.verified_potions,
                       self.user_relics, self.verified_relics):
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError("Resource IDs must be nonempty strings")


@dataclass(frozen=True)
class PairAudit:
    resource: str
    structural_failures: tuple[str, ...]
    missing_user_verdict: tuple[str, ...]
    missing_mechanics_review: tuple[str, ...]
    added_item: str | None = None

    @property
    def eligible(self) -> bool:
        return not (self.structural_failures or self.missing_user_verdict
                    or self.missing_mechanics_review)


def _number(value: Any) -> bool:
    return type(value) is int and value >= 0


def _potion_slots(state: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    capacity = state.get("potion_slot_capacity")
    items = state.get("potions")
    if not _number(capacity) or not isinstance(items, list):
        raise ValueError("P1: missing capacity or potion list")
    slots: dict[int, dict[str, Any]] = {}
    for item in items:
        if (not isinstance(item, dict) or not _number(item.get("slot"))
                or item["slot"] >= capacity or not isinstance(item.get("potion_id"), str)
                or not item["potion_id"] or not isinstance(item.get("state"), dict)
                or item["slot"] in slots):
            raise ValueError("P1: invalid potion or duplicate slot")
        slots[item["slot"]] = item
    if (state.get("potion_slots_occupied") != len(slots)
            or state.get("potion_slots_empty") != capacity - len(slots)):
        raise ValueError("P1: inconsistent potion slot bookkeeping")
    return slots


def _neutral(state: Mapping[str, Any], resource: str) -> dict[str, Any]:
    projected = deepcopy(dict(state))
    fields = {
        "hp": ("hp",),
        "gold": ("gold",),
        "potion": ("potions", "potion_slots_occupied", "potion_slots_empty"),
        "relic": ("relics",),
    }[resource]
    for field in fields:
        projected.pop(field, None)
    return projected


def audit_public_pair(
    lower: Mapping[str, Any], higher: Mapping[str, Any], *,
    resource: str, authority: ResourceAuthority | None = None,
    emulator_revision: str,
) -> PairAudit:
    """Pure structural comparison plus explicit approval/review gates.

    Public-payload equality does not establish hidden-state equality, legal
    continuation inclusion, identical stochastic coupling or reachability.
    Those must be separately attested by an actual emulator audit.
    """
    if resource not in REQUIRED:
        raise ValueError("Unknown resource")
    if not isinstance(lower, Mapping) or not isinstance(higher, Mapping):
        raise TypeError("Need two public state mappings")
    auth = authority or ResourceAuthority()
    fail: list[str] = []
    added: str | None = None
    if not emulator_revision or auth.emulator_revision != emulator_revision:
        fail.append("G6: missing or mismatched emulator revision")
    if _neutral(lower, resource) != _neutral(higher, resource):
        fail.append("G1/G3: non-resource context differs")

    if resource == "hp":
        lo, hi, maxhp = lower.get("hp"), higher.get("hp"), lower.get("max_hp")
        if (not all(type(v) is int for v in (lo, hi, maxhp))
                or not (0 < lo < hi <= maxhp)):
            fail.append("H1: HP must strictly increase within unchanged max_hp")
    elif resource == "gold":
        lo, hi = lower.get("gold"), higher.get("gold")
        if not (_number(lo) and _number(hi) and lo < hi):
            fail.append("D1: gold must strictly increase and be nonnegative")
    elif resource == "potion":
        try:
            lo_slots = _potion_slots(lower)
            hi_slots = _potion_slots(higher)
            extra = sorted(set(hi_slots) - set(lo_slots))
            if (len(extra) != 1 or any(hi_slots.get(k) != v
                                      for k, v in lo_slots.items())
                    or len(hi_slots) != len(lo_slots) + 1):
                fail.append("P1: exactly one potion must be inserted")
            else:
                added = hi_slots[extra[0]]["potion_id"]
        except ValueError as error:
            fail.append(str(error))
    elif resource == "relic":
        lo, hi = lower.get("relics"), higher.get("relics")
        if (not isinstance(lo, list) or not isinstance(hi, list)
                or len(hi) != len(lo) + 1 or hi[:-1] != lo):
            fail.append("R1: append one relic, preserve all previous relics")
        else:
            item = hi[-1]
            if (not isinstance(item, dict)
                    or not isinstance(item.get("relic_id"), str)
                    or not item["relic_id"] or item.get("state") != {}
                    or item.get("is_wax", False) is not False
                    or item.get("is_melted", False) is not False):
                fail.append("R2: invalid or unreviewed relic initial state")
            else:
                added = item["relic_id"]

    user_missing = [p for p in REQUIRED[resource] if p not in auth.user_approved]
    mechanics_missing = [p for p in REQUIRED[resource] if p not in auth.mechanics_verified]
    if resource in ("potion", "relic"):
        if added is None:
            fail.append("Missing added resource identity")
        elif resource == "potion":
            if added not in auth.user_potions:
                user_missing.append("P2:item:" + added)
            if added not in auth.verified_potions:
                mechanics_missing.append("P2:item:" + added)
        elif resource == "relic":
            if added not in auth.user_relics:
                user_missing.append("R2:item:" + added)
            if added not in auth.verified_relics:
                mechanics_missing.append("R2:item:" + added)
    return PairAudit(resource, tuple(fail), tuple(user_missing),
                     tuple(mechanics_missing), added)
