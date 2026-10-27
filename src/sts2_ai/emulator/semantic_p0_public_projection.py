"""Public-only adaptation of the restricted P0 semantic observation to v15.

P0 legitimately reports some hidden piles by *count*, while legacy v15
expected arrays of publicly identified card records. The adapter encodes
an **opaque, unknown-identity pile**, retaining the observed count without
fabricating individual cards, types or hidden positions. Missing optional
public arrays are explicitly marked unknown, never silently asserted empty.
The original source observation is not mutated.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

PROJECTION = "sts2-ai-p0-public-count-piles-v1"
CARD_ZONES = (
    "draw_pile", "discard_pile", "exhaust_pile", "play_pile", "choice_pool",
)
PUBLIC_OPTIONAL = ("relics", "potions")


def project_public_p0_observation(source: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(source, dict) or not isinstance(source.get("combat"), dict):
        raise ValueError("Expected fair combat observation object")
    if "semanticDeck" in source or "pending" in source:
        raise ValueError("Privileged Nature state must never reach player projection")
    if source.get("p0_public_projection") is not None:
        raise ValueError("Cannot project a previously modified P0 observation")
    public = deepcopy(source)
    combat = public["combat"]
    opaque: list[str] = []

    def normalize(container: dict[str, Any], key: str, *, marker: str) -> None:
        value = container.get(key)
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            return
        if value is None:
            # Unknown, rather than a claim that the source has no such items.
            opaque.append(marker)
            container[key] = []
            return
        if type(value) is int and value >= 0:
            declared = container.get(key + "_count")
            if declared is not None and (
                type(declared) is not int or declared != value
            ):
                raise ValueError(f"Inconsistent source P0 count at {marker}")
            opaque.append(marker)
            container[key] = []
            container[key + "_count"] = value
            return
        raise ValueError(f"Unexpected public P0 field {marker}")

    if not isinstance(combat.get("hand"), list):
        raise ValueError("Player hand must reveal physical card instance records")
    if not isinstance(combat.get("enemies"), list):
        raise ValueError("Combat enemies must be publicly enumerated")
    for zone in CARD_ZONES:
        normalize(combat, zone, marker="combat." + zone)
    for key in ("relic_counters", "player_powers"):
        normalize(combat, key, marker="combat." + key)
    for key in PUBLIC_OPTIONAL:
        normalize(public, key, marker=key)
    for enemy in combat["enemies"]:
        if not isinstance(enemy, dict):
            raise ValueError("Malformed public P0 enemy")
        if enemy.get("powers") is None:
            enemy["powers"] = []
            opaque.append("enemy.powers")
    if "combat.draw_pile" in opaque:
        # Preserve the actual source-declared draw count. This is never
        # converted into bogus physical instance identities.
        if type(combat.get("draw_pile_count")) is not int:
            if type(source["combat"].get("draw_pile")) is int:
                combat["draw_pile_count"] = source["combat"]["draw_pile"]
            else:
                raise ValueError("Unknown draw-pile count for P0 PPO")
        combat["draw_pile_opaque"] = True
    public["p0_public_projection"] = PROJECTION
    public["p0_opaque_public_fields"] = sorted(set(opaque))
    return public


def project_public_enemy_slot_for_legacy_instances(source: dict[str, Any]) -> dict[str, Any]:
    """Recover the unambiguous HUD slot in a single-enemy combat.

    The restricted P0 observation publishes the ordered living-enemy list
    with visible instance IDs, species and intents but omits the v8 name
    `formation_position`. In the selected single-enemy H1 fixture its
    only public slot is exactly 0. Do not manufacture multi-enemy formation
    positions: source P0 currently lacks the necessary public slot data.
    """
    if not isinstance(source, dict) or not isinstance(source.get("combat"), dict):
        raise ValueError("Legacy instance projection requires fair combat observation")
    if "semanticDeck" in source or "pending" in source:
        raise ValueError("Hidden chance state cannot enter legacy policy projection")
    enemies = source["combat"].get("enemies")
    if not isinstance(enemies, list):
        raise ValueError("Legacy instance projection requires a visible enemy list")
    if all(
        isinstance(enemy, dict) and type(enemy.get("formation_position")) is int
        for enemy in enemies
    ):
        return source
    if len(enemies) != 1 or not isinstance(enemies[0], dict):
        raise ValueError(
            "P0 has no formation positions for multiple living enemies; "
            "legacy instance policy projection unsupported"
        )
    enemy = enemies[0]
    if (type(enemy.get("instance_id")) is not int or
            not isinstance(enemy.get("enemy_id"), str) or
            not enemy.get("enemy_id")):
        raise ValueError("Cannot derive public slot for malformed enemy identity")
    projected = deepcopy(source)
    projected["combat"]["enemies"][0]["formation_position"] = 0
    return projected
