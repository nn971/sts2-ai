from __future__ import annotations

import json
from typing import Any, cast

from sts2_ai.emulator import Observation


class PrototypeHeuristicEvaluator:
    """Simple hand-written value baseline for prototype search.

    This is intentionally not a claim about strong Slay the Spire strategy. It exists to make
    search plumbing measurable before learned value models are available.
    """

    VERSION = "prototype-heuristic-v0"

    def evaluate(self, observation: Observation, *, terminal: bool) -> float:
        raw = json.loads(observation.payload_json)
        if not isinstance(raw, dict):
            raise ValueError("Prototype observation payload must be a JSON object")
        payload = cast(dict[str, Any], raw)

        outcome = payload.get("terminal_outcome")
        if terminal:
            if outcome == "victory":
                return 1_000_000.0
            if outcome == "defeat":
                return -1_000_000.0

        act = _number(payload.get("act"), default=0.0)
        floor = _number(payload.get("floor"), default=0.0)
        hp = _number(payload.get("hp"), default=0.0)
        max_hp = max(1.0, _number(payload.get("max_hp"), default=1.0))
        gold = _number(payload.get("gold"), default=0.0)

        deck = payload.get("deck")
        relics = payload.get("relics")
        potions = payload.get("potions")

        upgraded_cards = 0
        if isinstance(deck, list):
            for card in deck:
                if isinstance(card, dict) and _number(card.get("upgrade_level"), default=0.0) > 0:
                    upgraded_cards += 1

        relic_count = len(relics) if isinstance(relics, list) else 0
        potion_count = len(potions) if isinstance(potions, list) else 0

        # Progress dominates ordinary resource differences. HP then keeps search from preferring
        # reckless lines that merely advance one room. The remaining terms are deliberately small.
        return (
            (act * 1_000.0)
            + (floor * 100.0)
            + ((hp / max_hp) * 80.0)
            + (max_hp * 0.2)
            + (gold * 0.03)
            + (relic_count * 12.0)
            + (potion_count * 4.0)
            + (upgraded_cards * 2.0)
        )


def _number(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    return default
