"""H17 adaptive PUCT allocation and auditable per-card training coverage.

This module uses only public observations and already-computed actor logits.
It never inspects hidden emulator state and does not perform neural inference.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sts2_ai.emulator.protocol import LegalAction


@dataclass(frozen=True)
class AdaptiveSearch:
    probability16: float = 0.15
    probability64: float = 0.02

    def __post_init__(self) -> None:
        if (not math.isfinite(self.probability16)
                or not math.isfinite(self.probability64)
                or self.probability16 < 0 or self.probability64 < 0
                or self.probability16 + self.probability64 > 0.5):
            raise ValueError("Invalid adaptive PUCT probability budgets")


def card_for_action(state: dict[str, Any], action: LegalAction) -> str | None:
    """Resolve *public* play-card instance to its card type, not instance ID."""
    if action.kind != "play_card":
        return None
    combat = state.get("combat")
    if not isinstance(combat, dict):
        return None
    payload = json.loads(action.payload_json)
    ref = payload.get("CardInstanceId", payload.get("card_instance_id"))
    if type(ref) is not int:
        return None
    hand = combat.get("hand")
    if not isinstance(hand, list):
        return None
    for card in hand:
        if (isinstance(card, dict) and card.get("instance_id") == ref
                and isinstance(card.get("card_id"), str)):
            return card["card_id"]
    return None


def playable_card_types(state: dict[str, Any],
                        actions: Sequence[LegalAction]) -> set[str]:
    return {found for action in actions
            if (found := card_for_action(state, action)) is not None}


def normalized_entropy(logits: Sequence[float]) -> float:
    if len(logits) < 2:
        return 0.0
    if any(not math.isfinite(logit) for logit in logits):
        raise ValueError("Nonfinite actor scores")
    peak = max(logits)
    weights = [math.exp(logit - peak) for logit in logits]
    total = math.fsum(weights)
    return -math.fsum(
        (w / total) * math.log(w / total) for w in weights if w > 0
    ) / math.log(len(logits))


def choose_budget(
    cfg: AdaptiveSearch, state: dict[str, Any],
    actions: Sequence[LegalAction], logits: Sequence[float],
    exposure: Mapping[str, int], rng: random.Random,
) -> int:
    """Choose 0/16/64 full-combat simulations, with random audit coverage.

    Percentages are *targets* modulated by uncertainty, low HP, late floors,
    and rarity. This is not a hard quota, and sampled decision outcomes remain
    on-policy under the resulting behavior distribution.
    """
    if (len(actions) != len(logits) or len(actions) < 2
            or not isinstance(state.get("combat"), dict)):
        return 0
    uncertainty = normalized_entropy(logits)
    hp = state.get("hp")
    max_hp = state.get("max_hp")
    health_risk = (
        1.0 - min(1.0, max(0.0, hp / max_hp))
        if type(hp) is int and type(max_hp) is int and max_hp > 0 else 0.5
    )
    floor = state.get("floor")
    progress = min(1.0, max(0.0, floor / 16)) if type(floor) is int else 0.5
    cards = playable_card_types(state, actions)
    rarity = (
        max(1.0 / math.sqrt(1 + exposure.get(name, 0)) for name in cards)
        if cards else 0.25
    )
    # Random audit component guarantees some search for confidently-wrong
    # policies. Values bounded so no root becomes prohibitively probable.
    importance = (
        0.20 + 0.35 * uncertainty + 0.20 * health_risk
        + 0.10 * progress + 0.15 * rarity
    )
    multiplier = 0.50 + importance
    p64 = min(0.25, cfg.probability64 * multiplier)
    p16 = min(0.40, cfg.probability16 * multiplier)
    draw = rng.random()
    if draw < p64:
        return 64
    if draw < p64 + p16:
        return 16
    return 0


def update_coverage(
    episodes: Sequence[Any], counters: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    """Persistent card exposure by *type* across complete native runs.

    Deck is counted once per run, playable once per decision/card type,
    chosen once per play, searched once per searched root/card type.
    """
    output: dict[str, Counter[str]] = {
        label: Counter(counters.get(label, {}))
        for label in ("deck_runs", "playable", "chosen", "searched")
    }
    for episode in episodes:
        if not episode.complete:
            continue
        deck_types: set[str] = set()
        for state in episode.frames:
            deck = state.get("deck")
            if isinstance(deck, list):
                deck_types.update(
                    item["card_id"] for item in deck
                    if isinstance(item, dict)
                    and isinstance(item.get("card_id"), str)
                )
        output["deck_runs"].update(deck_types)
        for decision, state in zip(
            episode.decisions, episode.frames, strict=False,
        ):
            cards = playable_card_types(state, decision.legal_actions)
            output["playable"].update(cards)
            if decision.search_policy is not None:
                output["searched"].update(cards)
            chosen = card_for_action(
                state, decision.legal_actions[decision.chosen_index],
            )
            if chosen is not None:
                output["chosen"][chosen] += 1
    return {key: dict(sorted(group.items())) for key, group in output.items()}


def coverage_summary(coverage: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    return {
        "distinct_card_types": {
            key: len(coverage.get(key, {}))
            for key in ("deck_runs", "playable", "chosen", "searched")
        },
        "total_exposures": {
            key: sum(coverage.get(key, {}).values())
            for key in ("deck_runs", "playable", "chosen", "searched")
        },
        "least_played_cards": sorted(
            coverage.get("playable", {}).items(), key=lambda pair: (pair[1], pair[0]),
        )[:25],
    }
