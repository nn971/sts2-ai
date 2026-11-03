#!/usr/bin/env python3
"""Compare the H21 public-field contract to the emulator's actual C# DTOs.

This guards optional fields never encountered by seeded live-run probes.
Computed public properties (such as ShopOffer.UndiscountedPrice) are covered.
No code is generated and no private RunState fields are consumed.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from sts2_ai.training.h21_field_coverage import PUBLIC_V1_FIELD_CONTRACT


PUBLIC_RECORDS = {
    "observation": "PrototypeAiObservation",
    "card": "PrototypeAiCard",
    "relic": "PrototypeAiRelic",
    "potion": "PrototypeAiPotion",
    "map_node": "PrototypeAiMapNode",
    "combat": "PrototypeAiCombat",
    "combat_history": "PrototypeAiCombatHistory",
    "enemy": "PrototypeAiEnemy",
    "power": "PrototypeAiPower",
    "reward": "PrototypeAiReward",
    "reward_card_group": "PrototypeAiRewardCardGroup",
    "card_preview": "PrototypeAiCardAcquisitionPreview",
    "extra_relic": "PrototypeAiExtraRelicReward",
    "extra_potion": "PrototypeAiExtraPotionReward",
    "extra_gold": "PrototypeAiExtraGoldReward",
    "card_reroll": "PrototypeAiRewardCardReroll",
    "deck_choice": "PrototypeAiDeckChoice",
    "shop": "PrototypeAiShop",
    "shop_offer": "ShopOffer",
    "shop_card_preview": "PrototypeAiShopCardOfferPreview",
    "event": "PrototypeAiEvent",
    "event_deck_choice": "PrototypeAiEventDeckChoice",
    "event_potion_choice": "PrototypeAiEventPotionReplacement",
    "queued_deck_choice": "PrototypeAiEventQueuedDeckChoice",
    "event_choice_value": "PrototypeAiEventChoiceValue",
    "completed_room": "PrototypeCompletedRoomRecord",
}

MODEL_RECORDS = {"shop_offer", "completed_room"}


def snake(name: str) -> str:
    # Matches System.Text.Json's SnakeCaseLower policy for the public DTO
    # member names currently used in the emulator.
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


def parameters(source: str, record: str) -> tuple[str, ...]:
    # Remove line comments before top-level comma splitting: a public DTO
    # parameter comment can contain commas (e.g. inventory, even when empty).
    source = re.sub(r"//[^\n]*", "", source)
    pattern = r"\bpublic\s+sealed\s+record\s+" + re.escape(record) + r"\s*\("
    match = re.search(pattern, source)
    if not match:
        raise ValueError("Missing public emulator record " + record)
    start = match.end()
    depth = 1
    square = 0
    angle = 0
    parts = []
    begin = start
    for index in range(start, len(source)):
        char = source[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "[":
            square += 1
        elif char == "]":
            square -= 1
        elif char == "<":
            angle += 1
        elif char == ">":
            angle -= 1
        if depth == 0:
            parts.append(source[begin:index])
            break
        if char == "," and depth == 1 and square == 0 and angle == 0:
            parts.append(source[begin:index])
            begin = index + 1
    else:
        raise ValueError("Unterminated emulator public record " + record)

    names = []
    for part in parts:
        # Remove comments and complete [property: ...] attributes before
        # looking for the declaration (attributes can contain = expressions).
        part = re.sub(r"//[^\n]*", "", part)
        part = re.sub(r"\[property:[\s\S]*?\]", "", part)
        declaration = part.split("=", 1)[0].strip()
        name = re.search(r"([A-Za-z_]\w*)\s*$", declaration)
        if name is None:
            raise ValueError(
                "Unparsed emulator DTO parameter in "
                + record + ": " + repr(declaration))
        names.append(snake(name.group(1)))
    if len(set(names)) != len(names):
        raise ValueError("Duplicate public DTO field " + record)
    return tuple(names)


def compare_source_contract(root: Path) -> None:
    core = root / "src" / "Sts2Emulator.Core"
    public_source = (core / "PrototypeAiEnvironment.cs").read_text()
    model_source = (core / "PrototypeModel.cs").read_text()
    if set(PUBLIC_RECORDS) != set(PUBLIC_V1_FIELD_CONTRACT):
        raise ValueError("H21 contract record classes differ from source map")
    issues = []
    for key, record in PUBLIC_RECORDS.items():
        source = model_source if key in MODEL_RECORDS else public_source
        fields = set(parameters(source, record))
        # System.Text.Json serializes public computed getters by default.
        if key == "shop_offer":
            if not re.search(r"public\s+int\s+UndiscountedPrice\s*=>", source):
                raise ValueError("Expected ShopOffer computed getter is missing")
            fields.add("undiscounted_price")
        expected = set(PUBLIC_V1_FIELD_CONTRACT[key])
        if fields != expected:
            issues.append({
                "record": record, "missing_contract": sorted(fields - expected),
                "stale_contract": sorted(expected - fields),
            })
    if issues:
        raise ValueError("Public v1 DTO contract drift: " + repr(issues))
    print(f"H21 public v1 DTO contract verified: {len(PUBLIC_RECORDS)} record types")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emulator-root", type=Path, required=True)
    args = parser.parse_args()
    compare_source_contract(args.emulator_root)


if __name__ == "__main__":
    main()
