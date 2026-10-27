"""H8 research-only A10 Silent fixture corpus for real emulator reference PPO.

Matches the versioned C# ResearchCombatFixtureGeneratorV111 recipe. Source
seeds select independent physical shuffles and Chance-Law outcomes; the
configuration index selects a *structural* deck/HP/relic/potion/encounter.
The distribution is deliberately curated, not native gameplay frequency.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from sts2_ai.nature.combat_scenarios import SCENARIOS, CombatScenario

DISTRIBUTION = "silent-a10-structured-combat-v1"
SOURCE_SCHEMA = "sts2-ai-h8-structured-source-bank-v1"
ENCOUNTERS = (
    ("proto.encounter.shrinker_beetle_weak", "ShrinkerBeetleWeak", 2, 1),
    ("proto.encounter.nibbits_weak", "NibbitsWeak", 2, 1),
    ("proto.encounter.mawler_normal", "MawlerNormal", 2, 1),
    ("proto.encounter.cubex_construct_normal", "CubexConstructNormal", 2, 1),
    ("proto.encounter.vine_shambler_normal", "VineShamblerNormal", 2, 1),
    ("proto.encounter.ceremonial_beast_boss", "CeremonialBeastBoss", 6, 5),
    ("proto.encounter.vantom_boss", "VantomBoss", 6, 5),
)
STARTER = ("proto.silent.strike",) * 5 + ("proto.silent.defend",) * 5 + (
    "proto.silent.neutralize", "proto.silent.survivor",
)
ADDITIONS = (
    "proto.silent.acrobatics", "proto.silent.backflip",
    "proto.silent.slice", "proto.silent.poisoned_stab",
    "proto.silent.deadly_poison", "proto.silent.catalyst",
)
RELICS = (
    "proto.relic.lantern", "proto.relic.vajra",
    "proto.relic.ornamental_fan", "proto.relic.orichalcum",
)
POTIONS = ("proto.potion.block", "proto.potion.fire", "proto.potion.poison")
_SOURCE = re.compile(
    r"^h8-(train|selection|confirmation|zero-shot)-v1-"
    r"(\d+)-(\d+)-(\d+)-(\d+)$"
)


def _roll(curriculum_seed: int, index: int, domain: str, ordinal: int) -> int:
    return int.from_bytes(hashlib.sha256(
        f"{DISTRIBUTION}:{curriculum_seed}:{index}:{domain}:{ordinal}"
        .encode()
    ).digest()[:8], "big")


def fixture_for_index(index: int, *, curriculum_seed: int = 20261027) -> dict[str, Any]:
    """Produce the C#-validated snake_case JSON fixture, independent of RNG seed."""
    if (
        type(index) is not int or index < 0
        or type(curriculum_seed) is not int or curriculum_seed < 0
    ):
        raise ValueError("Invalid structural fixture index/curriculum seed")
    encounter_id, native_entry, floor, row = ENCOUNTERS[
        _roll(curriculum_seed, index, "encounter", 0) % len(ENCOUNTERS)
    ]
    cards = [(name, 0) for name in STARTER]
    for ordinal in range(8):
        cards.append((
            ADDITIONS[_roll(curriculum_seed, index, "card", ordinal) % len(ADDITIONS)],
            int(_roll(curriculum_seed, index, "upgrade", ordinal) % 5 == 0),
        ))
    cards.sort()
    first = _roll(curriculum_seed, index, "relic", 0) % len(RELICS)
    second = (first + 1 + _roll(curriculum_seed, index, "relic", 1) %
              (len(RELICS) - 1)) % len(RELICS)
    potions: list[str | None] = [None, None]
    if _roll(curriculum_seed, index, "potion-count", 0) % 3 == 0:
        potions[0] = POTIONS[
            _roll(curriculum_seed, index, "potion-kind", 0) % len(POTIONS)
        ]
    return {
        "encounter_id": encounter_id, "native_encounter_entry": native_entry,
        "total_floor": floor,
        "deck": [
            {"card_id": card, "upgrade_level": level} for card, level in cards
        ],
        "relic_ids": ["proto.relic.silent_ring", RELICS[first], RELICS[second]],
        "potion_ids": potions,
        "hp": (45, 55, 65, 70)[_roll(curriculum_seed, index, "hp", 0) % 4],
        "max_hp": 70,
        "map_column": 3, "map_row": row, "act": 1, "ascension": 10,
        "distribution_id": DISTRIBUTION,
    }


def structural_key(spec: dict[str, Any]) -> str:
    """Audit key for exact structured content (not a C# CanonicalJson hash)."""
    return hashlib.sha256(json.dumps(
        spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()


def audited_corpus(count: int, *, curriculum_seed: int = 20261027) -> tuple[dict[str, Any], ...]:
    if type(count) is not int or count < 1 or count > 10000:
        raise ValueError("Invalid audited corpus size")
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for index in range(count * 20):
        candidate = fixture_for_index(index, curriculum_seed=curriculum_seed)
        if (key := structural_key(candidate)) not in seen:
            seen.add(key)
            output.append(candidate)
        if len(output) == count:
            return tuple(output)
    raise RuntimeError("Research distribution failed distinct-spec quota")


def source_bank(
    split: str, count: int, *, study_seed: int = 73127,
    round_index: int = 0, train_specs: int = 256,
    heldout_specs: int = 64,
) -> tuple[str, ...]:
    """Disjoint *configuration-level* holdout and independent Chance-Law worlds."""
    if (split not in {"train", "selection", "confirmation", "zero-shot"} or
            type(count) is not int or count < 1 or
            type(study_seed) is not int or study_seed < 0 or
            type(round_index) is not int or round_index < 0 or
            type(train_specs) is not int or train_specs < 200 or
            type(heldout_specs) is not int or heldout_specs < 1):
        raise ValueError("Invalid H8 source-bank request")
    if split != "train" and round_index:
        raise ValueError("Holdout source seeds cannot depend on training rounds")
    size = train_specs if split != "zero-shot" else heldout_specs
    offset = 0 if split != "zero-shot" else train_specs
    samples = []
    for ordinal in range(count):
        shifted = ordinal + (round_index * count if split == "train" else 0)
        index = offset + shifted % size
        replica = shifted // size
        samples.append(
            f"h8-{split}-v1-{study_seed}-{round_index}-{index}-{replica}"
        )
    return tuple(samples)


def fixture_from_source(
    source_seed: str, *, curriculum_seed: int = 20261027,
    train_specs: int = 256, heldout_specs: int = 64,
) -> tuple[CombatScenario, dict[str, Any], int]:
    match = _SOURCE.fullmatch(source_seed)
    if match is None:
        raise ValueError("Unsupported H8 source-bank identifier")
    split, _, _, index_text, _ = match.groups()
    index = int(index_text)
    if index < 0 or (index >= train_specs if split != "zero-shot" else
                     not train_specs <= index < train_specs + heldout_specs):
        raise ValueError("H8 source accesses a configuration outside its split")
    fixture = fixture_for_index(index, curriculum_seed=curriculum_seed)
    scenario = next((
        s for s in SCENARIOS.values()
        if s.ascension == 10 and s.encounter_id == fixture["encounter_id"]
    ), None)
    if scenario is None:
        raise ValueError("Unregistered source-supported research encounter")
    return scenario, fixture, index
