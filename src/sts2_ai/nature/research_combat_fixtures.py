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
import threading
from copy import deepcopy
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
    if _RICH_SOURCE.fullmatch(source_seed):
        return rich_fixture_from_source(
            source_seed, curriculum_seed=curriculum_seed,
            train_specs=1024, heldout_specs=256,
        )
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

# A second distribution for H9 continuation. The original v1 corpus, its
# source namespaces, fixture hashes and sealed H9 evaluations are immutable.
RICH_DISTRIBUTION = "silent-a10-overgrowth-rich-v2"
_RICH_SOURCE = re.compile(
    r"^h9-rich-(train|selection|confirmation|zero-shot)-v2-"
    r"(\d+)-(\d+)-(\d+)-(\d+)$"
)
RICH_ENCOUNTERS = (
    ("proto.encounter.shrinker_beetle_weak", "ShrinkerBeetleWeak", 2, 1),
    ("proto.encounter.nibbits_weak", "NibbitsWeak", 2, 1),
    ("proto.encounter.fuzzy_wurm_crawler_weak", "FuzzyWurmCrawlerWeak", 2, 1),
    ("proto.encounter.slimes_weak", "SlimesWeak", 2, 1),
    ("proto.encounter.mawler_normal", "MawlerNormal", 2, 1),
    ("proto.encounter.cubex_construct_normal", "CubexConstructNormal", 2, 1),
    ("proto.encounter.vine_shambler_normal", "VineShamblerNormal", 2, 1),
    ("proto.encounter.flyconid_normal", "FlyconidNormal", 2, 1),
    ("proto.encounter.fogmog_normal", "FogmogNormal", 2, 1),
    ("proto.encounter.inklets_normal", "InkletsNormal", 2, 1),
    ("proto.encounter.overgrowth_crawlers", "OvergrowthCrawlers", 2, 1),
    ("proto.encounter.ruby_raiders_normal", "RubyRaidersNormal", 2, 1),
    ("proto.encounter.slimes_normal", "SlimesNormal", 2, 1),
    ("proto.encounter.slithering_strangler_normal", "SlitheringStranglerNormal", 2, 1),
    ("proto.encounter.snapping_jaxfruit_normal", "SnappingJaxfruitNormal", 2, 1),
    ("proto.encounter.bygone_effigy_elite", "BygoneEffigyElite", 3, 2),
    ("proto.encounter.byrdonis_elite", "ByrdonisElite", 3, 2),
    ("proto.encounter.phrog_parasite_elite", "PhrogParasiteElite", 3, 2),
    ("proto.encounter.ceremonial_beast_boss", "CeremonialBeastBoss", 6, 5),
    ("proto.encounter.vantom_boss", "VantomBoss", 6, 5),
    ("proto.encounter.the_kin_boss", "TheKinBoss", 6, 5),
)
# Index ranges preserve deterministic 30/40/15/15% tier allocation.
RICH_TIERS = (
    RICH_ENCOUNTERS[:4], RICH_ENCOUNTERS[4:15],
    RICH_ENCOUNTERS[15:18], RICH_ENCOUNTERS[18:],
)
RICH_ADDITIONS = ADDITIONS + (
    "proto.silent.blade_dance", "proto.silent.dagger_spray",
    "proto.silent.quick_slash", "proto.silent.dagger_throw",
    "proto.silent.cloak_and_dagger", "proto.silent.deflect",
    "proto.silent.sucker_punch", "proto.silent.dash",
    "proto.silent.leg_sweep", "proto.silent.skewer",
    "proto.silent.footwork", "proto.silent.noxious_fumes",
    "proto.silent.prepared", "proto.silent.piercing_wail",
    "proto.silent.backstab", "proto.silent.adrenaline",
    "proto.silent.afterimage", "proto.silent.bouncing_flask",
    "proto.silent.calculated_gamble", "proto.silent.predator",
    "proto.silent.blur",
)
RICH_ARCHETYPES = (
    ("proto.silent.blade_dance", "proto.silent.cloak_and_dagger",
     "proto.silent.dagger_spray", "proto.silent.slice"),
    ("proto.silent.poisoned_stab", "proto.silent.deadly_poison",
     "proto.silent.catalyst", "proto.silent.noxious_fumes",
     "proto.silent.bouncing_flask"),
    ("proto.silent.backflip", "proto.silent.deflect", "proto.silent.footwork",
     "proto.silent.blur", "proto.silent.leg_sweep"),
    ("proto.silent.quick_slash", "proto.silent.dagger_throw",
     "proto.silent.predator", "proto.silent.dash", "proto.silent.skewer"),
    ("proto.silent.acrobatics", "proto.silent.prepared",
     "proto.silent.calculated_gamble", "proto.silent.adrenaline"),
    ("proto.silent.piercing_wail", "proto.silent.sucker_punch",
     "proto.silent.afterimage", "proto.silent.backstab"),
)
RICH_RELICS = RELICS + (
    "proto.relic.bag_of_preparation", "proto.relic.anchor",
    "proto.relic.bag_of_marbles", "proto.relic.nunchaku",
    "proto.relic.bronze_scales", "proto.relic.snecko_skull",
    "proto.relic.kunai", "proto.relic.shuriken",
)
RICH_POTIONS = POTIONS + (
    "proto.potion.swift", "proto.potion.weak", "proto.potion.dexterity",
    "proto.potion.energy", "proto.potion.strength", "proto.potion.explosive",
    "proto.potion.fairy_in_a_bottle",
)


def _rich_roll(curriculum_seed: int, index: int, domain: str, ordinal: int) -> int:
    return int.from_bytes(hashlib.sha256(
        f"{RICH_DISTRIBUTION}:{curriculum_seed}:{index}:{domain}:{ordinal}"
        .encode()
    ).digest()[:8], "big")


# Source indices denote distinct structures, not merely independent RNG labels.
# Repeated starter-like loadouts get a deterministic replacement in the SAME tier.
_RICH_SPECS: dict[int, list[dict[str, Any]]] = {}
_RICH_KEYS: dict[int, set[str]] = {}
_RICH_LOCK = threading.Lock()


def _rich_fixture_candidate(
    index: int, *, curriculum_seed: int, attempt: int,
) -> dict[str, Any]:
    draw_index = index if attempt == 0 else index + attempt * 1_000_003

    def roll(domain: str, ordinal: int = 0) -> int:
        return _rich_roll(curriculum_seed, draw_index, domain, ordinal)

    bucket = index % 20
    tier = 0 if bucket < 6 else 1 if bucket < 14 else 2 if bucket < 17 else 3
    pool = RICH_TIERS[tier]
    encounter_id, native_entry, floor, row = pool[
        roll("encounter", 0) % len(pool)
    ]
    mode = roll("deck-mode", 0) % 4
    archetype = roll("archetype", 0) % len(RICH_ARCHETYPES)
    strikes = 5 - (0, 1, 2, 3)[mode]
    defends = 5 - (0, 0, 1, 2)[mode]
    cards = [("proto.silent.strike", 0)] * strikes + [
        ("proto.silent.defend", 0)
    ] * defends + [("proto.silent.neutralize", 0), ("proto.silent.survivor", 0)]
    amount = (0, 5, 9, 13)[mode] + (
        roll("deck-size", 0)
        % (3, 3, 4, 5)[mode]
    )
    for ordinal in range(amount):
        pool = (RICH_ARCHETYPES[archetype]
                if roll("archetype-choice", ordinal) % 4
                else RICH_ADDITIONS)
        card_id = pool[roll("card", ordinal) % len(pool)]
        level = int(roll("upgrade", ordinal) % 4 == 0)
        cards.append((card_id, level))
    cards.sort()
    extras = roll("relic-count", 0) % 3
    relics = ["proto.relic.silent_ring"]
    if extras:
        first = roll("relic", 0) % len(RICH_RELICS)
        relics.append(RICH_RELICS[first])
    if extras == 2:
        second = (first + 1 + roll("relic", 1)
                  % (len(RICH_RELICS) - 1)) % len(RICH_RELICS)
        relics.append(RICH_RELICS[second])
    potions: list[str | None] = [None, None]
    for ordinal in range(roll("potion-count", 0) % 3):
        potions[ordinal] = RICH_POTIONS[
            roll("potion-kind", ordinal) % len(RICH_POTIONS)
        ]
    return {
        "encounter_id": encounter_id, "native_encounter_entry": native_entry,
        "total_floor": floor, "map_column": 3, "map_row": row,
        "deck": [{"card_id": card, "upgrade_level": level} for card, level in cards],
        "relic_ids": relics, "potion_ids": potions,
        "hp": ((20, 30, 40, 50, 60, 70) if tier == 0
               else (30, 40, 50, 60, 70) if tier == 1
               else (40, 50, 60, 70) if tier == 2
               else (50, 60, 70))[
            roll("hp", 0)
            % (6 if tier == 0 else 5 if tier == 1 else 4 if tier == 2 else 3)
        ],
        "max_hp": 70, "act": 1, "ascension": 10,
        "distribution_id": RICH_DISTRIBUTION,
    }


def rich_fixture_for_index(
    index: int, *, curriculum_seed: int = 20261028,
) -> dict[str, Any]:
    """One distinct deterministic rich fixture per structural source index.

    If a structural candidate collides with an earlier one, resample at the
    same ordinal and tier. A process-local prefix cache gives stable results
    even when indices are requested out of order.
    """
    if (type(index) is not int or index < 0
            or type(curriculum_seed) is not int or curriculum_seed < 0):
        raise ValueError("Invalid rich research fixture index/seed")
    with _RICH_LOCK:
        specs = _RICH_SPECS.setdefault(curriculum_seed, [])
        seen = _RICH_KEYS.setdefault(curriculum_seed, set())
        while len(specs) <= index:
            ordinal = len(specs)
            for attempt in range(100):
                candidate = _rich_fixture_candidate(
                    ordinal, curriculum_seed=curriculum_seed,
                    attempt=attempt,
                )
                key = structural_key(candidate)
                if key not in seen:
                    specs.append(candidate)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("Rich curriculum exhausted unique fixtures")
        return deepcopy(specs[index])


def rich_source_bank(
    split: str, count: int, *, study_seed: int = 73127,
    round_index: int = 0, train_specs: int = 1024,
    heldout_specs: int = 256,
) -> tuple[str, ...]:
    """Isolated v2 source namespace with disjoint structural zero-shot indices."""
    if (split not in {"train", "selection", "confirmation", "zero-shot"}
            or type(count) is not int or count < 1
            or type(study_seed) is not int or study_seed < 0
            or type(round_index) is not int or round_index < 0
            or type(train_specs) is not int or train_specs < 200
            or type(heldout_specs) is not int or heldout_specs < 1):
        raise ValueError("Invalid rich source bank")
    if split != "train" and round_index:
        raise ValueError("Holdout source seeds cannot depend on training rounds")
    size = train_specs if split != "zero-shot" else heldout_specs
    offset = 0 if split != "zero-shot" else train_specs
    output = []
    for ordinal in range(count):
        shifted = ordinal + (round_index * count if split == "train" else 0)
        index = offset + shifted % size
        replica = shifted // size
        output.append(
            f"h9-rich-{split}-v2-{study_seed}-{round_index}-{index}-{replica}"
        )
    return tuple(output)


def rich_fixture_from_source(
    source_seed: str, *, curriculum_seed: int = 20261028,
    train_specs: int = 1024, heldout_specs: int = 256,
) -> tuple[CombatScenario, dict[str, Any], int]:
    match = _RICH_SOURCE.fullmatch(source_seed)
    if match is None:
        raise ValueError("Unsupported rich research source identifier")
    split, _, _, index_text, _ = match.groups()
    index = int(index_text)
    if (index < 0 or (index >= train_specs if split != "zero-shot"
                      else not train_specs <= index < train_specs + heldout_specs)):
        raise ValueError("Rich source accesses configuration outside its split")
    fixture = rich_fixture_for_index(index, curriculum_seed=curriculum_seed)
    scenario = CombatScenario(
        "rich-" + fixture["encounter_id"], fixture["encounter_id"],
        fixture["native_encounter_entry"], 10,
        total_floor=fixture["total_floor"], map_row=fixture["map_row"],
    )
    return scenario, fixture, index
