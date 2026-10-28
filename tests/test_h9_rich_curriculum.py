"""Structural checks for opt-in H9 rich v2 research sources."""
from __future__ import annotations

import pytest

from sts2_ai.nature.research_combat_fixtures import (
    RICH_ADDITIONS,
    RICH_DISTRIBUTION,
    RICH_ENCOUNTERS,
    RICH_POTIONS,
    RICH_RELICS,
    STARTER,
    fixture_for_index,
    fixture_from_source,
    rich_fixture_for_index,
    rich_source_bank,
    source_bank,
    structural_key,
)


def test_rich_population_has_distinct_decks_and_all_encounter_tiers() -> None:
    specs = [rich_fixture_for_index(i, curriculum_seed=20261028) for i in range(1280)]
    assert len({structural_key(x) for x in specs}) == 1280
    assert {x["encounter_id"] for x in specs} == {
        encounter[0] for encounter in RICH_ENCOUNTERS
    }
    assert {len(x["deck"]) for x in specs} >= {12, 13, 14, 17, 20}
    assert all(12 <= len(x["deck"]) <= 24 for x in specs)
    assert all(x["distribution_id"] == RICH_DISTRIBUTION for x in specs)
    assert {len(x["relic_ids"]) for x in specs} == {1, 2, 3}
    assert {sum(p is not None for p in x["potion_ids"]) for x in specs} == {0, 1, 2}
    allowed = set(STARTER) | set(RICH_ADDITIONS)
    for x in specs:
        assert x["act"] == 1 and x["ascension"] == 10
        assert len(set(x["relic_ids"])) == len(x["relic_ids"])
        assert x["relic_ids"][0] == "proto.relic.silent_ring"
        assert set(x["relic_ids"][1:]) <= set(RICH_RELICS)
        assert all(p is None or p in RICH_POTIONS for p in x["potion_ids"])
        assert all(card["card_id"] in allowed and card["upgrade_level"] in (0, 1)
                   for card in x["deck"])


def test_rich_sources_separate_worlds_and_v1_is_unchanged() -> None:
    train = rich_source_bank("train", 1024)
    selection = rich_source_bank("selection", 8)
    held = rich_source_bank("zero-shot", 256)
    assert len(set(train)) == 1024
    assert set(train).isdisjoint(selection)
    assert set(train).isdisjoint(held)
    for i in (0, 1, 200, 1023):
        scenario, spec, index = fixture_from_source(
            train[i], curriculum_seed=20261028,
        )
        assert index == i
        assert spec == rich_fixture_for_index(i, curriculum_seed=20261028)
        assert scenario.encounter_id == spec["encounter_id"]
        assert scenario.native_entry == spec["native_encounter_entry"]
    assert all(fixture_from_source(s, curriculum_seed=20261028)[2] >= 1024
               for s in held)
    assert not ({structural_key(rich_fixture_for_index(i, curriculum_seed=20261028))
                 for i in range(1024)} &
                {structural_key(rich_fixture_for_index(i, curriculum_seed=20261028))
                 for i in range(1024, 1280)})
    assert source_bank("train", 2)[0].startswith("h8-train-v1-")
    assert fixture_for_index(0)["distribution_id"] != RICH_DISTRIBUTION
    with pytest.raises(ValueError):
        fixture_from_source("h9-rich-zero-shot-v2-73127-0-25-0")
    with pytest.raises(ValueError):
        rich_source_bank("selection", 8, round_index=1)
