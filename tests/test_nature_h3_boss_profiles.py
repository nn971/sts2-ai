"""Act 1 A10 boss challenge is a research fixture, not the native route."""
from __future__ import annotations

from sts2_ai.nature.combat_scenarios import (
    DECK_PRESETS,
    SCENARIOS,
    deck_for_seed,
    parse_deck_pool,
    parse_scenario_pool,
    scenario_for_seed,
)
from sts2_ai.nature.h2_benchmark import balanced_h2_source_seeds


BOSSES = (
    "ceremonial-beast-boss-a10",
    "vantom-boss-a10",
)
BOSS_DECKS = (
    "silent-boss-cycle-20-v1",
    "silent-boss-poison-20-v1",
)


def test_boss_research_context_is_pinned_to_prototype_act1_final_room():
    for key in BOSSES:
        context = SCENARIOS[key]
        assert context.act == 1
        assert context.ascension == 10
        assert context.total_floor == 6
        assert context.map_column == 3
        assert context.map_row == 5
        assert context.encounter_id.endswith("_boss")
    assert set(BOSS_DECKS).issubset(DECK_PRESETS)


def test_boss_source_strata_use_balanced_disjoint_holdout_worlds():
    scenarios = parse_scenario_pool(",".join(BOSSES))
    decks = parse_deck_pool(",".join(BOSS_DECKS))
    seeds = balanced_h2_source_seeds(
        "confirmation", 40, scenarios, decks,
    )
    counts = {
        (scenario, deck): 0 for scenario in scenarios for deck in decks
    }
    for seed in seeds:
        scenario = scenario_for_seed(seed, scenarios).scenario_id
        deck = deck_for_seed(seed, decks)
        counts[(scenario, deck)] += 1
    assert set(counts.values()) == {10}
    assert set(seeds).isdisjoint(set(balanced_h2_source_seeds(
        "selection", 40, scenarios, decks,
    )))
