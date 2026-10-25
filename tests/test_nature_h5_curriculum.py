"""H5 source-bank tests: fixture support, split isolation and exact balance."""
import pytest

from sts2_ai.nature.combat_scenarios import deck_for_seed, scenario_for_seed
from sts2_ai.nature.h5_curriculum import (
    DECK_POOL,
    SCENARIO_POOL,
    SUPPORTED_CELLS,
    TRAIN_CELLS,
    ZERO_SHOT_CELLS,
    cell_for_source,
    source_bank,
    source_cells,
)


def test_catalog_has_no_cross_difficulty_combinations():
    assert len(SUPPORTED_CELLS) == 19
    assert len(TRAIN_CELLS) == 16
    assert len(ZERO_SHOT_CELLS) == 3
    assert not {x.key for x in TRAIN_CELLS} & {
        x.key for x in ZERO_SHOT_CELLS
    }
    assert {x.tier for x in TRAIN_CELLS} == {"weak", "normal", "boss"}


@pytest.mark.parametrize("split,count,cell_count", [
    ("train", 256, 16),
    ("selection", 160, 16),
    ("confirmation", 160, 16),
    ("zero-shot", 120, 3),
])
def test_source_bank_balanced_and_emulator_compatible(split, count, cell_count):
    seeds = source_bank(split, count)
    assert len(seeds) == len(set(seeds)) == count
    assert seeds == source_bank(split, count)
    per_cell = source_cells(seeds)
    assert len(per_cell) == cell_count
    assert max(per_cell.values()) - min(per_cell.values()) <= 1
    for seed in seeds:
        chosen = cell_for_source(seed)
        assert chosen is not None
        assert scenario_for_seed(seed, SCENARIO_POOL).scenario_id == chosen.scenario
        assert deck_for_seed(seed, DECK_POOL) == chosen.deck


def test_training_bank_is_round_disjoint_and_remainder_rotates():
    first = source_bank("train", 259, round_index=0)
    next_round = source_bank("train", 259, round_index=1)
    assert not set(first) & set(next_round)
    assert source_cells(first) != source_cells(next_round)


def test_all_splits_are_disjoint_and_heldout_combinations_are_unseen():
    splits = {
        split: source_bank(split, count)
        for split, count in [
            ("train", 256), ("selection", 160),
            ("confirmation", 160), ("zero-shot", 120),
        ]
    }
    for a, aa in splits.items():
        for b, bb in splits.items():
            if a != b:
                assert set(aa).isdisjoint(bb)
    assert {cell_for_source(s).key for s in splits["zero-shot"]}.isdisjoint(
        {cell_for_source(s).key for s in splits["train"]}
    )


@pytest.mark.parametrize("args", [
    ("unknown", 10), ("selection", 10), ("confirmation", 17),
    ("zero-shot", 2), ("train", 0),
])
def test_source_bank_rejects_invalid_requests(args):
    with pytest.raises(ValueError):
        source_bank(*args)


def test_confirmation_seed_independent_of_round():
    with pytest.raises(ValueError):
        source_bank("confirmation", 160, round_index=1)
