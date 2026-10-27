"""H8 structural source separation; no emulator, GPU or native fidelity needed."""
from __future__ import annotations

import pytest

from sts2_ai.nature.research_combat_fixtures import (
    ADDITIONS,
    DISTRIBUTION,
    audited_corpus,
    fixture_for_index,
    fixture_from_source,
    source_bank,
    structural_key,
)


def test_1000_are_distinct_structural_configurations() -> None:
    corpus = audited_corpus(1000)
    assert len(corpus) == 1000
    assert len({structural_key(c) for c in corpus}) == 1000
    assert {len(c["deck"]) for c in corpus} == {20}
    assert all(c["distribution_id"] == DISTRIBUTION for c in corpus)
    assert all(c["ascension"] == 10 and c["act"] == 1 for c in corpus)
    assert all(c["relic_ids"][0] == "proto.relic.silent_ring" for c in corpus)
    assert all(1 <= c["hp"] <= c["max_hp"] == 70 for c in corpus)
    assert {c["encounter_id"] for c in corpus} >= {
        "proto.encounter.mawler_normal",
        "proto.encounter.ceremonial_beast_boss",
    }


def test_generator_matches_explicit_audited_card_and_relic_rules() -> None:
    for index in range(250):
        spec = fixture_for_index(index)
        assert spec == fixture_for_index(index)
        assert len(spec["deck"]) == 20
        assert len(spec["relic_ids"]) == 3
        assert len(set(spec["relic_ids"])) == 3
        assert all(x["card_id"].startswith("proto.silent.") for x in spec["deck"])
        assert all(x["upgrade_level"] in (0, 1) for x in spec["deck"])
        assert sum(x["card_id"] in ADDITIONS for x in spec["deck"]) >= 8
        assert spec["deck"] == sorted(
            spec["deck"], key=lambda c: (c["card_id"], c["upgrade_level"]),
        )


def test_split_uses_disjoint_structural_indices_and_independent_source_worlds() -> None:
    training = source_bank("train", 256)
    confirmation = source_bank("confirmation", 256)
    zero_shot = source_bank("zero-shot", 64)
    assert len(set(training)) == 256
    assert len(set(confirmation)) == 256
    assert set(training).isdisjoint(confirmation)
    assert set(training).isdisjoint(zero_shot)
    train_specs = [fixture_from_source(s)[1] for s in training]
    confirmation_specs = [fixture_from_source(s)[1] for s in confirmation]
    unseen_specs = [fixture_from_source(s)[1] for s in zero_shot]
    assert train_specs == confirmation_specs
    assert not {structural_key(c) for c in train_specs} & {
        structural_key(c) for c in unseen_specs
    }
    assert [fixture_from_source(s)[2] for s in training] == list(range(256))
    assert [fixture_from_source(s)[2] for s in zero_shot] == list(range(256, 320))


def test_round_rotation_and_invalid_split_source() -> None:
    assert source_bank("train", 16, round_index=1) != source_bank("train", 16)
    assert source_bank("selection", 16) == source_bank("selection", 16)
    with pytest.raises(ValueError):
        source_bank("selection", 5, round_index=1)
    with pytest.raises(ValueError):
        fixture_from_source("h8-train-v1-1-0-256-0")
    with pytest.raises(ValueError):
        fixture_from_source("h8-zero-shot-v1-1-0-17-0")
    with pytest.raises(ValueError):
        fixture_for_index(-1)
