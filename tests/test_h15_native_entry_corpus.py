"""Native H15 combat-entry corpus planning and public configuration coverage."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.collect_h15_native_entries import (
    SCHEMA,
    atomic_write,
    consolidate,
    public_build,
    source_plan,
)


def test_native_plan_separates_runs_and_rotates_pinned_macro_policies():
    plans = source_plan(
        13, 5, seed_prefix="h15-native-test",
        macro_sha256=("a" * 64, "b" * 64),
    )
    assert len(plans) == 18
    assert len({p["seed"] for p in plans}) == 18
    assert {p["split"] for p in plans} == {"train", "heldout"}
    assert [p["macro_sha256"] for p in plans[:4]] == [
        "a" * 64, "b" * 64, "a" * 64, "b" * 64,
    ]
    assert all(p["seed"].split(":")[1] == p["split"] for p in plans)
    with pytest.raises(ValueError):
        source_plan(10, 0, seed_prefix="bad", macro_sha256=("a" * 64,))


def test_native_public_deck_hash_ignores_instance_id_and_order():
    deck = [
        {"instance_id": 99, "card_id": "proto.silent.strike", "upgrade_level": 1},
        {"instance_id": 21, "card_id": "proto.silent.defend", "upgrade_level": 0},
    ]
    observation = SimpleNamespace(payload_json=json.dumps({"deck": deck}))
    recipe = SimpleNamespace(
        floor=8, tier="elite", enemy_ids=("proto.enemy.test",),
        hp=37, max_hp=70, deck_size=2,
        relic_ids=("proto.relic.test",), potion_ids=(),
    )
    first = public_build(observation, recipe)
    observation2 = SimpleNamespace(payload_json=json.dumps({"deck": [
        {**deck[1], "instance_id": 555},
        {**deck[0], "instance_id": 777},
    ]}))
    second = public_build(observation2, recipe)
    assert first["deck_sha256"] == second["deck_sha256"]
    assert first["public_configuration_sha256"] == (
        second["public_configuration_sha256"]
    )
    assert first["hp"] == 37
    with pytest.raises(ValueError):
        public_build(observation, SimpleNamespace(**{
            **vars(recipe), "deck_size": 3,
        }))


def test_consolidation_keeps_source_run_splits_disjoint(tmp_path: Path):
    tasks = source_plan(
        1, 1, seed_prefix="h15-test", macro_sha256=("a" * 64,),
    )
    for task in tasks:
        recipe = {
            "seed": task["seed"], "emulator_revision": "c" * 40,
            "tier": "normal", "floor": 7,
            "enemy_ids": ["proto.enemy.x"],
        }
        run = {
            **task, "schema": SCHEMA, "ok": True, "outcome": "defeat",
            "censored": False, "emulator_revision": "c" * 40,
            "recipes": [recipe],
            "public_builds": [{
                "public_configuration_sha256": task["split"] + "-build",
                "deck_sha256": task["split"] + "-deck",
            }],
        }
        filename = (tmp_path / "runs" / task["split"] /
                    (task["id"] + ".json"))
        atomic_write(filename, json.dumps(run) + "\n")
    stats = consolidate(tmp_path, tasks)
    assert stats["native_combat_entries"] == {"train": 1, "heldout": 1}
    assert stats["distinct_decks"] == 2
    assert not stats["failed_or_missing_run_ids"]
    train_rows = (tmp_path / "train.jsonl").read_text().strip().splitlines()
    heldout_rows = (tmp_path / "heldout.jsonl").read_text().strip().splitlines()
    assert json.loads(train_rows[0])["seed"] == tasks[0]["seed"]
    assert json.loads(heldout_rows[0])["seed"] == tasks[1]["seed"]
