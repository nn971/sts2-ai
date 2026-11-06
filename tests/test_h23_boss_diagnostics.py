"""H23: no-training checkpoint/boss diagnostic invariants."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sts2_ai.training.h16_joint_act_pilot import (
    independent_search_seed,
    search_floor_allows,
)
from tools.compare_h22_boss_checkpoints import (
    boss_stats,
    paired_difference,
    select_for_diagnosis,
)
from tools.diagnose_h22_boss_search import (
    check_preboss_parity,
    compare_arms,
    describe_episode,
    summarize_arm,
)


def _selection_record(seed, floor, outcome, hp=70):
    return {
        "seed": seed, "outcome": outcome,
        "max_floor_reached": floor,
        "post_room_hp": (
            {"15": {"hp": hp, "max_hp": 70}} if floor == 16 else {}
        ),
    }


def test_h23_checkpoint_summary_and_paired_selection():
    rows = [
        _selection_record("a", 16, "act_clear", 60),
        _selection_record("b", 16, "defeat", 50),
        _selection_record("c", 11, "defeat"),
    ]
    summary = {
        "completed": 3, "censored": 0, "cleared": 1,
        "floor_progress": {
            "runs": rows,
            "mean_max_floor_reached": 43 / 3,
        },
        "combat_progress": {
            "lost_fights": [
                {"seed": "b", "floor": 16,
                 "enemy_hp_removed_fraction": 0.65},
                {"seed": "c", "floor": 11,
                 "enemy_hp_removed_fraction": 0.2},
            ],
        },
    }
    a = boss_stats(summary)
    assert a["boss_reached"] == 2
    assert a["cleared"] == 1
    assert a["boss_clear_rate_given_reached"] == pytest.approx(0.5)
    assert a["mean_boss_entry_hp"] == pytest.approx(55)
    assert a["mean_enemy_hp_removed_on_boss_defeat"] == pytest.approx(0.65)

    alt = {**a, "by_seed": {
        **a["by_seed"],
        "c": {"floor": 16, "boss_reached": True, "cleared": False},
    }, "boss_reached": 3}
    paired = paired_difference(a, alt)
    assert paired["floor_better_right"] == 1
    assert paired["boss_reached_only_right"] == 1
    assert paired["act_clear_only_right"] == 0
    assert select_for_diagnosis({"r0095": a, "r0120": alt}) == "r0120"

    broken = {**summary, "cleared": 2}
    with pytest.raises(ValueError, match="disagree"):
        boss_stats(broken)
    with pytest.raises(ValueError, match="nonmatching"):
        paired_difference(a, {**a, "by_seed": {"different": {}}})


def _episode(boss_action="attack", boss_hp=40, searches=0):
    first = {"floor": 15, "hp": 70, "max_hp": 70}
    boss = {
        "floor": 16, "hp": 70, "max_hp": 70,
        "combat": {"enemies": [{
            "instance_id": 1, "enemy_id": "test.boss",
            "hp": 100, "max_hp": 100,
        }]},
    }
    last = {
        "floor": 16, "hp": 0, "max_hp": 70,
        "combat": {"enemies": [{
            "instance_id": 1, "enemy_id": "test.boss",
            "hp": boss_hp, "max_hp": 100,
        }]},
    }
    def decision(action):
        return SimpleNamespace(
            legal_actions=[SimpleNamespace(action_id=action)], chosen_index=0,
        )
    return SimpleNamespace(
        seed="fixed", frames=(first, boss, last),
        decisions=(decision("preboss-map"), decision(boss_action)),
        outcome="defeat", complete=True,
        searches=searches,
        search_transitions=40 * searches, search_overrides=searches,
    )


def test_h23_boss_entry_pairing_and_public_prefix_parity():
    control = describe_episode(_episode())
    searched = describe_episode(_episode("attack-b", boss_hp=10, searches=1))
    assert control["reached_boss"]
    assert control["preboss_action_count"] == 1
    assert control["boss_damage_fraction"] == pytest.approx(0.6)
    assert searched["boss_damage_fraction"] == pytest.approx(0.9)
    check_preboss_parity([control], [searched])
    paired = compare_arms([control], [searched])
    assert paired["boss_action_changed"] == 1
    assert paired["both_lost_boss_damage_delta_right_minus_left"] == pytest.approx(0.3)
    assert summarize_arm([control])["boss_reached"] == 1
    assert summarize_arm([searched])["search_roots"] == 1

    changed = dict(searched)
    changed["boss_entry_public_hash"] = "tampered"
    with pytest.raises(ValueError, match="Preboss parity"):
        check_preboss_parity([control], [changed])


def test_h23_boss_floor_filter_is_opt_in_and_pre_rng():
    assert search_floor_allows({"floor": 1, "combat": {}}, None)
    assert not search_floor_allows({"floor": 1, "combat": {}}, 16)
    assert search_floor_allows({"floor": 16, "combat": {}}, 16)
    assert not search_floor_allows({"floor": 15}, 16)

def test_h23_independent_search_seed_is_stable_and_run_local():
    fixed = independent_search_seed("run-42", 17)
    assert 0 <= fixed < 2**63
    assert fixed == independent_search_seed("run-42", 17)
    assert fixed != independent_search_seed("run-42", 18)
    assert fixed != independent_search_seed("other-run", 17)
    with pytest.raises(ValueError, match="Invalid independent"):
        independent_search_seed("run-42", -1)
    with pytest.raises(ValueError, match="Invalid independent"):
        independent_search_seed("", 0)
