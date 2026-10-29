"""Contract-level tests for H14's public observation and combat exit."""
from __future__ import annotations

from copy import deepcopy

import pytest

from sts2_ai.training.h14_boundaries import (
    STATE_SCHEMA,
    TRANSITION_SCHEMA,
    H14CombatBoundaryTrace,
    public_persistent_state,
)


def state(*, hp=60, combat=False, reward=False):
    observation = {
        "act": 1, "floor": 4, "hp": hp, "max_hp": 72, "gold": 92,
        "character_id": "silent", "terminal_outcome": None,
        "seed": "HIDDEN-RUN-SEED",
        "future_enemy": "HIDDEN-ENEMY",
        "rng_cursor": "HIDDEN-RNG",
        "deck": [
            {"instance_id": 88, "card_id": "strike", "upgrade_level": 0,
             "state": {"secret_draw_position": 9}},
        ],
        "relics": [
            {"relic_id": "proto.relic.pen_nib",
             "state": {"AttacksPlayed": 9, "secret_key": "HIDDEN-RELIC"}},
            {"relic_id": "proto.relic.pendulum",
             "state": {"TurnsSeen": 2}},
        ],
        "potions": [
            {"slot": 0, "potion_id": "proto.potion.poison",
             "state": {"private_content": "HIDDEN-POTION"}},
            {"slot": 1, "potion_id": "proto.potion.poison",
             "state": {}},
        ],
        "map": [
            {"node_id": "HIDDEN-ID", "room_type": 0, "floor": 5,
             "next_node_ids": ["HIDDEN-CHILD"]},
        ],
    }
    if combat:
        observation["combat"] = {
            "turn": 2, "energy": 3, "player_block": 4,
            "draw_pile_count": 6,
            "hand": [{"card_id": "strike", "upgrade_level": 0}],
            "discard_pile": [], "exhaust_pile": [], "play_pile": [],
            "enemies": [{
                "enemy_id": "visible-beetle", "hp": 20, "block": 0,
                "move_id": "visible-attack", "intent_damage": 7,
                "statuses": {"poison": 3}, "powers": [],
                "hidden_intent": "HIDDEN-INTENT",
            }],
            "player_powers": [],
            "rng_state": "HIDDEN-COMBAT-RNG",
            "draw_pile_order": "HIDDEN-DRAW-ORDER",
        }
    if reward:
        observation["reward"] = {
            "card_options": ["dagger", "poisoned_stab"],
            "relic_option": "proto.relic.starter",
            "extra_card_reward_groups_remaining": 1,
            "pending_card_groups": [
                {"group_index": 0, "card_options": ["shiv", "slice"],
                 "hidden_roll": "HIDDEN-CARD-ROLL"},
            ],
            "hidden_rewards": ["HIDDEN-OFFER"],
        }
    return observation


def test_public_state_only_known_fields_and_persistent_counters():
    projected = public_persistent_state(state(reward=True))
    assert projected["schema"] == STATE_SCHEMA
    assert projected["relics"][0]["counter"] == {
        "name": "AttacksPlayed", "residue": 9, "modulus": 10,
    }
    assert projected["relics"][1]["counter"]["residue"] == 2
    assert projected["potions"] == [
        {"slot": 0, "potion_id": "proto.potion.poison"},
        {"slot": 1, "potion_id": "proto.potion.poison"},
    ]
    assert projected["visible_map_room_counts"] == [
        {"floor": 5, "room_type": "0", "count": 1},
    ]
    assert projected["reward_offers"]["card_options"] == [
        "dagger", "poisoned_stab",
    ]
    forbidden = (
        "HIDDEN-", "instance_id", "rng_cursor", "private_content",
        "secret_key", "secret_draw_position", "hidden_roll",
    )
    for token in forbidden:
        assert token not in repr(projected)


def test_combat_exit_freezes_before_reward_selection_and_logs_potions():
    recorder = H14CombatBoundaryTrace(
        source_run_seed="trace-1", combat_policy_id="h9-abc",
    )
    entry = state(combat=True)
    recorder.observe(entry)
    assert recorder._pending is not None
    combat_entry = recorder._pending["entry"]["combat"]
    assert combat_entry["hand"][0]["card_id"] == "strike"
    assert combat_entry["enemies"][0]["intent_damage"] == 7
    assert "HIDDEN-" not in repr(combat_entry)
    recorder.record_committed_action(
        entry, kind="use_potion", payload={"Slot": 1, "TargetEnemyId": 2},
    )
    later = deepcopy(entry)
    later["hp"] = 45
    later["potions"] = [later["potions"][0]]
    later["relics"][0]["state"]["AttacksPlayed"] = 0
    recorder.observe(later)
    exit_observation = state(hp=45, reward=True)
    exit_observation["potions"] = [entry["potions"][0]]
    exit_observation["relics"][0]["state"]["AttacksPlayed"] = 0
    recorder.observe(exit_observation)

    # Subsequent macro pickup is outside the option's transition boundary.
    selected = deepcopy(exit_observation)
    selected["reward"] = None
    selected["deck"].append(
        {"instance_id": 1234, "card_id": "dagger", "upgrade_level": 0})
    selected["gold"] = 180
    recorder.observe(selected)

    result = recorder.finish(censored=False)
    assert len(result) == 1
    combat = result[0]
    assert combat["schema"] == TRANSITION_SCHEMA
    assert combat["boundary"] == "reward_decision"
    assert combat["survived"] is True
    assert combat["hp_delta"] == -15
    assert combat["gold_delta_observed"] == 0
    assert combat["combat_decisions"] == 1
    assert combat["potion_uses"] == [
        {"slot": 1, "potion_id": "proto.potion.poison"}]
    assert combat["potion_inventory_delta"] == {
        "removed": [{"potion_id": "proto.potion.poison"}], "added": []}
    assert combat["deck_delta"] == {"removed": [], "added": []}
    assert combat["exit"]["relics"][0]["counter"]["residue"] == 0
    assert combat["exit"]["reward_offers"]["card_options"] == [
        "dagger", "poisoned_stab"]
    assert combat["effect_receipts"] == [{
        "kind": "extra_card_reward_groups_visible", "count": 1,
        "source": "unattributed_public_reward",
    }]
    assert "dagger" not in [c["card_id"] for c in combat["exit"]["deck"]]


def test_defeat_in_combat_closes_and_censored_combat_is_unlabeled():
    trace = H14CombatBoundaryTrace(
        source_run_seed="defeat-1", combat_policy_id="fixed",
    )
    trace.observe(state(combat=True))
    dead = state(hp=0, combat=True)
    dead["terminal_outcome"] = "defeat"
    trace.observe(dead)
    combat = trace.finish(censored=False)[0]
    assert combat["boundary"] == "defeat"
    assert combat["survived"] is False
    assert combat["effect_receipts"] == []

    censored = H14CombatBoundaryTrace(
        source_run_seed="cut", combat_policy_id="fixed")
    censored.observe(state(combat=True))
    assert censored.finish(censored=True) == []
    open_trace = H14CombatBoundaryTrace(
        source_run_seed="incomplete", combat_policy_id="fixed")
    open_trace.observe(state(combat=True))
    with pytest.raises(ValueError, match="Unclosed combat"):
        open_trace.finish(censored=False)


def test_invalid_public_counter_and_potion_slots_fail_closed():
    obs = state()
    obs["relics"][0]["state"]["AttacksPlayed"] = "9"
    with pytest.raises(ValueError, match="AttacksPlayed"):
        public_persistent_state(obs)
    obs = state()
    obs["potions"][1]["slot"] = 0
    with pytest.raises(ValueError, match="potion slots"):
        public_persistent_state(obs)
    trace = H14CombatBoundaryTrace(
        source_run_seed="x", combat_policy_id="y")
    entry = state(combat=True)
    trace.observe(entry)
    with pytest.raises(ValueError, match="visible held potion"):
        trace.record_committed_action(
            entry, kind="use_potion", payload={"Slot": 3},
        )


def test_hunt_receipts_distinguish_causal_bonus_from_other_offers():
    trace = H14CombatBoundaryTrace(
        source_run_seed="hunt-seed", combat_policy_id="h9-fixed",
    )
    trace.observe(state(combat=True))
    exit_state = state(hp=59, reward=True)
    exit_state["reward"]["extra_card_reward_groups_remaining"] = 2
    exit_state["reward"]["combat_effect_receipts"] = [{
        "kind": "extra_card_reward_earned",
        "source_card_id": "proto.silent.the_hunt",
        "count": 1,
        "hidden_rng": "HIDDEN-ROLL",
    }]
    trace.observe(exit_state)
    result = trace.finish(censored=False)[0]
    assert result["effect_receipts"] == [
        {
            "kind": "extra_card_reward_earned",
            "source_card_id": "proto.silent.the_hunt",
            "count": 1,
            "source": "emulator_combat_effect",
        },
        {
            "kind": "extra_card_reward_groups_visible",
            "count": 1,
            "source": "unattributed_public_reward",
        },
    ]
    assert "HIDDEN-ROLL" not in repr(result)

    bad = deepcopy(exit_state)
    bad["reward"]["extra_card_reward_groups_remaining"] = 0
    bad_trace = H14CombatBoundaryTrace(
        source_run_seed="bad-receipt", combat_policy_id="h9-fixed",
    )
    bad_trace.observe(state(combat=True))
    with pytest.raises(ValueError, match="exceed visible"):
        bad_trace.observe(bad)
