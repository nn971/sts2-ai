from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sts2_ai.emulator.protocol import Observation
from sts2_ai.training.h12_transition_data import PublicJointRunTrace
from sts2_ai.training.h13_public_transitions import (
    CONTEXT_VERSION,
    EmpiricalEncounterLaw,
    observed_combat_boundary,
    observed_room_encounters,
    public_choice_context,
)


def _state(*, floor=3, hp=50, combat=False, reward=False):
    state = {
        "act": 1, "floor": floor, "hp": hp, "max_hp": 72, "gold": 100,
        "deck": [{"card_id": "strike", "upgrade_level": 0}],
        "relics": [{"relic_id": "starter"}],
        "potions": [{"slot": 0, "potion_id": "poison"}],
        "map": [
            {"node_id": "SECRET-NODE", "floor": 4, "room_type": 0,
             "next_node_ids": []},
            {"node_id": "EVENT-NODE", "floor": 4, "room_type": 2,
             "next_node_ids": []},
        ],
        "terminal_outcome": None,
        "hidden_future_enemy": "SECRET-ENEMY",
    }
    if combat:
        state["combat"] = {
            "turn": 1, "energy": 3, "draw_pile_count": 6,
            "hand": [{"card_id": "strike"}],
            "enemies": [{"enemy_id": "beetle", "hp": 20}],
        }
    if reward:
        state["reward"] = {
            "card_options": ["dagger", "poisoned_stab"],
            "potion_option": "strength",
            "hidden_rewards": ["SECRET-RELIC"],
        }
    return state


def _decision(trace, state, kind, payload):
    action = SimpleNamespace(
        kind=kind, payload_json=json.dumps(payload),
        action_id="action-1",
    )
    obs = Observation(
        policy_id="fair-player-v1", payload_json=json.dumps(state),
        observation_hash="test-hash",
    )
    trace.record_decision(obs, (action,), SimpleNamespace(action=action))


def _finish(trace):
    return trace.finish(
        outcome="defeat", censored=False, act1_clear=False,
        frontier_progress=5.0, decisions=4, policy_identity={
            "macro_checkpoint_sha256": "a" * 64,
            "combat_checkpoint_sha256": "b" * 64,
        },
    )


def _sample_run():
    trace = PublicJointRunTrace("sample-0")
    choice = _state()
    trace.observe(choice)
    _decision(trace, choice, "choose_map_node", {"NodeId": "SECRET-NODE"})

    trace.observe(_state(floor=4, combat=True))
    immediate = _state(floor=4, hp=37, reward=True)
    trace.observe(immediate)
    _decision(trace, immediate, "take_reward_card", {"Index": 0})

    settled = _state(floor=4, hp=37)
    settled["gold"] = 160
    settled["deck"].append({"card_id": "dagger", "upgrade_level": 0})
    settled["potions"].append({"slot": 1, "potion_id": "strength"})
    trace.observe(settled)

    later = _state(floor=5, hp=37)
    later["gold"] = 200
    trace.observe(later)
    return _finish(trace)


def test_public_choice_records_only_known_destination_and_offers():
    state = _state(reward=True)
    context = public_choice_context(
        state, "choose_map_node", {"NodeId": "SECRET-NODE"},
    )
    assert context == {
        "map_target": {"act": 1, "floor": 4, "room_type": 0},
    }
    assert "SECRET-NODE" not in repr(context)
    assert "SECRET-ENEMY" not in repr(context)

    offers = public_choice_context(
        state, "take_reward_card", {"Index": 0},
    )
    assert offers["reward_offers"]["card_options"] == [
        "dagger", "poisoned_stab",
    ]
    assert "SECRET-RELIC" not in repr(offers)
    with pytest.raises(ValueError, match="absent or ambiguous"):
        public_choice_context(
            state, "choose_map_node", {"NodeId": "future-node"},
        )


def test_combat_and_reward_boundary_are_explicitly_distinct():
    run = _sample_run()
    assert run["public_choice_context_version"] == CONTEXT_VERSION
    boundaries = observed_combat_boundary(run, 0)
    immediate = boundaries["combat_immediate"]
    assert immediate["survived"]
    assert immediate["hp_delta"] == -13
    assert immediate["gold_delta"] == 0
    assert immediate["potions_gained_in_combat"] == []
    assert immediate["cards_added_in_combat"] == []
    assert boundaries["immediate_exit"]["gold"] == 100
    assert boundaries["postcombat_snapshot"]["gold"] == 160
    assert boundaries["postcombat_snapshot_is_policy_conditioned"]
    assert len(boundaries["reward_decisions"]) == 1
    reward = boundaries["reward_decisions"][0]
    assert reward["offers"]["card_options"] == ["dagger", "poisoned_stab"]
    assert reward["selected_kind"] == "take_reward_card"


def test_encounter_law_includes_no_combat_and_abstains_without_support():
    run = _sample_run()
    observed = observed_room_encounters(run)
    assert len(observed) == 1
    assert observed[0]["room_type"] == 0
    assert observed[0]["enemy_groups"] == [["beetle"]]
    assert "SECRET-NODE" not in repr(observed)
    assert "SECRET-ENEMY" not in repr(observed)

    quiet = PublicJointRunTrace("quiet")
    source = _state()
    quiet.observe(source)
    _decision(quiet, source, "choose_map_node", {"NodeId": "EVENT-NODE"})
    quiet.observe(_state(floor=4))
    no_combat = _finish(quiet)
    assert observed_room_encounters(no_combat)[0]["enemy_groups"] == []

    law = EmpiricalEncounterLaw(min_count=2)
    law.ingest([run, no_combat])
    assert law.distribution(act=1, room_type=0, floor=4) is None
    assert law.distribution(act=1, room_type=2, floor=4) is None
    low = EmpiricalEncounterLaw(min_count=1)
    low.ingest([run, no_combat])
    a = low.distribution(act=1, room_type=0, floor=4)
    b = low.distribution(act=1, room_type=2, floor=4)
    assert a is not None and a.count == 1
    assert a.outcomes == (((("beetle",),), 1.0),)
    assert b is not None and b.outcomes == (((), 1.0),)
    assert low.distribution(act=1, room_type=0, floor=10) is None


def test_old_h12_run_cannot_silently_supply_encounter_law():
    run = _sample_run()
    run.pop("public_choice_context_version")
    with pytest.raises(ValueError, match="old H12 journal"):
        observed_room_encounters(run)
