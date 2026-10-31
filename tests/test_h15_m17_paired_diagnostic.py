"""M17 R10/R30 paired-root diagnostic pure and replay provenance contracts."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from sts2_ai.emulator.protocol import LegalAction, Observation, Transition
from sts2_ai.emulator.jsonl_backend import FAIR_POLICY_ID
from tools.diagnose_h15_m17_r10_r30 import (
    SCHEMA,
    action_descriptor,
    choose_roots,
    choose_search_cases,
    jensen_shannon,
    make_report,
    paired_prediction,
    probabilities,
    replay_selected_root,
)

MENU = (
    LegalAction("block", "end_turn", "{}"),
    LegalAction("attack", "end_turn", "{}"),
)


class FixedModel:
    def __init__(self, logits, value):
        self.logits, self.value = logits, value

    def predict(self, observation, legal_actions):
        return SimpleNamespace(
            logits=self.logits,
            value=self.value, unclipped_value=self.value,
        )


def test_public_state_predictions_compare_same_menu_and_card_details():
    state = {
        "hp": 22,
        "combat": {"turn": 10, "enemies": [{"hp": 11}],
                   "hand": [{"instance_id": 7, "card_id": "proto.silent.slice"}]},
    }
    menu = (
        LegalAction("block", "end_turn", "{}"),
        LegalAction("attack", "play_card", '{"CardInstanceId":7}'),
    )
    obs = Observation(FAIR_POLICY_ID, json.dumps(state), "public-state")
    both = {
        "r10": FixedModel((0.0, 3.0), .65),
        "r30": FixedModel((3.0, 0.0), .35),
    }
    row = paired_prediction(both, obs, menu)
    assert row["raw_disagreement"]
    assert row["r10_raw"]["card_id"] == "proto.silent.slice"
    assert row["r30_raw"]["kind"] == "end_turn"
    assert row["legal_action_ids"] == ["block", "attack"]
    assert row["policy_js_nats"] > .4
    assert row["policy_total_variation"] > .8
    assert row["r30_value"] - row["r10_value"] == pytest.approx(-.3)
    assert jensen_shannon((.5, .5), (.5, .5)) == pytest.approx(0)
    assert probabilities((999.0, 1000.0))[1] > .7


def make_decision(index, turn=7, hp=25, enemy=30, *,
                  disagreement=False, js=0.01):
    return {
        "decision_index": index,
        "observation_hash": f"hash:{index}",
        "legal_action_count": 2,
        "legal_action_ids": ["block", "attack"],
        "nested_choice": False,
        "turn": turn,
        "player_hp": hp,
        "min_enemy_hp": enemy,
        "raw_disagreement": disagreement,
        "policy_js_nats": js,
        "r10_value": .5,
        "r30_value": .3,
        "source_action_id": "attack",
    }


def case(case_id, a, b, decisions):
    return {"case_id": case_id, "trajectories": {
        "r10": {"outcome": a, "decisions": list(decisions)},
        "r30": {"outcome": b, "decisions": list(decisions)},
    }}


def test_first_shared_divergence_selected_before_max_shift_and_control():
    rows = [
        make_decision(0, turn=1),
        make_decision(1, turn=8, disagreement=True, js=.04),
        make_decision(2, turn=9, enemy=11, disagreement=True, js=.30),
        make_decision(3, turn=10, enemy=8, disagreement=True, js=.20),
    ]
    selected = choose_roots(case("c", "victory", "defeat", rows),
                            max_roots=3)
    assert [x["decision_index"] for x in selected] == [1, 2, 3]
    assert selected[0]["reason"] == "first_policy_disagreement"
    assert selected[1]["reason"] == "near_lethal_disagreement"
    assert len({(x["source_trajectory"], x["decision_index"])
                for x in selected}) == 3
    assert choose_roots(case("c", "victory", "defeat", rows),
                        max_roots=0) == []


def test_search_case_selection_prioritizes_regressed_wins():
    empty = []
    cs = [
        case("stable", "victory", "victory", empty),
        case("recovered", "defeat", "victory", empty),
        case("lost", "victory", "defeat", empty),
    ]
    assert choose_search_cases(cs, 2) == ["lost", "recovered"]


class ReplayBackend:
    def __init__(self):
        self.states = {}
        self.next_id = 0
        self.released = []

    def begin_reference(self, nature_seed):
        assert nature_seed == 123
        self.next_id += 1
        name = f"handle-{self.next_id}"
        self.states[name] = 0
        return name

    def observe(self, state, policy):
        i = self.states[state]
        return Observation(FAIR_POLICY_ID, json.dumps({
            "hp": 25, "combat": {"turn": 7, "enemies": [{"hp": 20}]},
        }), f"h{i}")

    def legal_actions(self, state):
        return MENU

    def step(self, state, action):
        assert action in MENU
        self.states[state] += 1
        return Transition(state, action, state, False)

    def release_many(self, states):
        for state in states:
            self.released.append(state)
            del self.states[state]
        return len(states)


def test_replay_selected_root_checks_every_public_state_and_menu():
    backend = ReplayBackend()
    trajectory = {"decisions": [
        {**make_decision(i), "observation_hash": f"h{i}"}
        for i in range(3)
    ]}
    h, observation, menu = replay_selected_root(
        backend, 123, trajectory, 2,
    )
    assert backend.states[h] == 2
    assert observation.observation_hash == "h2"
    assert menu == MENU
    backend.release_many((h,))
    trajectory["decisions"][1]["legal_action_ids"] = ["incorrect"]
    with pytest.raises(RuntimeError, match="observation/menu"):
        replay_selected_root(backend, 123, trajectory, 2)
    assert backend.states == {}


def test_summary_distinguishes_shared_root_search_from_wins_and_losses():
    traces = [
        case("lost", "victory", "defeat",
             [make_decision(0, disagreement=True)]),
        case("recovered", "defeat", "victory",
             [make_decision(0, disagreement=False)]),
    ]
    summary = make_report(
        traces, [], {"fixture_seed": "seed", "seed_prefix": "prefix"},
        {"r10": "a" * 64, "r30": "b" * 64},
    )
    assert summary["schema"] == SCHEMA
    assert summary["r10_wins"] == summary["r30_wins"] == 1
    assert summary["regressed_case_ids"] == ["lost"]
    assert summary["improved_case_ids"] == ["recovered"]
    assert summary["public_decisions_paired"] == 4
    assert summary["raw_action_disagreement_fraction"] == pytest.approx(.5)
    assert summary["searched_shared_public_roots"] == 0
