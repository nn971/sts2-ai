"""H6-C replay: fair state capture, map/boss persistence and readable reports."""
from __future__ import annotations

import json
from types import SimpleNamespace

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.evaluation.macro_trace import (
    TraceRecorder,
    action_label,
    boss_identity,
    first_divergence,
    map_svg,
    render_html,
    render_markdown,
)
from sts2_ai.models.protocol import PolicyValueEstimate


class Model:
    def evaluate(self, _obs, legal):
        return PolicyValueEstimate(
            tuple(float(len(legal) - i) for i in range(len(legal))), 0.12
        )


def sample_trace(label="80", selected=0):
    recorder = TraceRecorder(Model(), model_label=label)
    nodes = [
        {"node_id": "a", "floor": 1, "room_type": "combat",
         "next_node_ids": ["b", "c"]},
        {"node_id": "b", "floor": 2, "room_type": "rest",
         "next_node_ids": ["d"]},
        {"node_id": "c", "floor": 2, "room_type": "elite",
         "next_node_ids": ["d"]},
        {"node_id": "d", "floor": 16, "room_type": "boss",
         "next_node_ids": []},
    ]
    actions = (
        LegalAction("a", "choose_map_node", '{"NodeId":"b"}'),
        LegalAction("b", "choose_map_node", '{"NodeId":"c"}'),
    )
    state = {
        "act": 1, "floor": 1, "hp": 30, "max_hp": 70, "gold": 90,
        "map": nodes, "boss_encounter_id": "sample.act1.boss",
        "deck": [{"card_id": "silent.strike"}],
    }
    obs = Observation("prototype-fair-v0", json.dumps(state), "public")
    recorder(None, obs, actions, SimpleNamespace(action=actions[selected]))
    state.pop("map")
    state["floor"] = 2
    state["hp"] = 25
    second = Observation("prototype-fair-v0", json.dumps(state), "public-2")
    action = (LegalAction("next", "skip_reward"),)
    recorder(None, second, action, SimpleNamespace(action=action[0]))
    summary = SimpleNamespace(
        outcome="defeat", act1_cleared=False, terminal_floor=2,
        frontier_progress=1.8, censored=False, hp_trajectory=(30, 25, 0),
    )
    return recorder.export("seed-x", summary, "pinned", "model-sha")


def test_capture_keeps_complete_public_map_and_boss_across_phases():
    trace = sample_trace()
    assert trace["boss_identity"] == "sample.act1.boss"
    assert len(trace["map_snapshots"]) == 1
    assert len(trace["map_snapshots"][0]["nodes"]) == 4
    assert trace["decisions"][0]["selected_node"] == "b"
    assert "map" not in trace["decisions"][0]["public_state"]
    assert trace["decisions"][1]["floor"] == 2
    assert len(trace["decisions"][0]["actions"]) == 2
    assert action_label(trace["decisions"][0]["actions"][0]) == "choose map node: b"


def test_first_policy_disagreement_and_map_are_rendered():
    a = sample_trace()
    b = sample_trace("160", selected=1)
    divergence = first_divergence(a, b)
    assert divergence == {
        "index": 0, "kind": "policy_disagreement", "same_public_state": True,
    }
    page = render_html([a, b])
    assert "sample.act1.boss" in page
    assert 'viewBox=' in page
    assert "Complete publicly visible Act 1 map" in page
    assert "first route" in page
    assert "First divergence" in page
    assert "choose map node: b" in page
    assert "choose map node: c" in page
    assert "0.1200" in page or "0.12" in page
    assert "position:sticky" in page
    markdown = render_markdown([a, b])
    assert "policy_disagreement" in markdown
    assert "floor 1" in markdown


def test_same_seed_different_observation_not_misclassified_as_policy():
    a = sample_trace()
    b = sample_trace("160", selected=0)
    b["decisions"][0]["public_state"]["hp"] = 10
    d = first_divergence(a, b)
    assert d["kind"] == "observation_divergence"


def test_unknown_boss_explicit_and_html_escapes():
    a = sample_trace()
    a["boss_identity"] = None
    a["boss_status"] = "not_exposed"
    a["decisions"][0]["actions"][0]["payload"]["NodeId"] = '<script>alert(1)</script>'
    page = render_html([a])
    assert "Boss: not exposed" in page
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert boss_identity({"map": [], "boss_encounter": None}) is None
    assert boss_identity({"run": {"boss_id": "visible-unique"}}) == "visible-unique"
    assert "Full map not exposed" in map_svg([], {}, {})


def test_deck_and_hidden_fields_never_inferred():
    rec = TraceRecorder(Model(), model_label="private-guard")
    actions = (LegalAction("x", "skip_reward"),)
    frame = {"act": 1, "floor": 0, "semanticDeck": {"private": "no"}}
    obs = Observation("prototype-fair-v0", json.dumps(frame), "private")
    import pytest
    with pytest.raises(ValueError, match="Privileged"):
        rec(None, obs, actions, SimpleNamespace(action=actions[0]))
