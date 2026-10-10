"""Width doubling preserves actual v8 inference while enabling new gradients."""
from __future__ import annotations

import copy
import json

import pytest

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.neural import NEURAL_FORMAT, NeuralPolicyValueModel
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.models.width_expansion import widen_neural, widen_phase_split
from test_enemy_instances_v8 import build_model, combat_frame, targeted


def _source() -> PhaseSplitNeuralModel:
    combat = build_model()
    raw = copy.deepcopy(combat.to_dict())
    raw["format"] = NEURAL_FORMAT
    raw["model_id"] = "strategy-fixture"
    for field in (
        "enemy_weight", "enemy_bias",
        "enemy_context_weight", "enemy_target_weight",
    ):
        raw.pop(field, None)
    strategy = NeuralPolicyValueModel.from_dict(raw)
    return PhaseSplitNeuralModel(
        strategy=strategy, combat=combat, model_id="baseline-v8",
        combat_value_objective="hp_preservation",
    )


def _observation(frame: dict) -> Observation:
    payload = json.dumps(frame, sort_keys=True)
    return Observation("prototype-fair-v0", payload, "fixture")


def test_hidden_width_widening_preserves_value_and_all_target_logits() -> None:
    source = _source()
    widened = widen_phase_split(source, 8)
    assert widened.strategy.hidden == widened.combat.hidden == 8
    assert widened.combat_value_objective == source.combat_value_objective
    assert widened.combat.format_id == source.combat.format_id
    cases = [
        (combat_frame(), (
            targeted(3), targeted(1), targeted(2),
            LegalAction("end", "end_turn", "{}"),
        )),
        ({"act": 1, "floor": 8, "hp": 33, "max_hp": 70,
          "gold": 92, "deck": [{"card_id": "proto.silent.strike"}]},
         (LegalAction("map-1", "choose_map_node", '{"NodeId":"a"}'),
          LegalAction("map-2", "choose_map_node", '{"NodeId":"b"}'))),
    ]
    for state, actions in cases:
        original = source.evaluate(_observation(state), actions)
        expanded = widened.evaluate(_observation(state), actions)
        assert expanded.value == pytest.approx(original.value, abs=1e-9)
        assert expanded.action_logits == pytest.approx(
            original.action_logits, abs=1e-9,
        )


def test_outgoing_splits_break_duplicate_symmetry_without_changing_outputs() -> None:
    original = _source().combat
    expanded = widen_neural(original, 8)
    for j, outgoing in enumerate((original.policy_weight, original.value_weight)):
        new = expanded.policy_weight if j == 0 else expanded.value_weight
        for i, value in enumerate(outgoing):
            assert new[2 * i] + new[2 * i + 1] == pytest.approx(value)
            if value:
                assert new[2 * i] != new[2 * i + 1]
    assert expanded.enemy_weight is not None
    assert original.enemy_weight is not None
    assert expanded.enemy_weight[2] == expanded.enemy_weight[3]
    assert expanded.enemy_weight[2] is not expanded.enemy_weight[3]


def test_refuses_bad_widths_and_supports_roundtrip(tmp_path) -> None:
    model = _source()
    with pytest.raises(ValueError, match="integer multiple"):
        widen_phase_split(model, 5)
    with pytest.raises(ValueError, match="integer multiple"):
        widen_phase_split(model, 4)
    expanded = widen_phase_split(model, 8)
    path = tmp_path / "widened.json"
    expanded.save(path)
    reloaded = PhaseSplitNeuralModel.load(path)
    assert reloaded.to_dict() == expanded.to_dict()
    assert model.to_dict() != expanded.to_dict()


def test_widening_is_deterministic() -> None:
    source = _source()
    assert widen_phase_split(source, 8).to_dict() == widen_phase_split(source, 8).to_dict()


def test_comparison_mode_does_not_select_a_lucky_individual() -> None:
    from sts2_ai.evaluation.checkpoint_selection import analyze
    from tools.evaluate_act1_checkpoints import identity_for

    goal = "native-act1-boss-v1"
    rows = {}
    for label, won in [("h32-r1", False), ("h64-r1", True)]:
        rows[label] = [{
            "seed": "capacity-seed-0", "outcome": "victory" if won else "defeat",
            "act1_cleared": won, "episode_goal_version": goal,
            "censored": False, "frontier_progress": 16.0,
            "boss_progress": None,
        }]
    analysis = analyze(rows, mode="comparison", reference_label="h32-r1")
    assert analysis["selected_label"] is None
    assert analysis["ranking"] is None
    manifest = identity_for(
        mode="comparison", specs={
            name: {"path": f"/{name}.json", "sha256": name}
            for name in rows
        }, prefix="capacity-fresh-v12", n=256,
        max_decisions=4096, emulator_revision="pin",
        reference_label="h32-r1", selection_sha256=None,
    )
    assert manifest["mode"] == "comparison"
