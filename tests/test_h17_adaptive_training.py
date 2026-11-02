"""H17 adaptive search budget, card exposures, replay and schedule contracts."""
from __future__ import annotations

import json
import random

import pytest
import torch

from sts2_ai.emulator import FAIR_POLICY_ID
from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.models.shared_transformer import H7Config, SharedObjectTransformer
from sts2_ai.training.h16_joint_act_pilot import (
    JointActPilotNet,
    PilotDecision,
    PilotEpisode,
    completed_room_hp,
    geometric_hp_return,
    hp_after_current_room,
    hp_fraction,
)
from sts2_ai.training.h17_adaptive import (
    AdaptiveSearch,
    card_for_action,
    choose_budget,
    coverage_summary,
    normalized_entropy,
    update_coverage,
)
from sts2_ai.training.h17_replay import (
    collect_replay,
    fit_replay,
    load_replay,
    merge_replay,
    sample_replay,
    write_replay,
)
from tools.train_h16_joint_act_pilot import (
    h17_horizon_for_round,
    h17_macro_weight_for_round,
)


def state(*, floor: int = 7, hp: int = 42, complete: bool = False):
    return {
        "act": 1, "floor": floor, "hp": hp, "max_hp": 70,
        "completed_rooms": (
            [{"act": 1, "floor": floor}] if complete else []
        ),
        "combat": None if complete else {
            "turn": 1, "enemies": [], "hand": [
                {"instance_id": 17, "card_id": "proto.silent.rare"},
                {"instance_id": 18, "card_id": "proto.silent.strike"},
            ],
        },
        "deck": [
            {"card_id": "proto.silent.strike"},
            {"card_id": "proto.silent.rare"},
        ],
        "relics": [], "potions": [], "map": [],
    }


ACTIONS = (
    LegalAction("rare", "play_card", '{"CardInstanceId":17}'),
    LegalAction("strike", "play_card", '{"CardInstanceId":18}'),
    LegalAction("end", "end_turn"),
)


def episode() -> PilotEpisode:
    first = state()
    observed = Observation(
        FAIR_POLICY_ID, json.dumps(first, sort_keys=True), "abc",
    )
    decision = PilotDecision(
        observed, ACTIONS, 0, (0., 0., 0.),
        search_policy=(0.8, 0.1, 0.1),
        predicted_value=0.35, search_simulations=16,
    )
    return PilotEpisode(
        "h17-demo", (decision,), (first, state(complete=True, hp=30)),
        "defeat", 1, 200,
    )



def test_fixed_hp_reference_rewards_max_hp_gains_without_tablet_exploit():
    # Same current HP, different maximum HP: fixed normalization agrees.
    assert hp_fraction(
        {"hp": 1, "max_hp": 1}, hp_reference=100,
    ) == pytest.approx(0.01)
    assert hp_fraction(
        {"hp": 39, "max_hp": 70}, hp_reference=100,
    ) == pytest.approx(0.39)
    assert hp_fraction(
        {"hp": 80, "max_hp": 80}, hp_reference=100,
    ) == pytest.approx(0.80)
    assert hp_fraction(
        {"hp": 110, "max_hp": 110}, hp_reference=100,
    ) == 1.0
    assert hp_fraction(
        {"hp": 200, "max_hp": 200}, hp_reference=100,
    ) == 1.0
    assert hp_fraction({"hp": 1, "max_hp": 1}) == 1.0

    # Tablet of Truth after an event at floor 3: 39/70 -> 1/1.
    before = state(floor=3, hp=39)
    after = state(floor=3, hp=1, complete=True)
    after["max_hp"] = 1
    frames = (before, after)
    assert completed_room_hp(frames)[3] == 1.0  # Old biased target.
    assert completed_room_hp(frames, hp_reference=100)[3] == 0.01
    assert hp_after_current_room(frames, 0, hp_reference=100) == 0.01
    lam = 0.4132
    unfair = geometric_hp_return((1.0,) * 4, 0.0, lam)
    corrected = geometric_hp_return((0.01,) * 4, 0.0, lam)
    assert unfair > 0.97
    assert corrected == pytest.approx(0.01 * unfair)

    with pytest.raises(ValueError, match="HP reference"):
        hp_fraction({"hp": 30, "max_hp": 70}, hp_reference=0)


def test_search_is_mixed_but_preserves_a_random_confidence_audit():
    initial = state()
    assert card_for_action(initial, ACTIONS[0]) == "proto.silent.rare"
    assert card_for_action(initial, ACTIONS[-1]) is None
    assert normalized_entropy([0, 0, 0]) == pytest.approx(1)
    assert normalized_entropy([0, -100, -100]) == pytest.approx(0, abs=1e-30)
    config = AdaptiveSearch()
    rng = random.Random(1717)
    draws = [
        choose_budget(
            config, initial, ACTIONS, [7., 0., 0.],
            {"proto.silent.rare": 0}, rng,
        ) for _ in range(3000)
    ]
    assert set(draws) == {0, 16, 64}
    assert 250 < draws.count(16) < 600
    assert 20 < draws.count(64) < 150
    assert choose_budget(
        config, state(complete=True), ACTIONS, [0, 0, 0], {}, rng,
    ) == 0
    assert choose_budget(
        config, initial, ACTIONS[:1], [0], {}, rng,
    ) == 0
    with pytest.raises(ValueError, match="Invalid adaptive"):
        AdaptiveSearch(0.7, 0.2)


def test_card_coverage_counts_by_type_and_resume():
    coverage = update_coverage([episode()], {})
    assert coverage["deck_runs"]["proto.silent.rare"] == 1
    assert coverage["playable"]["proto.silent.rare"] == 1
    assert coverage["chosen"]["proto.silent.rare"] == 1
    assert coverage["searched"]["proto.silent.rare"] == 1
    doubled = update_coverage([episode()], coverage)
    assert doubled["playable"]["proto.silent.rare"] == 2
    assert coverage_summary(doubled)["distinct_card_types"]["deck_runs"] == 2


def test_h17_schedules_and_boundaries():
    assert h17_horizon_for_round(0) == pytest.approx(0.4113950303216448)
    assert h17_horizon_for_round(100) == pytest.approx(
        (0.4113950303216448 + 0.77) / 2,
    )
    assert h17_horizon_for_round(250) == 0.77
    assert h17_macro_weight_for_round(0) == 0.5
    assert h17_macro_weight_for_round(60) == 0.25
    assert h17_macro_weight_for_round(120) == 0.0
    with pytest.raises(ValueError):
        h17_horizon_for_round(3, start=0.9, target=0.4)
    with pytest.raises(ValueError):
        h17_macro_weight_for_round(-1)


def test_supervised_replay_is_capped_prioritized_and_resume_safe(tmp_path):
    examples = collect_replay(
        [episode()], source_round=0, exposure={}, hp_reference=100,
    )
    assert len(examples) == 1
    assert examples[0]["search_policy"] == [0.8, 0.1, 0.1]
    assert examples[0]["room_hp"][0] == pytest.approx(0.30)
    assert len(examples[0]["room_hp"]) == 16 - 7 + 1
    pool = merge_replay(
        [], examples, round_index=0, exposure={}, capacity=4,
    )
    # Stale-policy REINFORCE is excluded: only previous-round data is sampled.
    assert sample_replay(pool, round_index=0, exposure={}) == []
    replay = sample_replay(pool, round_index=1, exposure={})
    assert len(replay) == 1
    output = tmp_path / "r0001.json.gz"
    write_replay(
        output, fingerprint="fingerprint", round_index=1, rows=pool,
    )
    assert load_replay(output, fingerprint="fingerprint", expected_round=1) == pool
    with pytest.raises(ValueError, match="mismatch"):
        load_replay(output, fingerprint="changed", expected_round=1)
    with pytest.raises(ValueError, match="Missing"):
        load_replay(tmp_path / "missing.gz", fingerprint="f", expected_round=3)


def test_replay_trains_value_and_search_targets_without_reinforce():
    model = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64),
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    examples = collect_replay([episode()], source_round=0, exposure={})
    report = fit_replay(
        model, optimizer, examples, horizon_lambda=0.6,
        minibatch=1, replay_weight=0.2,
    )
    assert report["steps"] == 1
    assert report["macro_examples"] == 0
    assert report["search_targets"] == 1
    assert report["loss"] >= 0
    assert fit_replay(model, optimizer, [], horizon_lambda=0.6)["steps"] == 0
