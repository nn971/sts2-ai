"""H26 public-threat, search-free return and actor-critic invariants."""
from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
import torch

from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.models.shared_macro_transformer_v2 import H21GroundedMacroTransformer
from sts2_ai.models.shared_transformer import H7Config
from sts2_ai.training.h16_joint_act_pilot import (
    JointActPilotNet,
    PilotDecision,
    PilotEpisode,
)
from sts2_ai.training.h24_joint_act_model import H24JointActPilotNet
from sts2_ai.training.h26_combat_rl import (
    combat_return_samples,
    fit_search_free_combat,
)
from sts2_ai.training.h26_tactical import (
    attach_visible_threat,
    combat_dense_rewards,
    visible_incoming_threat,
)


def _combat(*, known: bool = True, block: int = 5, hp: int = 60):
    enemy = {
        "instance_id": 2, "enemy_id": "proto.enemy.flail_knight",
        "hp": 40, "max_hp": 40, "block": 0, "move_id": "ram",
        "powers": [], "statuses": {},
    }
    if known:
        enemy.update(intent_damage=9, intent_hits=2)
    return {
        "hp": hp, "max_hp": 70, "gold": 30,
        "floor": 5, "deck": [], "relics": [], "potions": [],
        "map": [], "current_map_node_id": None,
        "potion_slot_capacity": 3, "potion_slots_empty": 3,
        "potion_slots_occupied": 0,
        "combat": {
            "turn": 1, "energy": 3, "player_block": block,
            "history": {},
            "hand": [], "discard_pile": [], "exhaust_pile": [],
            "enemies": [enemy],
        },
    }


def _actions():
    return (
        LegalAction("wait", "end_turn", "{}"),
        LegalAction("strike", "play_card", '{"CardInstanceId":1}'),
    )


def _models():
    torch.set_num_threads(1)
    source = JointActPilotNet(H21GroundedMacroTransformer(
        H7Config(width=32, heads=4, layers=1, feedforward=64),
    )).eval()
    macro = H24JointActPilotNet.warmstart_h22(
        source, act_only=True, search_act_mix=1.0, macro_progress=True,
    )
    combat = copy.deepcopy(macro)
    combat.macro_progress_head = None
    combat.h26_tactical = True
    return macro, combat


def _episode():
    s0 = _combat()
    s1 = _combat(hp=52)
    s1["combat"]["enemies"][0]["hp"] = 15
    s2 = {
        **{k: v for k, v in s0.items() if k != "combat"},
        "hp": 45, "floor": 5, "combat": None,
    }
    actions = (LegalAction("wait", "end_turn", "{}"),)
    decisions = tuple(PilotDecision(
        Observation("prototype-fair-v0", json.dumps(s), "test"), actions,
        0, (0.0,),
    ) for s in (s0, s1))
    return PilotEpisode("seed", decisions, (s0, s1, s2),
                        "defeat", 0, 0)


def test_visible_intent_damage_and_partial_uncertainty():
    threat = visible_incoming_threat(_combat())
    assert threat.known_damage == 18
    assert threat.known_hits == 2
    assert threat.known_hp_loss == 13
    assert threat.complete
    assert not threat.known_lethal
    missing = visible_incoming_threat(_combat(known=False))
    assert missing.unknown_intents == 1 and not missing.complete
    assert missing.known_damage == 0
    with pytest.raises(ValueError):
        visible_incoming_threat({"hp": 5})


def test_h26_tactical_token_is_policy_only_and_checkpointed(tmp_path):
    macro, combat = _models()
    state = _combat()
    actions = (LegalAction("wait", "end_turn", "{}"),)
    base = macro.encode_decision(state, actions)
    altered = combat.encode_decision(state, actions)
    assert len(altered.objects) == len(base.objects) + 1
    assert altered.neutral_value == base.neutral_value
    assert altered.actions == base.actions
    # The new feature is only a public policy token.
    path = tmp_path / "combat.pt"
    combat.save(path)
    restored = JointActPilotNet.load(path)
    assert restored.h26_tactical
    assert len(restored.encode_decision(state, actions).objects) == len(altered.objects)
    assert not macro.h26_tactical


def test_h26_tactical_capacity_preserves_h21_actions_and_relations():
    # Regression: full H21 frames routinely exceed legacy H7's 256 objects.
    macro, _ = _models()
    state = _combat()
    actions = (LegalAction("wait", "end_turn", "{}"),)
    base = macro.encode_decision(state, actions)
    assert base.objects
    for count in (256, 257, 1024, 1536):
        expanded = replace(
            base,
            objects=base.objects + (base.objects[-1],) * (
                count - len(base.objects)
            ),
        )
        actual = attach_visible_threat(expanded, state)
        assert len(actual.objects) == count + 1
        assert actual.objects[:count] == expanded.objects
        assert actual.actions == expanded.actions
        assert actual.relations == expanded.relations
        assert actual.neutral_value == expanded.neutral_value
    oversized = replace(
        base, objects=base.objects * (
            1 + 1536 // len(base.objects)
        ),
    )
    with pytest.raises(ValueError, match="exceeds its object cap"):
        attach_visible_threat(oversized, state)


def test_combat_shaping_and_searchfree_returns(monkeypatch):
    episode = _episode()
    shaped = combat_dense_rewards(list(episode.frames))
    assert len(shaped) == 2
    assert all(abs(v) <= 0.2 for v in shaped)
    from sts2_ai.training import h26_combat_rl as rl
    monkeypatch.setattr(rl, "_macro_act_value", lambda *a: 0.6)
    samples = combat_return_samples(episode, object())
    assert len(samples) == 2
    assert all(s.fight_won for s in samples)
    assert samples[0].target <= samples[1].target + 0.3


def test_h26_optimizer_updates_combat_with_no_search(monkeypatch):
    macro, combat = _models()
    episode = _episode()
    from sts2_ai.training import h26_combat_rl as rl
    monkeypatch.setattr(rl, "_macro_act_value", lambda *a: 0.5)
    before = [p.detach().clone() for p in macro.parameters()]
    optimizer = torch.optim.AdamW(combat.parameters(), lr=1e-4)
    report = fit_search_free_combat(
        combat, macro, optimizer, [episode], minibatch=2,
        epochs=1, max_samples=32,
    )
    assert report["examples"] == 2
    assert report["steps"] == 1
    assert torch.isfinite(torch.tensor(report["loss"]))
    assert report["combat_win_samples"] == 2
    for old, new in zip(before, macro.parameters(), strict=True):
        torch.testing.assert_close(old, new, atol=0, rtol=0)
