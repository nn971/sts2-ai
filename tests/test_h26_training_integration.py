"""H26 one-round CPU trainer contract without native emulator or PUCT."""
from __future__ import annotations

import json
import sys

import pytest
import torch
from test_h26_damage_risk import _catalog_v3

from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.models.shared_macro_transformer_v2 import H21GroundedMacroTransformer
from sts2_ai.models.shared_transformer import H7Config
from sts2_ai.training.h16_joint_act_pilot import (
    JointActPilotNet,
    PilotDecision,
    PilotEpisode,
)
from tools import train_h26_searchfree_a0 as trainer


def _episodes():
    resources = {
        "hp": 55, "max_hp": 70, "gold": 90, "floor": 6, "act": 1,
        "deck": [], "relics": [], "potions": [], "map": [],
        "current_map_node_id": None,
        "potion_slot_capacity": 3, "potion_slots_occupied": 0,
        "potion_slots_empty": 3,
    }
    macro_state = {
        **resources, "combat": None, "event": {
            "event_id": "native.ancient_temple",
            "offered_choice_ids": ["gold", "heal"],
            "visible_choice_values": [
                {"choice_id": "gold", "gold_gain": 110, "hp_loss": 6},
                {"choice_id": "heal", "heal": 15},
            ],
        },
    }
    choices = (
        LegalAction("gold", "event_choice", '{"ChoiceId":"gold"}'),
        LegalAction("heal", "event_choice", '{"ChoiceId":"heal"}'),
    )
    macro_decision = PilotDecision(
        Observation("prototype-fair-v0", json.dumps(macro_state), "macro"),
        choices, 0, (0., 0.),
    )
    macro_ep = PilotEpisode(
        "macro-seed", (macro_decision,),
        (macro_state, {**macro_state, "floor": 8, "hp": 0}),
        "defeat", searches=0, search_transitions=0,
    )
    enemy = {
        "instance_id": 1, "enemy_id": "proto.enemy.flail_knight",
        "hp": 30, "max_hp": 30, "block": 0, "move_id": "ram",
        "intent_damage": 12, "intent_hits": 1,
        "powers": [], "statuses": {},
    }
    fight_state = {
        **resources, "combat": {
            "turn": 1, "energy": 3, "player_block": 0, "history": {},
            "hand": [], "discard_pile": [], "exhaust_pile": [],
            "enemies": [enemy],
        },
    }
    later_state = json.loads(json.dumps(fight_state))
    later_state["combat"]["enemies"][0]["hp"] = 10
    later_state["hp"] = 50
    end_state = {**resources, "hp": 45, "combat": None}
    action = (LegalAction("wait", "end_turn", "{}"),)
    decisions = tuple(PilotDecision(
        Observation("prototype-fair-v0", json.dumps(state), "combat"),
        action, 0, (0.,),
    ) for state in (fight_state, later_state))
    fight_ep = PilotEpisode(
        "fight-seed", decisions, (fight_state, later_state, end_state),
        "defeat", searches=0, search_transitions=0,
    )
    return macro_ep, fight_ep


@pytest.mark.parametrize(
    ("with_damage_catalog", "disable_risk"),
    [(False, False), (True, False), (True, True)],
)
def test_h26_one_round_is_searchfree_and_updates_both_models(
    tmp_path, monkeypatch, with_damage_catalog, disable_risk,
):
    torch.set_num_threads(1)
    source = JointActPilotNet(H21GroundedMacroTransformer(
        H7Config(width=32, heads=4, layers=1, feedforward=64),
    )).eval()
    h22 = tmp_path / "h22.pt"
    source.save(h22)
    emu = tmp_path / "emulator"
    emu.mkdir()
    output = tmp_path / "training"
    macro_ep, fight_ep = _episodes()
    stages = []

    def fake_collect(args, macro_path, combat_path, seeds, *,
                     stage, round_number):
        assert macro_path.exists() and combat_path.exists()
        stages.append(stage)
        return [macro_ep, fight_ep], {
            "roots": 0, "transitions": 0, "examples": 0,
        }

    monkeypatch.setattr(trainer, "_collect", fake_collect)
    monkeypatch.setattr(trainer, "summarize", lambda episodes: {
        "cleared": 0, "completed": len(episodes), "censored": 0,
    })
    common = [
        "train_h26_searchfree_a0.py",
        "--initial-h22", str(h22),
        "--emulator-root", str(emu),
        "--emulator-revision", "test",
        "--output-root", str(output),
        "--rounds", "1", "--runs-per-round", "2", "--heldout-runs", "2",
        "--workers", "1", "--minibatch", "2", "--device", "cpu",
        "--execute",
    ]
    if with_damage_catalog:
        path = tmp_path / "source-damage-catalog.json"
        path.write_text(json.dumps(_catalog_v3()))
        common.extend(["--intent-catalog", str(path)])
        if disable_risk:
            common.append("--disable-next-damage-risk")
    monkeypatch.setattr(sys, "argv", common)
    trainer.main()
    assert stages == ["search_free_full_acts", "heldout"]
    saved = json.loads((output / "progress.json").read_text())
    row, = saved["rounds"]
    assert row["combat_search"]["roots"] == 0
    assert row["combat_search"]["transitions"] == 0
    assert row["combat_update"]["steps"] > 0
    assert row["macro_update"]["steps"] > 0
    assert row["combat_checkpoint_sha256"]
    assert row["macro_checkpoint_sha256"]
    assert row["optimizer_checkpoint_sha256"]
    assert len(saved["experiment"]["ai_source_sha256"]) == 64
    assert saved["experiment"]["next_damage_risk_enabled"] == (
        with_damage_catalog and not disable_risk
    )
    if with_damage_catalog:
        assert saved["experiment"]["schema"] == (
            "sts2-h26-search-free-conditional-gross-damage-v3"
        )
        assert "damage_risk_coverage" in row
    else:
        assert "damage_risk_coverage" not in row
    loaded_macro = JointActPilotNet.load(
        output / "models" / "macro-r0001.pt", device="cpu",
    )
    loaded_combat = JointActPilotNet.load(
        output / "models" / "combat-r0001.pt", device="cpu",
    )
    assert loaded_macro.macro_progress_head is not None
    assert not loaded_macro.h26_tactical
    assert loaded_combat.macro_progress_head is None
    assert loaded_combat.h26_tactical
    assert loaded_combat.h26_damage_ascension == (
        0 if with_damage_catalog and not disable_risk else None
    )
