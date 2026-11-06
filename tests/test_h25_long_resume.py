"""H25 long-duration wall-budget and two-optimizer checkpoint recovery."""
from __future__ import annotations

import json
import sys

import torch

from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.models.shared_macro_transformer_v2 import H21GroundedMacroTransformer
from sts2_ai.models.shared_transformer import H7Config
from sts2_ai.training.h16_joint_act_pilot import (
    JointActPilotNet,
    PilotDecision,
    PilotEpisode,
)
from sts2_ai.training.h25_alternating import CombatSearchExample
from tools import train_h25_alternating_a0 as trainer


def _episode():
    state = {
        "hp": 55, "max_hp": 70, "gold": 90, "floor": 6, "act": 1,
        "deck": [], "relics": [], "potions": [],
        "potion_slot_capacity": 3, "potion_slots_occupied": 0,
        "potion_slots_empty": 3, "map": [], "current_map_node_id": None,
        "combat": None,
        "event": {
            "event_id": "native.ancient_temple",
            "offered_choice_ids": ["gold", "heal"],
            "visible_choice_values": [
                {"choice_id": "gold", "gold_gain": 110, "hp_loss": 6},
                {"choice_id": "heal", "heal": 15},
            ],
        },
    }
    actions = (
        LegalAction("gold", "event_choice", '{"ChoiceId":"gold"}'),
        LegalAction("heal", "event_choice", '{"ChoiceId":"heal"}'),
    )
    observation = Observation("prototype-fair-v0", json.dumps(state), "h25-resume")
    decision = PilotDecision(observation, actions, 0, (0.0, 0.0))
    episode = PilotEpisode(
        "h25-test-seed", (decision,), (state, {**state, "hp": 0}),
        "defeat", searches=0, search_transitions=0,
    )
    search = CombatSearchExample(observation, actions, (0.3, 0.7), "seed", 0)
    return episode, search


def test_h25_cumulative_wall_budget_and_strict_optimizer_resume(
    tmp_path, monkeypatch,
):
    torch.set_num_threads(1)
    h22 = tmp_path / "r95.pt"
    source = JointActPilotNet(H21GroundedMacroTransformer(
        H7Config(width=32, heads=4, layers=1, feedforward=64),
    )).eval()
    source.save(h22)
    emulator = tmp_path / "emulator"
    emulator.mkdir()
    root = tmp_path / "study"
    episode, sample = _episode()
    collected = []

    def fake_collect(args, macro_path, combat_path, seeds, *, fight_budget,
                     stage, round_number):
        assert macro_path.exists() and combat_path.exists()
        collected.append((stage, round_number, fight_budget))
        return (
            [episode], [sample] if fight_budget else [],
            {
                "sampled_fights": int(bool(fight_budget)),
                "complete_fights": int(bool(fight_budget)),
                "incomplete_fights": 0,
                "roots": int(bool(fight_budget)), "transitions": 12,
                "examples": int(bool(fight_budget)),
            },
        )

    monkeypatch.setattr(trainer, "_collect", fake_collect)
    monkeypatch.setattr(
        trainer, "summarize",
        lambda episodes: {
            "played": len(episodes), "completed": len(episodes),
            "censored": 0, "cleared": 0,
        },
    )
    common = [
        "train_h25_alternating_a0.py",
        "--initial-h22", str(h22),
        "--emulator-root", str(emulator), "--emulator-revision", "test",
        "--output-root", str(root), "--rounds", "2",
        "--runs-per-round", "1", "--heldout-runs", "1", "--workers", "1",
        "--macro-update-every", "1",
        "--device", "cpu", "--execute",
    ]
    monkeypatch.setattr(
        sys, "argv", [*common, "--max-wall-seconds", "0.00001"],
    )
    trainer.main()
    progress = json.loads((root / "progress.json").read_text())
    assert len(progress["rounds"]) == 1
    assert (root / "models" / "optimizers-r0001.pt").exists()
    round1 = progress["rounds"][0]
    for key in (
        "macro_checkpoint_sha256", "combat_checkpoint_sha256",
        "optimizer_checkpoint_sha256",
    ):
        assert len(round1[key]) == 64
    assert progress["elapsed_wall_seconds"] > 0
    assert collected == [
        ("full_acts_and_sampled_fight_search", 1, 1), ("heldout", 1, 0),
    ]

    monkeypatch.setattr(
        sys, "argv",
        [*common, "--resume", "--max-wall-seconds", "120"],
    )
    trainer.main()
    progress = json.loads((root / "progress.json").read_text())
    assert len(progress["rounds"]) == 2
    assert [row["round"] for row in progress["rounds"]] == [1, 2]
    assert (root / "models" / "optimizers-r0002.pt").exists()
    assert collected[-2:] == [
        ("full_acts_and_sampled_fight_search", 2, 1), ("heldout", 2, 0),
    ]
    saved = torch.load(
        root / "models" / "optimizers-r0002.pt",
        weights_only=True, map_location="cpu",
    )
    assert saved["round"] == 2
    assert saved["macro_optimizer"]["state"]
    assert saved["combat_optimizer"]["state"]


def test_h25_atomic_optimizer_roundtrip(tmp_path):
    path = tmp_path / "opt.pt"
    trainer._atomic_torch_save(
        {"fingerprint": "abc", "round": 1, "values": torch.tensor([1.0])},
        path,
    )
    value = torch.load(path, weights_only=True, map_location="cpu")
    assert value["fingerprint"] == "abc" and value["round"] == 1
    torch.testing.assert_close(value["values"], torch.tensor([1.0]))


def test_h25_checkpoint_retention_preserves_latest_and_periodic(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    for n in range(1, 9):
        for kind in ("macro", "combat", "optimizers"):
            (folder / f"{kind}-r{n:04d}.pt").write_bytes(b"checkpoint")
    pruned = trainer.prune_h25_checkpoints(
        tmp_path, newest=8, keep_last=2, archive_every=4,
    )
    assert len(pruned) == 15
    for n in range(1, 9):
        expected = n in (4, 7, 8)
        for kind in ("macro", "combat", "optimizers"):
            assert (folder / f"{kind}-r{n:04d}.pt").exists() == expected


def test_h25_monitor_supports_live_without_round_commit(tmp_path):
    from tools.monitor_h25 import render_h25

    (tmp_path / "live.json").write_text(json.dumps({
        "timestamp_unix": 1000, "round": 1,
        "stage": "full_acts_and_sampled_fight_search",
        "completed_episodes": 0, "total_episodes": 16,
    }))
    dashboard = render_h25(tmp_path, now=1020)
    assert "No committed round yet" in dashboard
    assert "completed workers 0/16" in dashboard
    assert "heartbeat age 20s" in dashboard
