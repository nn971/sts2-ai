"""H7 terminal replay audit selection must not depend on outcomes or scheduling."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sts2_ai.nature.h5_curriculum import TRAIN_CELLS, cell_for_source, source_bank
from tools.train_h7_reference_combat import terminal_replay_audit_indices, collect


def test_stratified_indices_are_deterministic_cover_all_cells():
    seeds = source_bank("train", 128, study_seed=73127)
    first = terminal_replay_audit_indices(seeds, 8)
    assert first == terminal_replay_audit_indices(seeds, 8)
    assert len(first) == len(TRAIN_CELLS) == 16
    assert len(set(first)) == len(first)
    assert set(cell_for_source(seeds[i]) for i in first) == set(TRAIN_CELLS)
    assert terminal_replay_audit_indices(seeds, 1) == tuple(range(len(seeds)))
    assert len(terminal_replay_audit_indices(seeds, 4)) == 32
    assert len(terminal_replay_audit_indices(
        source_bank("train", 16, study_seed=73127), 8
    )) == 16


@pytest.mark.parametrize("rate", [0, -1, False, 0.5])
def test_bad_replay_rate_rejected(rate):
    with pytest.raises(ValueError, match="audit rate"):
        terminal_replay_audit_indices(("anything",), rate)


def test_missing_curated_seed_rejected(monkeypatch):
    monkeypatch.setattr(
        "tools.train_h7_reference_combat.cell_for_source", lambda _: None,
    )
    with pytest.raises(ValueError, match="uncurated"):
        terminal_replay_audit_indices(("out-of-curriculum-source",), 8)


def test_collect_accepts_only_preselected_unchecked_indices(monkeypatch):
    seeds = source_bank("train", 32, study_seed=49)
    audited = terminal_replay_audit_indices(seeds, 8)
    selected = set(audited)
    captured = {}

    def stub_batch(incoming, **kwargs):
        assert incoming == seeds
        assert kwargs["draw_mode"] == "reference"
        assert kwargs["terminal_replay_indices"] == audited
        captured["arguments"] = kwargs
        episodes = tuple(
            SimpleNamespace(
                source_seed=seed, scenario_id=cell_for_source(seed).scenario,
                deck_preset_id=cell_for_source(seed).deck,
                draw_mode="reference",
                reward_version="h4-boss-progress-v1",
                outcome="victory",
                terminal_replay_verified=index in selected,
            )
            for index, seed in enumerate(seeds)
        )
        return {
            "episodes": episodes,
            "terminal_replay_verified": len(audited),
            "terminal_replay_not_checked": len(seeds)-len(audited),
        }

    from sts2_ai.nature.h4_reward import H4_REWARD
    monkeypatch.setattr("tools.train_h7_reference_combat.draw_only_batch", stub_batch)
    args = SimpleNamespace(
        emulator_dll=SimpleNamespace(resolve=lambda: "/tmp/emulator.dll"),
        emulator_bindings=SimpleNamespace(), emulator_revision="test",
        workers=15, max_boundaries=1200, train_terminal_replay_every=8,
    )
    from pathlib import Path
    args.emulator_bindings = Path("/tmp/emulator-bindings")
    monkeypatch.setattr(
        "tools.train_h7_reference_combat.PpoDrawNature",
        lambda **kwargs: SimpleNamespace(),
    )
    spec = SimpleNamespace()
    # The mock retains the exact reward version requested by the collector.
    def corrected_stub(incoming, **kwargs):
        result = stub_batch(incoming, **kwargs)
        return {
            **result,
            "episodes": tuple(
                SimpleNamespace(
                    **{**vars(ep), "reward_version": H4_REWARD}
                )
                for ep in result["episodes"]
            ),
        }
    monkeypatch.setattr("tools.train_h7_reference_combat.draw_only_batch", corrected_stub)
    outcome = collect(seeds, spec=spec, args=args, reference_seed=10, capture=True)
    assert outcome["terminal_replay_verified"] == 16
    assert outcome["terminal_replay_not_checked"] == 16
    assert len(captured["arguments"]["terminal_replay_indices"]) == 16

    # Any replay status drift, whether in the audited or unchecked part,
    # is rejected before PPO sees an episode.
    def corrupt(incoming, **kwargs):
        result = corrected_stub(incoming, **kwargs)
        wrong = list(result["episodes"])
        wrong[0] = SimpleNamespace(**{
            **vars(wrong[0]), "terminal_replay_verified": False,
        })
        return {**result, "episodes": tuple(wrong)}

    monkeypatch.setattr("tools.train_h7_reference_combat.draw_only_batch", corrupt)
    with pytest.raises(RuntimeError, match="incompatible reference-law"):
        collect(seeds, spec=spec, args=args, reference_seed=10, capture=True)
