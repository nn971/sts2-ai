"""H6 regression: macro PPO must not change even one tactical weight."""
import random
from pathlib import Path

import pytest
from test_phase_split_selfplay import TwoPhaseToy, small_model

from sts2_ai.emulator import LegalAction
from sts2_ai.models.neural import NEURAL_FORMAT, TACTICAL_FORMAT
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.phase_split_selfplay import train_phase_split
from sts2_ai.training.selfplay import collect_public_episode


def test_frozen_combat_ppo_preserves_exact_export_and_checkpoint(tmp_path: Path):
    pytest.importorskip("torch")
    backbone = PhaseSplitNeuralModel(
        small_model(NEURAL_FORMAT),
        small_model(TACTICAL_FORMAT, value_bias=0.271828),
        "h6-two-phase-toy",
    )
    warm = tmp_path / "backbone.json"
    backbone.save(warm)
    ckpt = tmp_path / "macro.pt"
    opts = dict(
        backend=TwoPhaseToy(), rounds=2, episodes_per_round=24,
        dimension=32, hidden=4, max_decisions=8, workers=1,
        optimizer_method="ppo", hp_monotonic_weight=0,
        ppo_backend="batched", ppo_epochs=2,
        tactical_state_encoding="legacy",
        combat_objective="continuation",
        temperature_start=1.0, temperature_end=1.0,
        warm_start=warm, freeze_combat=True, checkpoint=ckpt,
    )
    final, rows = train_phase_split(**opts)
    assert len(rows) == 2
    assert final.combat.to_dict() == backbone.combat.to_dict()
    assert any(row.strategic_decisions for row in rows)
    assert all(row.completed > 0 for row in rows)
    resumed, later = train_phase_split(**{**opts, "rounds": 3, "resume": True})
    assert len(later) == 3
    assert resumed.combat.to_dict() == backbone.combat.to_dict()
    assert resumed.strategy.to_dict() != backbone.strategy.to_dict()


def test_freeze_requires_checkpoint_and_ppo(tmp_path: Path):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="pinned warm start"):
        train_phase_split(
            TwoPhaseToy(), rounds=1, episodes_per_round=2,
            dimension=32, hidden=4, freeze_combat=True,
            optimizer_method="ppo", hp_monotonic_weight=0,
        )


def test_frozen_macro_run_uses_explicit_ascension_ten(tmp_path: Path):
    pytest.importorskip("torch")
    requested = []

    class SpyBackend(TwoPhaseToy):
        def reset(self, seed: str, ascension: int = 0) -> str:
            requested.append(ascension)
            return super().reset(seed, ascension)

    backbone = PhaseSplitNeuralModel(
        small_model(NEURAL_FORMAT), small_model(TACTICAL_FORMAT),
        "ascension-h6-toy",
    )
    warm = tmp_path / "ascension-backbone.json"
    backbone.save(warm)
    final, rows = train_phase_split(
        SpyBackend(), rounds=1, episodes_per_round=4, workers=1,
        dimension=32, hidden=4, max_decisions=8,
        optimizer_method="ppo", ppo_backend="batched",
        hp_monotonic_weight=0, warm_start=warm, freeze_combat=True,
        ascension=10, tactical_state_encoding="legacy",
        temperature_start=1.0, temperature_end=1.0,
    )
    assert requested == [10] * 4
    assert rows[0].strategic_decisions == 4
    assert rows[0].tactical_decisions == 0
    assert final.combat.to_dict() == backbone.combat.to_dict()


def test_h6_combat_is_greedy_even_when_macro_temperature_is_high():
    class TwoActionCombat(TwoPhaseToy):
        def legal_actions(self, state):
            if self.states[state] == "combat":
                return (
                    LegalAction("attack-first", "play_card"),
                    LegalAction("attack-last", "play_card"),
                )
            return super().legal_actions(state)

    backbone = PhaseSplitNeuralModel(
        small_model(NEURAL_FORMAT), small_model(TACTICAL_FORMAT),
        "greedy-combat-backbone",
    )
    for seed in (1, 7, 401):
        run = collect_public_episode(
            TwoActionCombat(), backbone, seed=f"h6-test-{seed}",
            actor_rng=random.Random(seed), temperature=100.0,
            greedy_combat=True, max_decisions=8,
        )
        choice = run.decisions[0]
        assert choice.phase == "combat"
        assert choice.legal_actions[choice.chosen_index].action_id == "attack-last"
