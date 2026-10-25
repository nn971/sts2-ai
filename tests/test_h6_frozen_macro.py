"""H6 strategy-only PPO, frozen-backbone provenance and public sampling tests."""
from __future__ import annotations

import random

import pytest
from test_phase_split_selfplay import TwoPhaseToy, small_model

from sts2_ai.models.neural import NEURAL_FORMAT, TACTICAL_FORMAT
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.frozen_macro import (
    FrozenCombatMacroPPO,
    combat_digest,
    macro_only_episode,
)
from sts2_ai.training.h6_sources import run_sources, source_digest
from sts2_ai.training.selfplay import collect_public_episode

torch = pytest.importorskip("torch")


def start_model() -> PhaseSplitNeuralModel:
    return PhaseSplitNeuralModel(
        strategy=small_model(NEURAL_FORMAT),
        combat=small_model(TACTICAL_FORMAT),
        model_id="h6-test-start",
    )


def test_disjoint_run_source_banks():
    banks = {
        split: run_sources(split, 16, study_seed=43)
        for split in ("train", "selection", "confirmation")
    }
    assert not set(banks["train"]) & set(banks["selection"])
    assert not set(banks["train"]) & set(banks["confirmation"])
    assert not set(banks["selection"]) & set(banks["confirmation"])
    assert set(run_sources("train", 16, round_index=0)).isdisjoint(
        run_sources("train", 16, round_index=1)
    )
    assert source_digest(banks["train"]) == source_digest(banks["train"])
    with pytest.raises(ValueError):
        run_sources("confirmation", 16, round_index=1)


def test_macro_ppo_updates_only_strategy_and_restores():
    baseline = start_model()
    digest = combat_digest(baseline)
    learner = FrozenCombatMacroPPO(
        baseline, torch=torch, epochs=2, batch_size=4,
    )
    assert learner.model().combat.to_dict() == baseline.combat.to_dict()
    outcomes = []
    for i in range(12):
        episode = collect_public_episode(
            TwoPhaseToy(), learner.model(),
            seed=f"h6-toy-{i}",
            actor_rng=random.Random(i),
            max_decisions=8,
            combat_greedy=True,
            temperature=0.85,
        )
        assert episode.completed
        assert sum(d.phase == "combat" for d in episode.decisions) == 1
        projected = macro_only_episode(episode)
        assert len(projected.decisions) == 1
        assert projected.decisions[0].phase == "strategy"
        assert projected.combat_outcomes == ()
        outcomes.append((episode, float(episode.won)))
    result = learner.update(outcomes, seed=91)
    assert result["combat_optimizer_steps"] == 0
    assert result["strategy_decisions"] == 12
    assert result["ppo_steps"] > 0
    assert learner.revision == 1
    assert combat_digest(learner.model()) == digest
    assert learner.model().combat.to_dict() == baseline.combat.to_dict()
    assert learner.model().strategy.to_dict() != baseline.strategy.to_dict()
    saved = learner.dump()
    restored = FrozenCombatMacroPPO(
        baseline, torch=torch, epochs=2, batch_size=4,
    )
    restored.restore(saved)
    assert restored.revision == 1
    assert restored.model().to_dict() == learner.model().to_dict()
    assert restored.frozen_sha256 == digest


def test_censored_runs_are_not_optimized():
    learner = FrozenCombatMacroPPO(start_model(), torch=torch)
    episode = collect_public_episode(
        TwoPhaseToy(), learner.model(), seed="h6-censored",
        actor_rng=random.Random(3), max_decisions=1,
        combat_greedy=True,
    )
    assert not episode.completed and episode.outcome == "truncated"
    with pytest.raises(ValueError, match="censored"):
        learner.update([(episode, 0.0)], seed=1)


def test_collector_passes_a10_and_supports_greedy_macro():
    class AscensionToy(TwoPhaseToy):
        def reset(self, seed: str, ascension: int = 0):
            self.last_ascension = ascension
            return super().reset(seed, ascension)

    backend = AscensionToy()
    episode = collect_public_episode(
        backend, start_model(), seed="h6-a10-toy",
        actor_rng=random.Random(2), max_decisions=8,
        ascension=10, combat_greedy=True, strategy_greedy=True,
    )
    assert backend.last_ascension == 10
    assert episode.completed
    assert len(episode.decisions) == 2
    assert not backend.states
    with pytest.raises(ValueError, match="ascension"):
        collect_public_episode(
            backend, start_model(), seed="bad-a",
            actor_rng=random.Random(3), ascension=11,
        )
