"""H3 reference-law multiple importance sampling and draw-provenance tests."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import pytest
from test_nature_contextual_draw import visible
from test_nature_draw_only_law import pending

from sts2_ai.nature.draw_only_episode import DrawOnlyStep
from sts2_ai.nature.draw_only_p0 import DrawOnlyRouter, SeededLawSampler
from sts2_ai.nature.ppo_draw import PpoDrawNature
from sts2_ai.nature.reference_importance import (
    NatureImportanceError,
    balance_weight,
    log_q_over_p,
    reference_target_weights,
)


@dataclass(frozen=True)
class FakeEpisode:
    draw_mode: str
    terminal_replay_verified: bool
    controlled_draw_count: int
    decisions: tuple[DrawOnlyStep, ...]


def episode(mode: str, q_over_p: float, *, draws: int = 1):
    step = DrawOnlyStep(
        phase="nature:draw_card", support_id="law",
        selected_id="card_instance:7", result_fingerprint="child",
        log_p=math.log(0.5), log_q=math.log(0.5 * q_over_p),
    )
    return FakeEpisode(
        draw_mode=mode, terminal_replay_verified=True,
        controlled_draw_count=draws,
        decisions=(step,) * draws,
    )


def test_balance_heuristic_exactly_recovers_reference_expectation():
    # p(1)=.5, q(1)=.9, p(0)=.5, q(0)=.1.
    # The proposal is .25 p + .75 q, yielding (.8,.2).
    alpha = 0.25
    mix_one = alpha * 0.5 + (1 - alpha) * 0.9
    mix_zero = alpha * 0.5 + (1 - alpha) * 0.1
    assert mix_one == pytest.approx(0.8)
    assert mix_zero == pytest.approx(0.2)
    assert (
        mix_one * balance_weight( math.log(0.9/0.5), alpha)
        + mix_zero * balance_weight(math.log(0.1/0.5), alpha)
    ) == pytest.approx(1.0)
    # The reward R=1 only on the harmful/rare outcome under q.
    assert mix_zero * balance_weight(
        math.log(0.1/0.5), alpha
    ) == pytest.approx(0.5)
    assert all(
        balance_weight(math.log(r), alpha) <= 1 / alpha
        for r in (1e-200, 0.1, 1, 1e10, 1e200)
    )


def test_full_combat_product_and_reference_ess_fallback():
    episodes = tuple(
        episode("reference", 2.0) for _ in range(2)
    ) + tuple(episode("learned", 0.25) for _ in range(6))
    values, report = reference_target_weights(
        episodes, reference_count=2, min_effective_fraction=0.1,
    )
    assert len(values) == 8
    assert all(0 < w <= 4 for w in values)
    assert report["player_is_effective_samples"] <= 8
    assert report["player_is_reference_only_fallback"] == 0.0
    assert report["player_is_nonreference_used"] == 6
    assert log_q_over_p(episode("learned", 0.25, draws=3)) == pytest.approx(
        (3 * math.log(.25), 3)
    )
    vals, fallback = reference_target_weights(
        episodes, reference_count=2, min_effective_fraction=1.0,
    )
    assert vals == (1., 1., 0., 0., 0., 0., 0., 0.)
    assert fallback["player_is_reference_only_fallback"] == 1.
    assert fallback["player_is_effective_samples"] == 2


def test_source_likelihood_invariants_reject_missing_scores():
    broken = episode("reference", 1.0)
    with pytest.raises(NatureImportanceError, match="Both p and q"):
        log_q_over_p(replace(
            broken, decisions=(replace(broken.decisions[0], log_q=None),)
        ))
    with pytest.raises(NatureImportanceError, match="Missing physical"):
        log_q_over_p(replace(broken, controlled_draw_count=2))
    with pytest.raises(NatureImportanceError, match="allocation"):
        reference_target_weights((broken, episode("learned", 1.0)),
                                 reference_count=0)


def test_reference_arm_scores_learned_q_without_using_it_to_draw():
    nature = PpoDrawNature(seed=15)
    router = DrawOnlyRouter(nature, SeededLawSampler(19))
    ref = router.choose(
        pending(), fingerprint="parent", episode_key="world",
        ordinal=0, draw_mode="reference", score_reference_q=True,
        public_observation=visible(),
    )
    assert ref.controller == "engine-law-reference"
    assert ref.nature_choice is None
    assert ref.log_p == pytest.approx(math.log(.5))
    raw = pending()
    from sts2_ai.emulator.semantic_p0_combat import _pending_event
    event, features, _ids = _pending_event(raw, "parent")
    _, scored = nature.propose(
        event, context_key=f"{raw['siteId']}|{event.owner_id}",
        candidate_features=features, episode_key="world:0",
        public_observation=visible(),
    )
    assert ref.log_q == pytest.approx(
        math.log(scored.probabilities[ref.selected_index]), abs=1e-12,
    )
    # The exact same p outcome is selected whether or not q was queried.
    usual = DrawOnlyRouter(nature, SeededLawSampler(19)).choose(
        pending(), fingerprint="parent", episode_key="world",
        ordinal=0, draw_mode="reference",
    )
    assert ref.selected_outcome_id == usual.selected_outcome_id
    assert usual.log_p == usual.log_q
