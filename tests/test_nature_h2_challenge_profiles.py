"""H2 conditioned combat contexts, source replay and split isolation."""
from __future__ import annotations

from dataclasses import replace

import pytest

from sts2_ai.nature.combat_scenarios import (
    BASELINE_SCENARIO,
    SCENARIOS,
    parse_scenario_pool,
    scenario_for_seed,
)
from sts2_ai.nature.draw_only_episode import play_draw_only_episode
from sts2_ai.nature.two_site_opponent import TwoSiteNature
from test_nature_draw_only_player_choice import FakeSession
from tools.train_draw_only_nature_h1 import (
    CHALLENGE_SCHEMA,
    SCHEMA,
    paired_holdout_metrics,
    source_seed,
)


def test_h2_explicit_eligible_single_enemy_encounters_and_ascension():
    assert set(SCENARIOS) == {
        "shrinker-a0", "shrinker-a10", "nibbits-a0", "nibbits-a10"
    }
    assert SCENARIOS[BASELINE_SCENARIO].ascension == 0
    assert SCENARIOS["nibbits-a10"].native_entry == "NibbitsWeak"
    assert SCENARIOS["nibbits-a10"].ascension == 10
    assert CHALLENGE_SCHEMA != SCHEMA


def test_h2_pool_assignment_is_deterministic_and_order_independent():
    a = parse_scenario_pool("shrinker-a0,nibbits-a10")
    b = parse_scenario_pool("nibbits-a10,shrinker-a0")
    seeds = tuple(source_seed("train", i, training_seed=29, challenge=True)
                  for i in range(200))
    mapped = tuple(scenario_for_seed(seed, a).scenario_id for seed in seeds)
    assert mapped == tuple(scenario_for_seed(seed, b).scenario_id for seed in seeds)
    assert set(mapped) == set(a)
    assert scenario_for_seed("anything", (BASELINE_SCENARIO,)).scenario_id == (
        BASELINE_SCENARIO
    )


@pytest.mark.parametrize("bad", [
    "", "nibbits-a0,nibbits-a0", "nibbits-a11", "shrinker-a0,",
    "nibbits-a0,unknown",
])
def test_invalid_pools_fail_closed(bad):
    with pytest.raises(ValueError):
        parse_scenario_pool(bad)


def test_h2_train_and_holdout_seeds_differ_from_h1_and_stay_fixed_between_restarts():
    for i in range(20):
        a = source_seed("evaluation", i, training_seed=29, challenge=True)
        b = source_seed("evaluation", i, training_seed=101, challenge=True)
        c = source_seed("evaluation", i, training_seed=29, challenge=False)
        assert a == b
        assert a != c
        assert a != source_seed("train", i, training_seed=29, challenge=True)


class CapturingSession(FakeSession):
    def __init__(self):
        self.kwargs = None

    def begin_seeded_semantic_combat(self, seed, **kwargs):
        self.kwargs = kwargs
        return super().begin_seeded_semantic_combat(seed, **kwargs)


def test_explicit_scenario_controls_engine_encounter_but_not_chance_controller():
    s = CapturingSession()
    options = dict(
        source_seed="challenge-world",
        reference_seed=88,
        nature=TwoSiteNature(seed=8),
        emulator_revision="pinned",
        player_choose=lambda _, actions: actions[0]["actionId"],
        entry_mode="seeded-fixed-shrinker",
        draw_mode="reference",
        scenario=SCENARIOS["nibbits-a10"],
    )
    played = play_draw_only_episode(s, **options)
    assert played.scenario_id == "nibbits-a10"
    assert s.kwargs is not None
    assert s.kwargs["encounter_id"] == "proto.encounter.nibbits_weak"
    assert s.kwargs["native_entry"] == "NibbitsWeak"
    assert s.kwargs["ascension"] == 10
    assert s.kwargs["replay_opening_draw_hooks"] is True
    assert "fixture_selection_card_id" not in s.kwargs
    assert played.digest == play_draw_only_episode(
        s, **{**options, "replay": played.decisions}
    ).digest


def test_source_route_rejects_altered_conditioned_fixture():
    with pytest.raises(ValueError, match="Invalid draw-only fixture"):
        play_draw_only_episode(
            FakeSession(), source_seed="x", reference_seed=2,
            nature=TwoSiteNature(seed=5), emulator_revision="source",
            entry_mode="unrecognized", scenario=SCENARIOS["nibbits-a10"],
        )
    from sts2_ai.nature.draw_only_p0 import DrawOnlyContractError

    with pytest.raises(DrawOnlyContractError, match="Source-route"):
        play_draw_only_episode(
            FakeSession(), source_seed="x", reference_seed=2,
            nature=TwoSiteNature(seed=5), emulator_revision="source",
            entry_mode="source-route", scenario=SCENARIOS["nibbits-a10"],
        )


def test_paired_holdout_refuses_different_challenge_world_ids():
    from test_nature_draw_only_h1 import episode

    a = replace(episode("same", mode="learned", score=0.4),
                scenario_id="nibbits-a10")
    b = replace(episode("same", mode="reference", score=0.8),
                scenario_id="shrinker-a0")
    with pytest.raises(ValueError, match="differ"):
        paired_holdout_metrics([a], [b], bootstrap_seed=1, replicates=100)
