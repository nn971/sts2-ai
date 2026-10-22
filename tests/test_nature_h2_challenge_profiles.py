"""H2 conditioned combat contexts, source replay and split isolation."""
from __future__ import annotations

from dataclasses import replace

import pytest
from test_nature_draw_only_player_choice import FakeSession

from sts2_ai.nature.combat_scenarios import (
    BASELINE_DECK,
    BASELINE_SCENARIO,
    DECK_PRESETS,
    SCENARIOS,
    deck_for_seed,
    parse_deck_pool,
    parse_scenario_pool,
    scenario_for_seed,
)
from sts2_ai.nature.draw_only_episode import play_draw_only_episode
from sts2_ai.nature.two_site_opponent import TwoSiteNature
from tools.train_draw_only_nature_h1 import (
    CHALLENGE_SCHEMA,
    SCHEMA,
    paired_holdout_metrics,
    source_seed,
)


def test_h2_explicit_eligible_single_enemy_encounters_and_ascension():
    assert set(SCENARIOS) == {
        "shrinker-a0", "shrinker-a10", "nibbits-a0", "nibbits-a10",
        "mawler-a10", "cubex-a10", "vine-shambler-a10",
        "ceremonial-beast-boss-a10", "vantom-boss-a10",
    }
    for key in ("mawler-a10", "cubex-a10", "vine-shambler-a10"):
        assert SCENARIOS[key].ascension == 10
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


def test_h2_holdout_reports_source_paired_deltas_by_enemy():
    from test_nature_draw_only_h1 import episode

    nature = [
        replace(episode("source-n", mode="learned", score=0.2),
                scenario_id="nibbits-a10"),
        replace(episode("source-s", mode="learned", score=0.5),
                scenario_id="shrinker-a10"),
    ]
    reference = [
        replace(episode("source-n", mode="reference", score=0.7),
                scenario_id="nibbits-a10"),
        replace(episode("source-s", mode="reference", score=0.8),
                scenario_id="shrinker-a10"),
    ]
    result = paired_holdout_metrics(
        nature, reference, bootstrap_seed=991, replicates=100,
    )
    assert result["paired_nature_minus_reference_score"] == pytest.approx(-0.4)
    assert set(result["paired_by_scenario"]) == {"nibbits-a10", "shrinker-a10"}
    assert result["paired_by_scenario"]["nibbits-a10"][
        "mean_nature_minus_reference_score"
    ] == pytest.approx(-0.5)
    assert result["paired_by_scenario"]["shrinker-a10"][
        "mean_nature_minus_reference_score"
    ] == pytest.approx(-0.3)


def test_h2_deck_and_encounter_worlds_are_reproducible_and_independent():
    pool = parse_deck_pool("silent-poison-kit-v1,silent-strike-heavy-v1")
    assert len(DECK_PRESETS) == 6
    assert deck_for_seed("world", (BASELINE_DECK,)) == BASELINE_DECK
    seeds = tuple(source_seed("evaluation", i, training_seed=29, challenge=True)
                  for i in range(120))
    selected = tuple(deck_for_seed(seed, pool) for seed in seeds)
    assert set(selected) == set(pool)
    assert selected == tuple(deck_for_seed(seed, tuple(reversed(pool)))
                             for seed in seeds)
    for bad in ("", "silent-starter-v1,", "unknown", "silent-starter-v1,silent-starter-v1"):
        with pytest.raises(ValueError):
            parse_deck_pool(bad)


def test_h2_engine_receives_true_persistent_deck_preset_before_shuffle():
    session = CapturingSession()
    kw = dict(
        source_seed="choice", reference_seed=10, nature=TwoSiteNature(seed=11),
        emulator_revision="pinned", player_choose=lambda _, a: a[0]["actionId"],
        draw_mode="reference", entry_mode="seeded-fixed-shrinker",
        scenario=SCENARIOS["nibbits-a10"],
        deck_preset_id="silent-poison-kit-v1",
    )
    a = play_draw_only_episode(session, **kw)
    assert session.kwargs["fixture_deck_preset_id"] == "silent-poison-kit-v1"
    assert a.deck_preset_id == "silent-poison-kit-v1"
    assert a.digest == play_draw_only_episode(
        session, replay=a.decisions, **kw
    ).digest


def test_h2_holdout_rejects_deck_identity_mismatch():
    from test_nature_draw_only_h1 import episode

    a = replace(episode("s", mode="learned", score=0.4),
                deck_preset_id="silent-poison-kit-v1")
    b = replace(episode("s", mode="reference", score=0.6),
                deck_preset_id="silent-starter-v1")
    with pytest.raises(ValueError, match="differ"):
        paired_holdout_metrics([a], [b], bootstrap_seed=1, replicates=100)


def test_h2_holdout_batch_runs_identical_scenario_and_deck_pools(monkeypatch):
    """Learned and reference worlds must keep identical A10 context arms."""
    import tools.train_draw_only_nature_h1 as trainer

    seeds = ("holdout-a", "holdout-b")
    scenario_pool = ("mawler-a10", "cubex-a10", "vine-shambler-a10")
    deck_pool = ("silent-defend-heavy-v1", "silent-poison-kit-v1")
    calls = []

    def fake_batch(received_seeds, *, draw_mode, **kwargs):
        calls.append((received_seeds, draw_mode, kwargs))
        return {
            "source_count": len(received_seeds),
            "scenario_counts": {"mawler-a10": 2},
            "deck_preset_counts": {"silent-defend-heavy-v1": 2},
        }

    monkeypatch.setattr(trainer, "draw_only_batch", fake_batch)
    left, right = trainer.collect_paired_holdout(
        seeds, nature=TwoSiteNature(seed=1), spec=None,
        emulator_command=("dotnet", "pinned-emulator.dll"),
        emulator_bindings=None, emulator_revision="pinned",
        reference_seed=42, workers=15, max_boundaries=600,
        scenario_pool=scenario_pool, deck_pool=deck_pool,
    )
    assert left == right
    assert [mode for _, mode, _ in calls] == ["learned", "reference"]
    for seen_seeds, _, options in calls:
        assert seen_seeds == seeds
        assert options["scenario_pool"] == scenario_pool
        assert options["deck_pool"] == deck_pool
        assert options["reference_seed"] == 42

    def shifted_reference(received_seeds, *, draw_mode, **kwargs):
        result = fake_batch(received_seeds, draw_mode=draw_mode, **kwargs)
        if draw_mode == "reference":
            result["deck_preset_counts"] = {"silent-starter-v1": 2}
        return result

    monkeypatch.setattr(trainer, "draw_only_batch", shifted_reference)
    with pytest.raises(RuntimeError, match="reference context disagrees"):
        trainer.collect_paired_holdout(
            seeds, nature=TwoSiteNature(seed=1), spec=None,
            emulator_command=("dotnet", "pinned-emulator.dll"),
            emulator_bindings=None, emulator_revision="pinned",
            reference_seed=42, workers=15, max_boundaries=600,
            scenario_pool=scenario_pool, deck_pool=deck_pool,
        )
