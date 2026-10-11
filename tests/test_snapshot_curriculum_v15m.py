"""v15m: authentic snapshot sampling, replay, and independent train/dev groups."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest
from test_combat_snapshots_v14 import FAIR, ToyBackend

from sts2_ai.agents.base import Decision
from sts2_ai.evaluation.combat_snapshots import SnapshotCollector
from sts2_ai.models.protocol import PolicyValueEstimate
from sts2_ai.training.snapshot_curriculum import (
    collect_snapshot_cohort,
    sample_snapshot_curriculum,
    source_key,
    split_snapshot_sources,
)


class ActionModel:
    """The policy sees only a public frame and its legal menu."""

    model_id = "snapshot-toy"

    def __init__(self, action: str = "attack") -> None:
        self.action = action

    def evaluate(self, observation, legal_actions):
        assert observation.policy_id == "prototype-fair-v0"
        assert isinstance(json.loads(observation.payload_json), dict)
        return PolicyValueEstimate(
            tuple(100.0 if a.action_id == self.action else -100.0
                  for a in legal_actions),
            0.5,
        )


def authentic_recipe(backend: ToyBackend, seed: str = "test-source"):
    root = backend.reset_native_overgrowth(seed)
    try:
        go = backend.legal_actions(root)[0]
        battle = backend.step(root, go).child
        try:
            attack = backend.legal_actions(battle)[0]
            collector = SnapshotCollector(
                backend, seed=seed, source_policy="fixed-source",
            )
            collector.on_decision(
                root, backend.observe(root, FAIR), (go,),
                Decision(go, "fixed-source"),
            )
            collector.on_decision(
                battle, backend.observe(battle, FAIR),
                backend.legal_actions(battle), Decision(attack, "fixed-source"),
            )
            return collector.recipes[0]
        finally:
            backend.release_many([battle])
    finally:
        backend.release_many([root])


def test_source_run_split_is_stable_disjoint_and_immutable() -> None:
    backend = ToyBackend()
    template = authentic_recipe(backend)
    entries = [
        replace(template, seed=f"source-{i // 2}",
                exact_hash=f"exact-{i}") for i in range(80)
    ]
    train, development = split_snapshot_sources(
        entries, holdout_fraction=0.3, salt="reproducible",
    )
    reversed_train, reversed_dev = split_snapshot_sources(
        list(reversed(entries)), holdout_fraction=0.3, salt="reproducible",
    )
    assert train == reversed_train and development == reversed_dev
    assert {source_key(r) for r in train}.isdisjoint(
        {source_key(r) for r in development}
    )
    assert {r.exact_hash for r in train + development} == {
        r.exact_hash for r in entries
    }
    with pytest.raises(ValueError, match="Duplicated"):
        split_snapshot_sources(entries + [entries[0]])
    with pytest.raises(ValueError, match="at least two"):
        split_snapshot_sources([entries[0]])
    assert backend.states == {}


def test_balanced_progress_tier_selection_and_natural_mode() -> None:
    backend = ToyBackend()
    template = authentic_recipe(backend)
    weak = [
        replace(template, seed=f"weak-{i}", exact_hash=f"weak-{i}")
        for i in range(18)
    ]
    elite = replace(
        template, seed="elite", floor=8, progress="middle",
        tier="elite", exact_hash="elite",
    )
    boss = replace(
        template, seed="boss", floor=16, progress="late",
        tier="boss", exact_hash="boss",
    )
    recipes = weak + [elite, boss]
    selected = sample_snapshot_curriculum(
        recipes, count=30, seed=17, mode="balanced",
    )
    assert selected == sample_snapshot_curriculum(
        recipes, count=30, seed=17, mode="balanced",
    )
    assert {
        (r.progress, r.tier) for r in selected
    } == {("early", "weak"), ("middle", "elite"), ("late", "boss")}
    assert sum(r.tier == "boss" for r in selected) == 10
    assert sum(r.tier == "elite" for r in selected) == 10
    natural = sample_snapshot_curriculum(
        recipes, count=len(recipes), seed=17, mode="natural",
    )
    assert {r.exact_hash for r in natural} == {r.exact_hash for r in recipes}
    with pytest.raises(ValueError, match="Duplicated"):
        sample_snapshot_curriculum(recipes + [boss], count=1, seed=1)
    assert backend.states == {}


def test_snapshot_cohort_replays_once_and_produces_onpolicy_combat_segments() -> None:
    class CountingBackend(ToyBackend):
        def __init__(self):
            super().__init__()
            self.prefix_actions = 0

        def step(self, handle, action):
            if action.action_id == "go":
                self.prefix_actions += 1
            return super().step(handle, action)

    backend = CountingBackend()
    recipe = authentic_recipe(backend)
    baseline = backend.prefix_actions
    episodes = collect_snapshot_cohort(
        backend, ActionModel("attack"), [recipe] * 4,
        round_index=3, actor_seed=91, temperature=0.9, max_decisions=4,
    )
    assert len(episodes) == 4
    assert backend.prefix_actions == baseline + 1
    assert len({episode.seed for episode in episodes}) == 4
    assert all(episode.completed and episode.won for episode in episodes)
    assert all(len(ep.decisions) == 1 and
               ep.decisions[0].phase == "combat" for ep in episodes)
    assert all(len(ep.combat_outcomes) == 1 for ep in episodes)
    assert all(ep.combat_outcomes[0].start_decision == 0 and
               ep.combat_outcomes[0].end_decision == 1 and
               ep.combat_outcomes[0].exit.hp == 21 for ep in episodes)
    assert backend.states == {}

    defeated = collect_snapshot_cohort(
        backend, ActionModel("fumble"), [recipe],
        round_index=4, actor_seed=91, temperature=1.0,
    )[0]
    assert defeated.completed and defeated.outcome == "defeat"
    assert defeated.combat_outcomes[0].result == "defeat"
    assert backend.states == {}


def test_snapshot_failure_is_fatal_and_releases_every_handle() -> None:
    backend = ToyBackend()
    recipe = authentic_recipe(backend)
    with pytest.raises(ValueError, match="diverged"):
        collect_snapshot_cohort(
            backend, ActionModel(), [
                replace(recipe, observation_hash="corrupted"),
            ], round_index=0, actor_seed=1, temperature=1.0,
        )
    assert backend.states == {}
    with pytest.raises(ValueError, match="revision mismatch"):
        collect_snapshot_cohort(
            backend, ActionModel(),
            [replace(recipe, emulator_revision="wrong-revision")],
            round_index=0, actor_seed=1, temperature=1.0,
        )
    assert backend.states == {}


def test_portable_relational_warm_start_and_rejected_mismatch() -> None:
    torch = pytest.importorskip("torch")
    from test_pooled_ppo_v15 import frozen_policy
    from test_object_instances_v15 import frame
    from sts2_ai.models.object_embeddings import ObjectVocabulary
    from sts2_ai.models.object_instances import extract_combat_objects
    from sts2_ai.models.phase_split import PhaseSplitNeuralModel
    from sts2_ai.models.relational_objects import RelationalObjectTorch
    from sts2_ai.training.pooled_ppo_v15 import initialize_combat_head

    base = frozen_policy()
    vocabulary = ObjectVocabulary.from_object_sets([extract_combat_objects(frame())])
    original = RelationalObjectTorch(vocabulary, width=16)
    with torch.no_grad():
        original.module.policy_gate.fill_(0.35)
        original.module.value_gate.fill_(0.2)
    initial = PhaseSplitNeuralModel(
        base.strategy, original.export(base.combat),
        "warm-start-source", combat_value_objective="hp_preservation",
    )
    restored = RelationalObjectTorch(vocabulary, width=16)
    initialize_combat_head(restored, base, initial)
    for name, param in original.module.state_dict().items():
        assert torch.equal(param, restored.module.state_dict()[name])
    with pytest.raises(ValueError, match="objective mismatch"):
        initialize_combat_head(
            restored, base, replace(initial, combat_value_objective="hp_first"),
        )
    with pytest.raises(ValueError, match="width/vocabulary mismatch"):
        initialize_combat_head(RelationalObjectTorch(vocabulary, width=32), base, initial)


def test_native_snapshot_training_is_a_distinct_ppo_checkpoint(tmp_path) -> None:
    pytest.importorskip("torch")
    from test_pooled_ppo_v15 import frozen_policy
    from test_object_instances_v15 import frame
    from test_combat_snapshots_v14 import observation, public
    from sts2_ai.emulator import LegalAction
    from sts2_ai.emulator.episode_goal import NATIVE_ACT1_BOSS_GOAL
    from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
    from sts2_ai.models.object_embeddings import ObjectVocabulary
    from sts2_ai.models.object_instances import extract_combat_objects
    from sts2_ai.training.pooled_ppo_v15 import train_pooled_combat_ppo

    class RichToy(ToyBackend):
        def observe(self, handle, policy):
            if self.states[handle] != "battle":
                return super().observe(handle, policy)
            state = public(floor=1, room_type=0, combat=True)
            template = frame()
            state["combat"] = template["combat"]
            state["deck"] = template["deck"]
            state["relics"] = template["relics"]
            state["potions"] = template["potions"]
            state["character_id"] = template["character_id"]
            state["gold"] = template["gold"]
            return observation(state)

        def legal_actions(self, handle):
            if self.states[handle] == "battle":
                return (
                    LegalAction("attack", "play_card",
                                '{"CardInstanceId":11,"TargetEnemyId":1}'),
                    LegalAction("fumble", "end_turn", "{}"),
                )
            return super().legal_actions(handle)

    backend = RichToy()
    recipe = authentic_recipe(backend)
    base = frozen_policy()
    vocabulary = ObjectVocabulary.from_object_sets([extract_combat_objects(frame())])
    checkpoint = tmp_path / "snapshot.pt"
    options = dict(
        rounds=1, episodes_per_round=2, workers=1, seed=3,
        environment=NATIVE_OVERGROWTH, episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
        max_decisions=4, width=16, architecture="relational",
        epochs=1, batch_size=2, sample_limit=8,
        snapshot_recipes=[recipe], checkpoint=checkpoint,
    )
    model, history = train_pooled_combat_ppo(
        backend, base, vocabulary, **options,
    )
    assert history[0].rollout_source == "combat_snapshots"
    assert history[0].resolved_combats == 2
    assert history[0].ppo_samples_used == 2
    assert model.strategy.to_dict() == base.strategy.to_dict()
    assert checkpoint.is_file() and backend.states == {}
    with pytest.raises(ValueError, match="configuration/emulator mismatch"):
        train_pooled_combat_ppo(
            backend, base, vocabulary,
            **{**options, "rounds": 2, "snapshot_sampling": "natural",
               "resume": True},
        )
    resumed, rows = train_pooled_combat_ppo(
        backend, base, vocabulary,
        **{**options, "rounds": 2, "resume": True},
    )
    assert len(rows) == 2 and resumed.combat is not None
    assert backend.states == {}
