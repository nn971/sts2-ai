"""v15p: paired checkpoint analysis, stochastic reproducibility, PPO diagnostics."""
from __future__ import annotations

from dataclasses import replace

import pytest
from test_combat_snapshots_v14 import SimpleAgent, ToyBackend
from test_object_instances_v15 import frame
from test_pooled_ppo_v15 import combat_episode, frozen_policy
from test_snapshot_curriculum_v15m import authentic_recipe

from sts2_ai.evaluation.combat_diagnostics import diagnose_reports, render_markdown
from sts2_ai.evaluation.combat_snapshots import evaluate_model_suite, pair_model_suite
from sts2_ai.evaluation.stochastic_combat import (
    StochasticCombatAgent,
    snapshot_actor_seed,
)
from sts2_ai.models.object_embeddings import ObjectVocabulary
from sts2_ai.models.object_instances import extract_combat_objects
from sts2_ai.models.pooled_objects import PooledObjectTorch
from sts2_ai.models.protocol import PolicyValueEstimate
from sts2_ai.training.pooled_ppo_v15 import (
    ppo_update_pooled,
    prepare_pooled_samples,
)


def fake_report(round_index: int, hp: tuple[int, ...], *,
                corpus: str = "fingerprint", reverse: bool = False):
    entries = []
    for i, score in enumerate(hp):
        entries.append({
            "seed": f"source-{i // 2}",
            "floor": i + 1, "tier": ("normal", "elite")[i % 2],
            "progress": "middle",
            "enemy_ids": [f"enemy-{i}"],
            "baseline": {
                "outcome": "victory", "hp_remaining": 10,
                "potions_remaining": [], "decisions": 7,
                "enemy_hp_remaining": 0,
            },
            "candidate": {
                "outcome": "victory" if score > 0 else "defeat",
                "hp_remaining": score,
                "potions_remaining": [], "decisions": 6,
                "enemy_hp_remaining": 0,
            },
        })
    if reverse:
        entries.reverse()
    return {
        "corpus": "test.jsonl", "baseline_model": "base.json",
        "candidate_model": f"stage-{round_index}.json",
        "emulator_revision": "pinned",
        "evaluation": {
            "identity": {
                "corpus_sha256": corpus, "baseline_sha256": "basehash",
                "emulator_revision": "pinned", "max_combat_decisions": 256,
            },
        },
        "rows": entries,
    }


def test_checkpoint_diagnostic_is_paired_and_deterministic() -> None:
    early = fake_report(20, (7, 8, 6, 7))
    late = fake_report(200, (9, 9, 8, 8))
    a = diagnose_reports({20: early, 200: late}, bootstrap_samples=80)
    b = diagnose_reports({200: late, 20: early}, bootstrap_samples=80)
    assert a == b
    assert a["source_runs"] == 2 and a["scenarios"] == 4
    assert a["checkpoints"][0]["delta_exit_hp"] == pytest.approx(-3)
    assert a["checkpoints"][1]["delta_exit_hp"] == pytest.approx(-1.5)
    assert a["first_to_last"]["delta_exit_hp"] == pytest.approx(1.5)
    assert a["first_to_last"]["delta_win_rate"] == 0
    assert len(a["first_to_last"]["cluster_95pct_delta_hp"]) == 2
    assert "Last checkpoint by tier" in render_markdown(a)


def test_checkpoint_diagnostic_rejects_mismatched_reports() -> None:
    good = fake_report(20, (5, 6, 7, 8))
    later = fake_report(200, (5, 6, 7, 8))
    with pytest.raises(ValueError, match="different corpus"):
        diagnose_reports({20: good, 200: fake_report(200, (5, 6, 7, 8), corpus="other")})
    with pytest.raises(ValueError, match="order/identity mismatch"):
        diagnose_reports({20: good, 200: fake_report(200, (5, 6, 7, 8), reverse=True)})
    later["rows"][0]["baseline"]["hp_remaining"] = 8
    with pytest.raises(ValueError, match="Baseline results differ"):
        diagnose_reports({20: good, 200: later})


class ToyStochasticModel:
    model_id = "fixed-stochastic"

    def evaluate(self, observation, legal_actions):
        return PolicyValueEstimate(tuple(0.0 for _ in legal_actions), 0.5)


def test_stochastic_policy_reproducible_seeded_action_choices() -> None:
    backend = ToyBackend()
    recipe = authentic_recipe(backend)
    assert snapshot_actor_seed(recipe, 0) == snapshot_actor_seed(recipe, 0)
    assert snapshot_actor_seed(recipe, 0) != snapshot_actor_seed(recipe, 1)
    assert snapshot_actor_seed(recipe, 0) != snapshot_actor_seed(
        replace(recipe, exact_hash="different"), 0,
    )
    with pytest.raises(ValueError, match="nonnegative"):
        snapshot_actor_seed(recipe, -1)
    with pytest.raises(ValueError, match="temperature"):
        StochasticCombatAgent(ToyStochasticModel(), seed=1, temperature=0)
    agents = {
        "a": SimpleAgent("attack"),
        "b": SimpleAgent("fumble"),
    }

    def factory(label, entry):
        return StochasticCombatAgent(
            ToyStochasticModel(), seed=snapshot_actor_seed(entry, 7),
            temperature=1.0,
        )

    first = evaluate_model_suite(
        backend, [recipe], agents, agent_factory=factory,
        max_combat_decisions=4,
    )
    second = evaluate_model_suite(
        backend, [recipe], agents, agent_factory=factory,
        max_combat_decisions=4,
    )
    assert first == second
    assert first[0].outcomes["a"] == first[0].outcomes["b"]
    assert pair_model_suite(first, "a", "b")[0].baseline.outcome in ("victory", "defeat")
    assert backend.states == {}


def test_opt_in_ppo_gradient_diagnostics_have_finite_metrics() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(15)
    base = frozen_policy()
    vocabulary = ObjectVocabulary.from_object_sets([extract_combat_objects(frame())])
    head = PooledObjectTorch(vocabulary, width=16)
    samples, _ = prepare_pooled_samples(
        [combat_episode()], base.combat, head, temperature=0.8,
        gamma=0.995, gae_lambda=0.98, episode_weights=[1.25],
    )
    optimizer = torch.optim.AdamW(head.module.parameters(), lr=0.0003)
    diag = ppo_update_pooled(
        head, samples, optimizer, temperature=0.8, epochs=1,
        batch_size=2, diagnostics=True, seed=8,
    )
    assert diag.steps == 1
    assert diag.mean_preclip_grad_norm is not None
    assert diag.mean_preclip_grad_norm >= 0
    assert diag.max_preclip_grad_norm == pytest.approx(diag.mean_preclip_grad_norm)
    assert 0 <= diag.gradient_clip_fraction <= 1
    assert diag.raw_advantage_std >= 0
    assert 0 < diag.decision_weight_ess_fraction <= 1
    normal = ppo_update_pooled(
        head, samples, optimizer, temperature=0.8, epochs=1,
        batch_size=2, diagnostics=False, seed=8,
    )
    assert normal.mean_preclip_grad_norm is None
    assert normal.raw_advantage_std is None
