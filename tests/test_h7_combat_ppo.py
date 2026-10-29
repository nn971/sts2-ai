"""H7 source-fair PPO provenance, finite updates and behavior-policy matching."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from test_shared_transformer_h7 import actions, frame, legal, model  # noqa: E402

from sts2_ai.emulator.semantic_p0_frozen_player import FrozenP0Player  # noqa: E402
from sts2_ai.models.h7_public_policy import H7PlayerSpec  # noqa: E402
from sts2_ai.training.h7_combat_ppo import H7CombatPPO  # noqa: E402


def make_cohort(tmp_path: Path):
    fresh = model()
    fresh.save_checkpoint(tmp_path / "h7-initial.pt")
    spec = H7PlayerSpec.from_file(
        tmp_path / "h7-initial.pt", seed=23, greedy=False, temperature=0.85,
    )
    choices = [
        {"actionId": str(i), "kind": item.kind, "payloadJson": item.payload_json}
        for i, item in enumerate(legal())
    ]
    episodes = []
    captures = []
    for seed in (31, 79):
        actor = FrozenP0Player(
            spec.load_pinned(), actor_seed=seed, temperature=0.85,
            greedy=False, action_scope="all-legal-v1", capture_training=True,
        )
        selected = actor.choose(frame(), choices)
        episodes.append(SimpleNamespace(
            outcome="victory" if seed == 31 else "defeat",
            score=1.65 if seed == 31 else 0.45,
            terminal_replay_verified=True, draw_mode="reference",
            decisions=(SimpleNamespace(phase="player", selected_id=selected),),
        ))
        captures.append(tuple(actor.training_actions))
    return fresh, spec, tuple(episodes), tuple(captures)


def test_h7_ppo_updates_random_policy_with_public_actor_data(tmp_path: Path):
    torch.set_num_threads(1)
    net, spec, episodes, captures = make_cohort(tmp_path)
    trainer = H7CombatPPO(net, lr=0.0001, temperature=0.85, epochs=2, minibatch=2)
    before = net.policy_heads[0][-1].weight.detach().clone()
    stats = trainer.update(
        episodes, captures, behavior_model_id=f"h7-{spec.model_sha256[:16]}",
        seed=97,
    )
    assert stats["decisions"] == 2 and stats["updates"] == 2
    assert stats["source_decisions"] == 2
    assert all(
        isinstance(value, (int, float)) and abs(value) < 100
        for value in stats.values()
    )
    assert not torch.equal(net.policy_heads[0][-1].weight, before)


def test_h7_ppo_rejects_wrong_actor_and_tampered_source(tmp_path: Path):
    net, spec, episodes, captures = make_cohort(tmp_path)
    learner = H7CombatPPO(net, temperature=0.85, epochs=1, minibatch=2)
    with pytest.raises(ValueError, match="provenance"):
        learner.update(episodes, captures, behavior_model_id="not-source-actor", seed=9)
    broken = deepcopy(captures)
    broken[0][0]["source_legal_actions"][0]["kind"] = "fake_action"
    with pytest.raises(ValueError, match="provenance"):
        learner.update(
            episodes, broken, behavior_model_id=f"h7-{spec.model_sha256[:16]}",
            seed=9,
        )


def test_h7_ppo_rejects_non_reference_draw_or_unverified_terminal(tmp_path: Path):
    net, spec, episodes, captures = make_cohort(tmp_path)
    learner = H7CombatPPO(net, temperature=0.85, epochs=1, minibatch=2)
    untrusted = list(episodes)
    untrusted[0] = SimpleNamespace(**{
        **vars(episodes[0]), "draw_mode": "learned",
    })
    with pytest.raises(ValueError, match="reference Chance Law"):
        learner.update(
            tuple(untrusted), captures,
            behavior_model_id=f"h7-{spec.model_sha256[:16]}", seed=9,
        )
    untrusted[0] = SimpleNamespace(**{
        **vars(episodes[0]), "terminal_replay_verified": False,
    })
    with pytest.raises(ValueError, match="verified terminal"):
        learner.update(
            tuple(untrusted), captures,
            behavior_model_id=f"h7-{spec.model_sha256[:16]}", seed=9,
        )


def test_h7_ppo_uses_genuine_legal_actions_not_unseen_target(tmp_path: Path):
    _, _, episodes, captures = make_cohort(tmp_path)
    assert len(captures[0][0]["source_legal_actions"]) == len(legal())
    assert {a.kind for a in actions(("end_turn", {}))} == {"end_turn"}


def test_h7_sampled_terminal_replay_requires_explicit_optimizer_opt_in(tmp_path: Path):
    torch.set_num_threads(1)
    net, spec, episodes, captures = make_cohort(tmp_path)
    unchecked = list(episodes)
    unchecked[1] = SimpleNamespace(**{
        **vars(episodes[1]), "terminal_replay_verified": False,
    })
    policy_id = f"h7-{spec.model_sha256[:16]}"
    learner = H7CombatPPO(net, temperature=0.85, epochs=1, minibatch=2)
    with pytest.raises(ValueError, match="Only verified terminal"):
        learner.update(tuple(unchecked), captures, behavior_model_id=policy_id, seed=9)
    diagnostics = learner.update(
        tuple(unchecked), captures, behavior_model_id=policy_id,
        seed=9, allow_sampled_terminal_replay=True,
    )
    assert diagnostics["source_decisions"] == 2
    assert diagnostics["updates"] == 1
    # The episode provenance never changes: unchecked is still False.
    assert unchecked[1].terminal_replay_verified is False

    tampered = list(unchecked)
    tampered[0] = SimpleNamespace(**{
        **vars(unchecked[0]), "draw_mode": "learned",
    })
    with pytest.raises(ValueError, match="reference Chance Law"):
        learner.update(
            tuple(tampered), captures, behavior_model_id=policy_id,
            seed=9, allow_sampled_terminal_replay=True,
        )


def test_native_cpu_actor_audits_singleton_provenance_without_loosening_h8(tmp_path: Path):
    from types import MethodType

    torch.set_num_threads(1)
    net, spec, episodes, captures = make_cohort(tmp_path)
    learner = H7CombatPPO(net, temperature=0.85, epochs=1, minibatch=2)
    prepared = [
        learner._prepare_episode(
            episode, capture,
            behavior_model_id=f"h7-{spec.model_sha256[:16]}",
            temperature=0.85, gamma=1.0, gae_lambda=0.98,
        )[0]
        for episode, capture in zip(episodes, captures, strict=True)
    ]
    native = replace(prepared[0], actor_origin="native-cpu")
    with pytest.raises(ValueError, match="explicit cross-device"):
        learner.update_prepared([native, prepared[1]], seed=9)

    # A fake CPU behavior record must be rejected by the actual singleton
    # reference check, even if its error is within the GPU safety bound.
    tampered_native = replace(native, old_logp=native.old_logp + 0.0205)
    with pytest.raises(ValueError, match="Native CPU actor provenance mismatch"):
        learner.update_prepared(
            [tampered_native, prepared[1]], seed=9,
            allow_native_cpu_actor=True,
        )

    # Synthetic/reference-law decisions keep their strict 5e-4 check.
    wrong_synthetic = replace(
        prepared[1], old_logp=prepared[1].old_logp + 0.002,
    )
    with pytest.raises(ValueError, match="origin=h8-synthetic"):
        learner.update_prepared(
            [native, wrong_synthetic], seed=9, allow_native_cpu_actor=True,
        )

    # Simulate a 2-item batch numerical path that differs from the CPU
    # singleton actor, without altering the pinned model parameters.
    original = type(net).forward

    def batch_sensitive(self, batch, **kwargs):
        logits, values = original(self, batch, **kwargs)
        if batch.phases.shape[0] > 1:
            delta = torch.zeros_like(logits)
            delta[:, 0] = 0.08
            logits = logits + delta
        return logits, values

    net.forward = MethodType(batch_sensitive, net)
    both_native = [
        replace(prepared[0], actor_origin="native-cpu"),
        replace(prepared[1], actor_origin="native-cpu"),
    ]
    stats = learner.update_prepared(
        both_native, seed=9, allow_native_cpu_actor=True,
    )
    assert stats["native_actor_decisions"] == 2
    assert stats["native_cpu_audits"] >= 1
    assert stats["max_native_logp_drift"] > 0.001
    assert stats["max_native_cpu_logp_drift"] < 5e-4
    assert stats["source_decisions"] == 2
    assert both_native[0].old_logp == prepared[0].old_logp


def test_h7_native_cross_device_guard_rejects_large_ratio_divergence(tmp_path: Path):
    from types import MethodType

    torch.set_num_threads(1)
    net, spec, episodes, captures = make_cohort(tmp_path)
    learner = H7CombatPPO(net, temperature=0.85, epochs=1, minibatch=2)
    prepared = [
        replace(
            learner._prepare_episode(
                ep, cap, behavior_model_id=f"h7-{spec.model_sha256[:16]}",
                temperature=0.85, gamma=1.0, gae_lambda=0.98,
            )[0],
            actor_origin="native-cpu",
        )
        for ep, cap in zip(episodes, captures, strict=True)
    ]
    original = type(net).forward

    def grossly_wrong_batch(self, batch, **kwargs):
        logits, values = original(self, batch, **kwargs)
        logits = logits.clone()
        logits[:, 0] += 2.0
        return logits, values

    net.forward = MethodType(grossly_wrong_batch, net)
    with pytest.raises(ValueError, match="importance-ratio bound"):
        learner.update_prepared(
            prepared, seed=9, allow_native_cpu_actor=True,
        )
