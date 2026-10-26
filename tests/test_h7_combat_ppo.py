"""H7 source-fair PPO provenance, finite updates and behavior-policy matching."""
from __future__ import annotations

from copy import deepcopy
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
