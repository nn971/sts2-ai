"""H7 throughput reports must preserve complete source and context provenance."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sts2_ai.evaluation.h7_readiness import REPORT_SCHEMA, make_readiness_report
from sts2_ai.nature.h5_curriculum import source_bank


def _cohort(seeds: tuple[str, ...]):
    from sts2_ai.nature.h5_curriculum import cell_for_source

    return {
        "episodes": tuple(
            SimpleNamespace(
                source_seed=seed,
                scenario_id=cell_for_source(seed).scenario,
                deck_preset_id=cell_for_source(seed).deck,
                draw_mode="reference",
                outcome="victory" if i % 2 else "defeat",
                terminal_replay_verified=True,
                decisions=(
                    SimpleNamespace(phase="player"),
                    SimpleNamespace(phase="draw_card"),
                    SimpleNamespace(phase="player_choice"),
                ),
            )
            for i, seed in enumerate(seeds)
        ),
        "elapsed_seconds": 8.0,
        "process_count": 8,
    }


def test_report_has_full_curriculum_and_measured_decision_throughput():
    seeds = source_bank("train", 16, study_seed=49)
    report = make_readiness_report(
        _cohort(seeds), seeds, workers=15,
        model_sha256="a" * 64, parameter_count=14_659_972,
    )
    assert report["schema"] == REPORT_SCHEMA
    assert report["train_cells"] == 16
    assert all(count == 1 for count in report["by_cell"].values())
    assert report["combats"] == 16
    assert report["player_decisions"] == 32
    assert report["player_decisions_per_second"] == 4
    assert report["actor_inference_device"] == "cpu"
    assert report["worker_processes_observed"] == 8


def test_report_rejects_source_loss_tampering_or_incomplete_curriculum():
    seeds = source_bank("train", 16, study_seed=49)
    cohort = _cohort(seeds)
    with pytest.raises(ValueError, match="unverified"):
        make_readiness_report(
            {**cohort, "episodes": cohort["episodes"][::-1]}, seeds,
            workers=15, model_sha256="a" * 64, parameter_count=14_659_972,
        )
    with pytest.raises(ValueError, match="source-complete"):
        make_readiness_report(
            cohort, seeds[:-1], workers=15,
            model_sha256="a" * 64, parameter_count=14_659_972,
        )
    with pytest.raises(ValueError, match="source-complete"):
        make_readiness_report(
            cohort, seeds, workers=4,
            model_sha256="a" * 64, parameter_count=14_659_972,
        )
    damaged = list(cohort["episodes"])
    damaged[0] = SimpleNamespace(**{
        **vars(damaged[0]), "terminal_replay_verified": False,
    })
    with pytest.raises(ValueError, match="unverified"):
        make_readiness_report(
            {**cohort, "episodes": tuple(damaged)}, seeds,
            workers=15, model_sha256="a" * 64, parameter_count=14_659_972,
        )


def test_h7_adamw_resume_matches_uninterrupted_next_round(tmp_path):
    """Small-model continuation: saved optimizer and RNG exactly match live PPO."""
    torch = pytest.importorskip("torch")
    from test_h7_combat_ppo import make_cohort
    from test_shared_transformer_h7 import frame, legal, model
    from sts2_ai.emulator.semantic_p0_frozen_player import FrozenP0Player
    from sts2_ai.models.h7_public_policy import H7PlayerSpec
    from sts2_ai.training.h7_combat_ppo import H7CombatPPO

    torch.set_num_threads(1)
    network, first_spec, episodes, captures = make_cohort(tmp_path)
    original = H7CombatPPO(network, temperature=0.85, epochs=1, minibatch=2)
    original.update(
        episodes, captures,
        behavior_model_id=f"h7-{first_spec.model_sha256[:16]}", seed=101,
    )
    checkpoint_path = tmp_path / "h7-optimizer.pt"
    torch.save({
        "model": original.model.state_dict(),
        "optimizer": original.optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
    }, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    resumed = H7CombatPPO(model(), temperature=0.85, epochs=1, minibatch=2)
    resumed.model.load_state_dict(checkpoint["model"])
    resumed.optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng"])

    next_model = tmp_path / "h7-round2.pt"
    original.model.save_checkpoint(next_model)
    spec = H7PlayerSpec.from_file(next_model, seed=23, temperature=0.85)
    choices = [
        {"actionId": str(i), "kind": action.kind, "payloadJson": action.payload_json}
        for i, action in enumerate(legal())
    ]
    next_episodes = []
    next_captures = []
    for seed in (31, 79):
        actor = FrozenP0Player(
            spec.load_pinned(), actor_seed=seed, temperature=0.85,
            greedy=False, action_scope="all-legal-v1", capture_training=True,
        )
        selected = actor.choose(frame(), choices)
        next_episodes.append(SimpleNamespace(
            outcome="victory" if seed == 31 else "defeat",
            score=1.65 if seed == 31 else 0.45,
            terminal_replay_verified=True, draw_mode="reference",
            decisions=(SimpleNamespace(phase="player", selected_id=selected),),
        ))
        next_captures.append(tuple(actor.training_actions))
    source_id = f"h7-{spec.model_sha256[:16]}"
    args = (tuple(next_episodes), tuple(next_captures))
    original.update(*args, behavior_model_id=source_id, seed=202)
    resumed.update(*args, behavior_model_id=source_id, seed=202)

    for key, reference in original.model.state_dict().items():
        torch.testing.assert_close(
            reference, resumed.model.state_dict()[key], atol=0, rtol=0,
        )
    saved_a = original.optimizer.state_dict()
    saved_b = resumed.optimizer.state_dict()
    assert saved_a["param_groups"] == saved_b["param_groups"]
    for param_id, state in saved_a["state"].items():
        for name, tensor in state.items():
            torch.testing.assert_close(
                tensor, saved_b["state"][param_id][name], atol=0, rtol=0,
            )
