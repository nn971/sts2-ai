"""Trainable Nature controlling only card draws and committed enemy intents."""
from __future__ import annotations

import math

import pytest

from sts2_ai.emulator.semantic_chance import SemanticBoundary
from sts2_ai.nature.semantic import SemanticEvent, SemanticProposal
from sts2_ai.nature.two_site_opponent import (
    PILOT_OPERATIONS,
    PilotEpisode,
    TwoSiteNature,
)
from sts2_ai.nature.two_site_semantic_runner import (
    TwoSitePilotUnavailable,
    play_semantic_two_site,
)
from sts2_ai.training.two_site_combat_toy import (
    ToyPlayer,
    run_paired_toy_pilot,
    run_toy_combat,
    semantic_toy_event,
)


def test_two_site_nature_eligibility_and_unverified_laws() -> None:
    opponent = TwoSiteNature(seed=29)
    intent = semantic_toy_event(
        "enemy_intent", ("weak", "strong", "forbidden"), episode=0, turn=0,
        intent_weights=(2, 1, 0),
    )
    proposal, choice = opponent.propose(
        intent, context_key="turn:1",
        candidate_features=("weak", "strong", "forbidden"),
        episode_key="episode-0",
    )
    assert proposal.selected_id != "forbidden"
    assert choice.probabilities[2] == 0
    assert sum(choice.probabilities) == pytest.approx(1.0)
    assert intent.probability_status == "UNVERIFIED"
    assert opponent.dump()["calibrated_native_reference_law"] is False
    wrong = SemanticEvent(
        event_id="revealed", boundary_fingerprint="fp", site_id="toy.reveal",
        owner_id="deck", ordinal=0, operation="reveal_card",
        candidate_ids=("x", "y"), support_fingerprint="sp",
        randomization_id="e", constraint_fingerprint="cf",
    )
    with pytest.raises(NotImplementedError, match="Outside intent/draw"):
        opponent.propose(
            wrong, context_key="turn:1", candidate_features=("x", "y"),
            episode_key="0",
        )


def test_adversarial_reinforce_shifts_q_toward_lower_player_return() -> None:
    opponent = TwoSiteNature(seed=41, learning_rate=0.24, mix=0.9)
    initial, _ = opponent._distribution(
        "draw_card", "turn:0", ("good", "bad"), (0.5, 0.5),
    )
    assert initial == pytest.approx((0.5, 0.5))
    for round_id in range(35):
        episodes = []
        for j in range(80):
            event = semantic_toy_event(
                "draw_card", ("good#1", "bad#2"),
                episode=round_id * 80 + j, turn=0,
            )
            proposal, choice = opponent.propose(
                event, context_key="turn:0", candidate_features=("good", "bad"),
                episode_key=str(round_id * 80 + j),
            )
            # In the declared toy player return, drawing 'bad' hurts.
            episodes.append(PilotEpisode(
                1.0 if proposal.selected_index == 0 else -1.0,
                (choice,),
            ))
        stats = opponent.update(tuple(episodes))
        assert math.isfinite(stats["nature_gradient_norm"])
    learned, _ = opponent._distribution(
        "draw_card", "turn:0", ("good", "bad"), (0.5, 0.5),
    )
    assert learned[1] > 0.75
    assert learned[0] >= (1 - opponent.mix) * 0.5 - 1e-12
    assert opponent.revision == 35
    restored = TwoSiteNature.restore(opponent.dump())
    assert restored.dump() == opponent.dump()


def test_invalid_stale_or_censored_training_rows_rejected() -> None:
    opponent = TwoSiteNature(seed=9)
    event = semantic_toy_event("draw_card", ("a", "b"), episode=1, turn=0)
    _, choice = opponent.propose(
        event, context_key="first", candidate_features=("a", "b"),
        episode_key="1",
    )
    with pytest.raises(ValueError, match="Only finite terminal"):
        PilotEpisode(1.0, (choice,), terminal=False)
    with pytest.raises(ValueError, match="at least two"):
        opponent.update((PilotEpisode(0.0, (choice,)),))
    opponent.update((PilotEpisode(0.0, (choice,)), PilotEpisode(1.0, (choice,))))
    with pytest.raises(ValueError, match="Stale Nature"):
        opponent.update((PilotEpisode(0.0, (choice,)), PilotEpisode(1.0, (choice,))))


def test_paired_native_vs_nature_training_reproducible_and_budgeted() -> None:
    params = dict(
        rounds=5, episodes_per_batch=12, seed=29, evaluation_episodes=96,
    )
    first = run_paired_toy_pilot(**params)
    assert first == run_paired_toy_pilot(**params)
    assert first["synthetic_not_sts2"] is True
    assert first["nature_eligibility"] == sorted(PILOT_OPERATIONS)
    assert first["progress"][-1]["simulated_combats_per_arm"] == 120
    assert first["final_nature_checkpoint"]["revision"] == 5
    for result in first["progress"]:
        assert 0 <= result["native"]["win_rate"] <= 1
        assert 0 <= result["nature_train"]["win_rate"] <= 1


def test_synthetic_combat_does_not_forge_sts2_gameplay() -> None:
    player = ToyPlayer(seed=17)
    native = run_toy_combat(player, episode=9, native_seed=19)
    controlled = run_toy_combat(player, episode=9, nature=TwoSiteNature(seed=29))
    assert native.nature.choices == ()
    assert all(x.operation in PILOT_OPERATIONS for x in controlled.nature.choices)
    assert math.isfinite(native.reward) and math.isfinite(controlled.reward)


class MinimalSemanticCombat:
    """Synthetic 2 chance / 1 player / terminal adapter, not actual C# combat."""

    def __init__(self, *, unsupported: bool = False):
        self.unsupported = unsupported
        self.released = 0
        self.state = 0
        self.choices: list[str] = []

    def _boundary(self, stage):
        if stage in (0, 1):
            operation = ("enemy_intent" if stage == 0 else
                         "reveal_card" if self.unsupported else "draw_card")
            event = semantic_toy_event(
                operation if operation != "reveal_card" else "draw_card",
                ("weak", "strong") if stage == 0 else ("Strike#1", "Defend#2"),
                episode=0, turn=stage,
                intent_weights=(2, 1) if stage == 0 else None,
            )
            if self.unsupported and stage == 1:
                from dataclasses import replace
                event = replace(event, operation="reveal_card")
            return SemanticBoundary(
                handle=f"s{stage}", phase="chance", fingerprint=f"p{stage}",
                event=event, observation=None, legal_actions=(), terminal=None,
            )
        if stage == 2:
            return SemanticBoundary(
                handle="s2", phase="player", fingerprint="p2",
                event=None, observation={"hp": 12, "intent": self.choices[0]},
                legal_actions=("end_turn",), terminal=None,
            )
        return SemanticBoundary(
            handle="s3", phase="terminal", fingerprint="p3",
            event=None, observation=None, legal_actions=(),
            terminal="toy-validated-win",
        )

    def begin(self, *, fixture_id: str, seed: str):
        self.state = 0
        return self._boundary(self.state)

    def resolve(self, frame, proposal: SemanticProposal):
        assert frame.phase == "chance"
        proposal.validate(frame.event)
        self.choices.append(proposal.selected_id)
        self.state += 1
        return self._boundary(self.state)

    def step(self, frame, action_id):
        assert frame.phase == "player" and action_id == "end_turn"
        self.state = 3
        return self._boundary(self.state)

    def release(self, *frames):
        self.released += len(frames)
        return len(frames)


def test_real_adapter_shape_uses_only_two_sites_and_verified_terminal() -> None:
    bridge = MinimalSemanticCombat()
    result = play_semantic_two_site(
        bridge, TwoSiteNature(seed=73), fixture_id="toy", seed="toy-1",
        episode_key="1",
        describe_event=lambda event: (
            f"turn:{event.ordinal}",
            ("weak", "strong") if event.operation == "enemy_intent"
            else ("Strike", "Defend"),
        ),
        player_choose=lambda observation, actions: actions[0],
        score_terminal=lambda outcome: (
            1.0 if outcome == "toy-validated-win" else float("nan")
        ),
    )
    assert result.nature_decisions == 2
    assert len(result.trajectory.player_script()) == 1
    assert bridge.released == 4
    assert len(result.nature_episode.choices) == 2

    blocked = MinimalSemanticCombat(unsupported=True)
    with pytest.raises(TwoSitePilotUnavailable, match="reveal_card"):
        play_semantic_two_site(
            blocked, TwoSiteNature(seed=73), fixture_id="toy", seed="toy-1",
            episode_key="1", describe_event=lambda event: ("turn:0", ("a", "b")),
            player_choose=lambda obs, menu: menu[0],
            score_terminal=lambda outcome: 1.0,
        )
    assert blocked.released > 0
