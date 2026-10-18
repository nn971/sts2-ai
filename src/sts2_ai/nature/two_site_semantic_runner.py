"""Opt-in real semantic combat binding for the two-site Nature pilot.

The emulator is the *only* source of eligible card instances, enemy intents,
public observations, combat mechanics and terminal reward. This layer never
fabricates a hidden draw order, native intent probabilities or player state.
It is ready to use when the restricted playable semantic JSONL capability
tracked by sts2-emulator issue #143 is actually advertised.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sts2_ai.emulator.semantic_chance import (
    PlayableSemanticCombatBridge,
    SemanticNatureBridge,
)

from .semantic import SemanticEvent, SemanticProposal
from .semantic_trajectory import (
    SemanticTrajectory,
    collect_semantic_trajectory,
)
from .two_site_opponent import PILOT_OPERATIONS, PilotChoice, PilotEpisode, TwoSiteNature


class TwoSitePilotUnavailable(RuntimeError):
    """Mechanically legal trial cannot proceed with only intent and draw control."""


def describe_source_candidate_features(
    event: SemanticEvent,
) -> tuple[str, tuple[str, ...]]:
    """Use emulator-supplied owner/site context and intent/card model IDs."""
    if not event.candidate_features:
        raise TwoSitePilotUnavailable(
            "Semantic event has no source-backed candidate model labels"
        )
    return f"{event.site_id}|{event.owner_id}", event.candidate_features


@dataclass(frozen=True, slots=True)
class TwoSiteCombatResult:
    trajectory: SemanticTrajectory
    nature_episode: PilotEpisode
    player_return: float
    nature_decisions: int


class _NaturePolicyAdapter:
    def __init__(
        self, nature: TwoSiteNature, *,
        episode_key: str,
        describe: Callable[[SemanticEvent], tuple[str, tuple[str, ...]]],
    ):
        self.nature = nature
        self.episode_key = episode_key
        self.describe = describe
        self.policy_id = f"{nature.policy_id}:{nature.revision}"
        self.choices: list[PilotChoice] = []

    def propose(self, event: SemanticEvent) -> SemanticProposal:
        if event.operation not in PILOT_OPERATIONS:
            raise TwoSitePilotUnavailable(
                f"Unsupported two-site Nature event: {event.operation} at {event.site_id}"
            )
        context, features = self.describe(event)
        if not context or not isinstance(features, tuple):
            raise TwoSitePilotUnavailable("Missing event context or stable card-model metadata")
        proposal, choice = self.nature.propose(
            event, context_key=context, candidate_features=features,
            episode_key=f"{self.episode_key}:{len(self.choices)}",
        )
        # No chance state is pushed into the player policy.
        self.choices.append(choice)
        return proposal


def play_semantic_two_site(
    bridge: SemanticNatureBridge | PlayableSemanticCombatBridge,
    nature: TwoSiteNature, *,
    fixture_id: str, seed: str, episode_key: str,
    describe_event: Callable[[SemanticEvent], tuple[str, tuple[str, ...]]],
    player_choose: Callable[[dict[str, Any], tuple[str, ...]], str],
    score_terminal: Callable[[str], float],
    max_boundaries: int = 512,
) -> TwoSiteCombatResult:
    """Collect one genuinely terminal combat through the emulator's semantic wire.

    `describe_event` must supply source-backed model labels for each
    candidate physical card instance; opaque instance IDs alone are not
    suitable shared-model training features. `score_terminal` must use a
    validated real-game outcome label, never an emulator's bare terminal
    Boolean or a truncated/censored trace.
    """
    adapter = _NaturePolicyAdapter(
        nature, episode_key=episode_key, describe=describe_event,
    )
    trajectory = collect_semantic_trajectory(
        bridge, adapter, fixture_id=fixture_id, seed=seed,
        player_choose=player_choose, max_boundaries=max_boundaries,
    )
    reward = score_terminal(trajectory.terminal_outcome)
    if not isinstance(reward, (float, int)) or not math.isfinite(reward):
        raise ValueError("Invalid real semantic terminal combat return")
    return TwoSiteCombatResult(
        trajectory=trajectory,
        nature_episode=PilotEpisode(float(reward), tuple(adapter.choices)),
        player_return=float(reward),
        nature_decisions=len(adapter.choices),
    )
