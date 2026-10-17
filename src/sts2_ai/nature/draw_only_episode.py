"""Replay-checked real single-combat draw-only Nature episodes.

Everything after a selected outcome is executed by sts2-emulator's semantic
combat continuation. No hidden state, card order or synthetic combat actions
are created in Python. This is an engineering harness, not a training result.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from sts2_ai.emulator.semantic_p0_combat import strike_first_player
from sts2_ai.nature.draw_only_p0 import (
    DRAW_ONLY_SCHEMA,
    DrawOnlyContractError,
    DrawOnlyRouter,
    SeededLawSampler,
)
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.two_site_opponent import PilotChoice, PilotEpisode, TwoSiteNature


class DrawOnlySession(Protocol):
    capabilities: dict[str, Any]

    def begin_seeded_semantic_route(
        self, seed: str, *, region: str,
        native_policy_baseline: bool,
    ) -> Any: ...

    def load_semantic_snapshot(self, snapshot: str) -> Any: ...
    def release_semantic_many(self, states: list[Any]) -> int: ...


@dataclass(frozen=True)
class DrawOnlyStep:
    phase: str
    support_id: str
    selected_id: str
    result_fingerprint: str
    law_id: str | None = None
    conditioning_fingerprint: str | None = None
    log_p: float | None = None
    log_q: float | None = None


@dataclass(frozen=True)
class DrawOnlyEpisode:
    schema: str
    emulator_revision: str
    source_seed: str
    reference_seed: int
    player_model_id: str
    nature_revision: int
    source_opening_fingerprint: str
    first_public_fingerprint: str
    decisions: tuple[DrawOnlyStep, ...]
    nature_choices: tuple[PilotChoice, ...]
    reference_intent_count: int
    controlled_draw_count: int
    nontrivial_draw_count: int
    outcome: str
    exit_hp: int
    score: float
    terminal_fingerprint: str
    terminal_replay_verified: bool

    @property
    def digest(self) -> str:
        return canonical_sha256(asdict(self))

    def for_nature_training(self) -> PilotEpisode:
        return PilotEpisode(player_return=self.score, choices=self.nature_choices)


def check_draw_only_capabilities(capabilities: dict[str, Any]) -> None:
    manifest = capabilities.get("stochasticCapabilities") or {}
    if (capabilities.get("semanticChanceSchemaId") !=
            "sts2-v111-semantic-playable-p0-v1"
            or capabilities.get("semanticPlayable") is not True
            or capabilities.get("semanticIntentSupportSchemaId") !=
            "sts2-v111-semantic-weighted-intent-v1"
            or not {"enemy_intent", "draw_card"}.issubset(set(
                capabilities.get("semanticControlledSites", [])
            ))
            or manifest.get("chanceLawSchemaId") != "sts2-v111-chance-law-v1"
            or capabilities.get("referenceNatureSessionSchemaId") !=
            "sts2-v111-reference-nature-session-v1"
            or capabilities.get("semanticSeededRouteEntryProviderId") !=
            "engine-owned-v111-native-weak-route-entry-unverified-v1"):
        raise DrawOnlyContractError("Emulator lacks unified draw-only law capabilities")


def _score(outcome: str, entry_hp: int, max_hp: int, exit_hp: int) -> float:
    # Mirror the existing v15 HP-preservation reward: no extra death bonus.
    if (type(entry_hp) is not int or type(max_hp) is not int
            or type(exit_hp) is not int or max_hp <= 0
            or entry_hp < 0 or entry_hp > max_hp or exit_hp < 0):
        raise DrawOnlyContractError("Invalid combat HP for player scoring")
    if outcome == "defeat":
        return 0.0
    if outcome != "victory":
        raise DrawOnlyContractError("Unknown genuine terminal outcome")
    return 1.0 - max(0, entry_hp - exit_hp) / max_hp


def play_draw_only_episode(
    session: DrawOnlySession, *, source_seed: str, reference_seed: int,
    nature: TwoSiteNature,
    emulator_revision: str,
    player_choose: Callable[[dict[str, Any], list[dict[str, Any]]], str] =
        strike_first_player,
    player_model_id: str = "test-only-strike-first",
    region: str = "Overgrowth", max_boundaries: int = 600,
    replay: tuple[DrawOnlyStep, ...] | None = None,
) -> DrawOnlyEpisode:
    """Run actual supported first-room mechanics until a genuine terminal.

    A replay checks the source action menu, pending law IDs, selected outcomes,
    and every resulting state fingerprint; terminal snapshot round-trips.
    Reference RNG is a separate declared-law sampler, never the native cursor.
    """
    check_draw_only_capabilities(session.capabilities)
    if (region not in ("Overgrowth", "Underdocks") or not source_seed
            or not emulator_revision or not player_model_id
            or max_boundaries <= 0):
        raise ValueError("Invalid draw-only fixture configuration")
    router = DrawOnlyRouter(nature, SeededLawSampler(reference_seed))
    state = session.begin_seeded_semantic_route(
        source_seed, region=region, native_policy_baseline=False,
    )
    held = [state]
    decisions: list[DrawOnlyStep] = []
    choices: list[PilotChoice] = []
    opening = state.status().get("sourceOpeningFingerprint")
    first_public = None
    entry_hp = max_hp = None
    draw_count = nontrivial_count = 0
    try:
        if not isinstance(opening, str) or not opening:
            raise DrawOnlyContractError("Missing source-authentic opening")
        for ordinal in range(max_boundaries):
            if state.terminal:
                break
            if state.boundary == "nature":
                if state.pending is None:
                    raise DrawOnlyContractError("Nature boundary lacks an event")
                picked = router.choose(
                    state.pending, fingerprint=state.fingerprint,
                    episode_key=source_seed, ordinal=ordinal,
                )
                if picked.nature_choice is not None:
                    choices.append(picked.nature_choice)
                    draw_count += 1
                    nontrivial_count += len(state.pending["candidates"]) > 1
                if picked.operation == "enemy_intent":
                    child = state.resolve_intent(picked.selected_outcome_id)
                else:
                    instance = int(picked.selected_outcome_id.removeprefix("card_instance:"))
                    child = state.resolve_draw_card(instance)
                decision = DrawOnlyStep(
                    phase=f"nature:{picked.operation}",
                    support_id=picked.law_id,
                    selected_id=picked.selected_outcome_id,
                    result_fingerprint=child.fingerprint,
                    law_id=picked.law_id,
                    conditioning_fingerprint=picked.conditioning_fingerprint,
                    log_p=picked.log_p, log_q=picked.log_q,
                )
            elif state.boundary == "player":
                if state.pending is not None:
                    raise DrawOnlyContractError("Player boundary retained hidden chance")
                obs, actions = state.observe(), state.actions()
                if first_public is None:
                    first_public = canonical_sha256(obs)
                    entry_hp, max_hp = obs.get("hp"), obs.get("max_hp")
                menu_hash = canonical_sha256([obs, actions])
                if replay is not None:
                    if ordinal >= len(replay) or replay[ordinal].phase != "player":
                        raise DrawOnlyContractError("Player replay phase drift")
                    action_id = replay[ordinal].selected_id
                else:
                    action_id = player_choose(obs, actions)
                if action_id not in {a["actionId"] for a in actions}:
                    raise DrawOnlyContractError("Player action outside genuine legal menu")
                child = state.step(action_id)
                decision = DrawOnlyStep(
                    phase="player", support_id=menu_hash,
                    selected_id=action_id, result_fingerprint=child.fingerprint,
                )
            else:
                raise DrawOnlyContractError(
                    f"Unimplemented semantic boundary: {state.boundary}"
                )
            if replay is not None and (
                ordinal >= len(replay) or replay[ordinal] != decision
            ):
                raise DrawOnlyContractError("Draw-only replay transition mismatch")
            decisions.append(decision)
            held.append(child)
            state = child
        if not state.terminal:
            raise DrawOnlyContractError("CENSORED: no terminal combat before cap")
        if (state.outcome not in ("victory", "defeat") or
                type(state.exit_hp) is not int or state.boundary != "terminal"
                or state.pending is not None or entry_hp is None or max_hp is None
                or first_public is None):
            raise DrawOnlyContractError("Malformed genuine terminal combat")
        if replay is not None and len(replay) != len(decisions):
            raise DrawOnlyContractError("Replay ended at different boundary count")
        restored = session.load_semantic_snapshot(state.save_snapshot())
        held.append(restored)
        if restored.fingerprint != state.fingerprint:
            raise DrawOnlyContractError("Private terminal snapshot replay mismatch")
        return DrawOnlyEpisode(
            schema=DRAW_ONLY_SCHEMA, emulator_revision=emulator_revision,
            source_seed=source_seed, reference_seed=reference_seed,
            player_model_id=player_model_id, nature_revision=nature.revision,
            source_opening_fingerprint=opening,
            first_public_fingerprint=first_public,
            decisions=tuple(decisions), nature_choices=tuple(choices),
            reference_intent_count=router.reference.calls,
            controlled_draw_count=draw_count,
            nontrivial_draw_count=nontrivial_count,
            outcome=state.outcome, exit_hp=state.exit_hp,
            score=_score(state.outcome, entry_hp, max_hp, state.exit_hp),
            terminal_fingerprint=state.fingerprint,
            terminal_replay_verified=True,
        )
    finally:
        session.release_semantic_many(held)
