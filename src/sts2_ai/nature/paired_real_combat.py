"""Paired real-emulator Nature vs ordinary-RNG combat, with strict replay.

A pair is *two independent sessions* created with the same seed, region
and source-route opening. The emulator owns both chance implementations
and mechanics. Nature controls only legal enemy_intent / draw_card
boundaries; the ordinary arm owns native-v111 shuffle commitments.

This module measures a FROZEN player's responses. It does not train the
player, calibrate Steam's conditional chance laws, or infer that Nature
improves general gameplay from smoke tests.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from sts2_ai.emulator.semantic_p0_combat import (
    EMULATOR_SCHEMA,
    INTENT_SCHEMA,
    P0SemanticInterfaceError,
    _pending_event,
    strike_first_player,
)
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.two_site_opponent import PilotChoice, TwoSiteNature

PAIR_SCHEMA = "sts2-ai-paired-source-route-combat-v1"
NATIVE_PROVIDER = "engine-owned-native-v111-comparator-unverified-v1"
SOURCE_ROUTE_PROVIDER = "engine-owned-v111-native-weak-route-entry-unverified-v1"
EXPECTED_MODES = {
    "nature": "nature-two-site",
    "ordinary": "engine-owned-native-v111-policy-baseline",
}


class PairedCombatError(RuntimeError):
    """Unmatched source openings, unsupported mechanics or replay drift."""


class PairSession(Protocol):
    capabilities: dict[str, Any]

    def begin_seeded_semantic_route(
        self, seed: str, *, region: str,
        native_policy_baseline: bool,
    ) -> Any: ...
    def load_semantic_snapshot(self, snapshot: str) -> Any: ...
    def release_semantic_many(self, states: list[Any]) -> int: ...


@dataclass(frozen=True)
class PairedSourceFixture:
    fixture_id: str
    seed: str
    region: str = "Overgrowth"

    def __post_init__(self) -> None:
        if not self.fixture_id or not self.seed:
            raise ValueError("Paired combat requires source-group identity and seed")
        if self.region not in ("Overgrowth", "Underdocks"):
            raise ValueError("Only implemented source-routed Act 1 regions supported")


@dataclass(frozen=True)
class PairDecision:
    phase: str
    source_fingerprint: str
    support_identity: str
    selected_id: str
    resulting_fingerprint: str

    def __post_init__(self) -> None:
        if self.phase not in ("player", "nature:enemy_intent", "nature:draw_card"):
            raise ValueError("Unexpected paired-combat decision")
        if not all((self.source_fingerprint, self.support_identity,
                    self.selected_id, self.resulting_fingerprint)):
            raise ValueError("Incomplete paired-combat decision")


@dataclass(frozen=True)
class ArmResult:
    arm: str
    opening_fingerprint: str
    source_route_provider: str
    first_public_fingerprint: str
    decisions: tuple[PairDecision, ...]
    outcome: str
    exit_hp: int
    terminal_fingerprint: str
    native_decision_count: int
    native_shuffle_count: int
    nature_choices: tuple[PilotChoice, ...]
    chance_probability_status: str = "UNVERIFIED"

    @property
    def score(self) -> float:
        return float(self.outcome == "victory") + max(0, self.exit_hp) / 100.0

    @property
    def digest(self) -> str:
        return canonical_sha256({
            "arm": self.arm, "opening": self.opening_fingerprint,
            "route": self.source_route_provider,
            "first_public": self.first_public_fingerprint,
            "decisions": [asdict(x) for x in self.decisions],
            "outcome": self.outcome, "exit_hp": self.exit_hp,
            "terminal": self.terminal_fingerprint,
            "native_decisions": self.native_decision_count,
            "native_shuffles": self.native_shuffle_count,
        })


def without_committed_intents(observation: dict[str, Any]) -> dict[str, Any]:
    """Remove ONLY intent-specific publicly committed fields for the opening check."""
    copy = json.loads(json.dumps(observation))
    for enemy in copy.get("combat", {}).get("enemies", []):
        for key in list(enemy):
            if key.replace("_", "").lower() in {
                "moveid", "intentdamage", "intenthits", "intentbasedamage",
            }:
                enemy.pop(key)
    return copy


def check_comparator_capabilities(session: PairSession) -> None:
    h = session.capabilities
    required = (
        h.get("wireSchemaId") == "prototype-ai-jsonl-v0"
        and h.get("semanticChanceSchemaId") == EMULATOR_SCHEMA
        and h.get("semanticPlayable") is True
        and set(h.get("semanticControlledSites", []))
        == {"enemy_intent", "draw_card"}
        and h.get("semanticIntentSupportSchemaId") == INTENT_SCHEMA
        and h.get("semanticNativePolicyComparatorId") == NATIVE_PROVIDER
        and h.get("semanticSeededRouteEntryProviderId") == SOURCE_ROUTE_PROVIDER
        and h.get("fairPolicyId") == "prototype-fair-v0"
        and h.get("semanticProbabilityStatus") == "UNVERIFIED"
    )
    if not required:
        raise PairedCombatError("Pinned emulator lacks paired two-site comparator")


def _run_arm(
    session: PairSession, fixture: PairedSourceFixture, *,
    arm: str, nature: TwoSiteNature | None,
    choose: Callable[[dict[str, Any], list[dict[str, Any]]], str],
    max_boundaries: int,
    replay: tuple[PairDecision, ...] | None = None,
) -> ArmResult:
    if arm not in EXPECTED_MODES:
        raise ValueError("Invalid paired source arm")
    native = arm == "ordinary"
    if native == (nature is not None) and replay is None:
        raise ValueError("Nature policy required only for the Nature arm")
    state = session.begin_seeded_semantic_route(
        fixture.seed, region=fixture.region, native_policy_baseline=native,
    )
    held = [state]
    selections: list[PairDecision] = []
    nature_choices: list[PilotChoice] = []
    first_public = None
    try:
        initial_status = state.status()
        opening = initial_status.get("sourceOpeningFingerprint")
        if not isinstance(opening, str) or not opening:
            raise PairedCombatError("Missing common pre-controller source opening")
        if initial_status.get("sourceRouteProvider") != SOURCE_ROUTE_PROVIDER:
            raise PairedCombatError("Source-routed opening not verified")
        if initial_status.get("sourceRouteProbabilityStatus") != "UNVERIFIED":
            raise PairedCombatError("Source-route probability provenance mismatch")
        if initial_status.get("controllerMode") != EXPECTED_MODES[arm]:
            raise PairedCombatError("Unexpected emulator chance controller mode")
        if native:
            if initial_status.get("nativePolicyProvider") != NATIVE_PROVIDER:
                raise PairedCombatError("Ordinary RNG provider is unverified/unsupported")
            if initial_status.get("chanceSites") != []:
                raise PairedCombatError("Ordinary RNG must own all chance choices")
        elif set(initial_status.get("chanceSites", [])) != {
            "enemy_intent", "draw_card",
        }:
            raise PairedCombatError("Nature controls sites beyond intent and draw")

        for ordinal in range(max_boundaries):
            if state.terminal:
                break
            prior = state.fingerprint
            if state.boundary == "nature" and not native:
                if state.pending is None:
                    raise PairedCombatError("Missing Nature pending chance event")
                event, features, identities = _pending_event(state.pending, prior)
                phase = f"nature:{event.operation}"
                support = event.replay_identity
                if replay is not None:
                    if ordinal >= len(replay):
                        raise PairedCombatError("Nature replay exhausted")
                    expected = replay[ordinal]
                    if (expected.phase != phase or
                            expected.support_identity != support or
                            expected.selected_id not in identities):
                        raise PairedCombatError("Nature replay support divergence")
                    selected_id = expected.selected_id
                else:
                    if nature is None:
                        raise PairedCombatError("Missing Nature proposal policy")
                    proposal, choice = nature.propose(
                        event,
                        context_key=f"{event.site_id}|{event.owner_id}",
                        candidate_features=features,
                        episode_key=f"{fixture.fixture_id}:{ordinal}",
                    )
                    selected_id = proposal.selected_id
                    nature_choices.append(choice)
                if event.operation == "enemy_intent":
                    child = state.resolve_intent(selected_id)
                else:
                    child = state.resolve_draw_card(int(selected_id))
            elif state.boundary == "player":
                if state.pending is not None:
                    raise PairedCombatError("Pending hidden chance at player boundary")
                observation, actions = state.observe(), state.actions()
                if first_public is None:
                    first_public = canonical_sha256(
                        without_committed_intents(observation)
                    )
                phase = "player"
                support = canonical_sha256([observation, actions])
                if replay is not None:
                    if ordinal >= len(replay):
                        raise PairedCombatError("Player replay exhausted")
                    expected = replay[ordinal]
                    if (expected.phase != phase or
                            expected.support_identity != support):
                        raise PairedCombatError("Player public information/menu changed")
                    selected_id = expected.selected_id
                else:
                    selected_id = choose(observation, actions)
                if selected_id not in {a["actionId"] for a in actions}:
                    raise PairedCombatError("Player chose an action outside source menu")
                child = state.step(selected_id)
            else:
                raise PairedCombatError(f"Unsupported source combat phase: {state.boundary}")
            decision = PairDecision(phase, prior, support, selected_id, child.fingerprint)
            if replay is not None and decision != replay[ordinal]:
                raise PairedCombatError("Replayed transition fingerprint diverged")
            selections.append(decision)
            state = child
            held.append(child)
        if not state.terminal:
            raise PairedCombatError("CENSORED: combat boundary cap exhausted")
        if (state.boundary != "terminal" or state.pending is not None
                or state.outcome not in ("victory", "defeat")
                or type(state.exit_hp) is not int):
            raise PairedCombatError("Malformed genuine combat terminal outcome")
        if first_public is None:
            raise PairedCombatError("No fair player boundary reached")
        if replay is not None and len(replay) != len(selections):
            raise PairedCombatError("Replayed combat terminated early")
        final = state.status()
        if final.get("probabilityStatus") != "UNVERIFIED":
            raise PairedCombatError("Chance-law status unexpectedly changed")
        nd = final.get("nativePolicyDecisionCount")
        ns = final.get("nativePolicyShuffleCount")
        if type(nd) is not int or type(ns) is not int:
            raise PairedCombatError("Missing native audit counts")
        if native and (nd <= 0 or ns <= 0):
            raise PairedCombatError("Ordinary comparator never committed native shuffle")
        if not native and (nd != 0 or ns != 0):
            raise PairedCombatError("Nature emitted native comparator decisions")
        restored = session.load_semantic_snapshot(state.save_snapshot())
        held.append(restored)
        if restored.fingerprint != state.fingerprint:
            raise PairedCombatError("Terminal private snapshot replay diverged")
        return ArmResult(
            arm=arm, opening_fingerprint=opening,
            source_route_provider=initial_status["sourceRouteProvider"],
            first_public_fingerprint=first_public,
            decisions=tuple(selections), outcome=state.outcome,
            exit_hp=state.exit_hp, terminal_fingerprint=state.fingerprint,
            native_decision_count=nd, native_shuffle_count=ns,
            nature_choices=tuple(nature_choices),
        )
    finally:
        session.release_semantic_many(held)


def play_paired_source_combat(
    session: PairSession, fixture: PairedSourceFixture, *,
    nature: TwoSiteNature,
    choose_factory: Callable[[str], Callable[[dict[str, Any], list[dict[str, Any]]], str]]
    | None = None,
    max_boundaries: int = 600,
    verify_replay: bool = True,
) -> tuple[ArmResult, ArmResult]:
    """Fresh identically initialized arms, with independent mechanical replay."""
    check_comparator_capabilities(session)
    if max_boundaries <= 0:
        raise ValueError("Combat boundary cap must be positive")
    factory = choose_factory or (lambda _arm: strike_first_player)
    nature_result = _run_arm(
        session, fixture, arm="nature", nature=nature,
        choose=factory("nature"), max_boundaries=max_boundaries,
    )
    ordinary_result = _run_arm(
        session, fixture, arm="ordinary", nature=None,
        choose=factory("ordinary"), max_boundaries=max_boundaries,
    )
    if nature_result.opening_fingerprint != ordinary_result.opening_fingerprint:
        raise PairedCombatError("Nature/ordinary run source openings differ")
    if nature_result.first_public_fingerprint != ordinary_result.first_public_fingerprint:
        raise PairedCombatError("Nature/ordinary initial fair public states differ")
    if verify_replay:
        for result in (nature_result, ordinary_result):
            repeated = _run_arm(
                session, fixture, arm=result.arm, nature=None,
                choose=factory(result.arm),
                max_boundaries=max_boundaries, replay=result.decisions,
            )
            if repeated.digest != result.digest:
                raise PairedCombatError(
                    f"{result.arm} replay digest/terminal provenance mismatch"
                )
    return nature_result, ordinary_result


def paired_record(
    fixture: PairedSourceFixture, nature: ArmResult, ordinary: ArmResult,
) -> dict[str, Any]:
    if nature.arm != "nature" or ordinary.arm != "ordinary":
        raise ValueError("Incorrect pair assignment")
    return {
        "schema": PAIR_SCHEMA,
        "fixture": asdict(fixture),
        "source_opening_fingerprint": nature.opening_fingerprint,
        "same_opening": nature.opening_fingerprint == ordinary.opening_fingerprint,
        "same_initial_public_state": (
            nature.first_public_fingerprint == ordinary.first_public_fingerprint
        ),
        "nature": {
            **asdict(nature), "trace_digest": nature.digest,
            "player_return": nature.score,
        },
        "ordinary": {
            **asdict(ordinary), "trace_digest": ordinary.digest,
            "player_return": ordinary.score,
        },
        "nature_minus_ordinary_return": nature.score - ordinary.score,
        "probability_status": "UNVERIFIED",
        "installed_game_fidelity": "NOT_TESTED",
        "evidence_level": "integration-smoke",
    }
