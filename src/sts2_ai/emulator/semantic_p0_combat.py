"""Real P0 semantic combat adapter for sts2-emulator's versioned Python binding.

The emulator, not this adapter, owns mechanics, hidden deck state, weighted
intent eligibility, and all non-Nature randomness. The only Nature choices
here are enemy_intent and unresolved physical draw_card instances.

Requires the external sts2-emulator v111 RNG/Inspector integration build.
The gitlink in sts2-ai is deliberately not modified.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.semantic import SemanticEvent
from sts2_ai.nature.two_site_opponent import PilotChoice, PilotEpisode, TwoSiteNature

EMULATOR_SCHEMA = "sts2-v111-semantic-playable-p0-v1"
INTENT_SCHEMA = "sts2-v111-semantic-weighted-intent-v1"
OPENING_PROVIDER = "engine-owned-seeded-source-shaped-unverified-v1"
OTHER_CHANCE_NATIVE_CODEC = "engine-owned-native-v111-cost-stream-unverified-v1"
TRACE_SCHEMA = "sts2-ai-nature-real-two-site-combat-trace-v1"


class P0SemanticInterfaceError(RuntimeError):
    """Fail closed on an unsupported or incompatible source interface."""


class SemanticCombatState(Protocol):
    handle: str
    fingerprint: str
    boundary: str
    pending: dict[str, Any] | None
    terminal: bool
    outcome: str | None
    exit_hp: int | None

    def observe(self) -> dict[str, Any]: ...
    def actions(self) -> list[dict[str, Any]]: ...
    def step(self, action_id: str) -> SemanticCombatState: ...
    def resolve(self, selected_index: int) -> SemanticCombatState: ...
    def resolve_intent(self, intent_id: str) -> SemanticCombatState: ...
    def save_snapshot(self) -> str: ...


class SemanticSession(Protocol):
    capabilities: dict[str, Any]

    def begin_seeded_semantic_combat(self, seed: str, **kwargs: Any) -> SemanticCombatState: ...
    def load_semantic_snapshot(self, snapshot: str) -> SemanticCombatState: ...
    def release_semantic_many(self, states: list[SemanticCombatState]) -> int: ...


@dataclass(frozen=True, slots=True)
class CombatFixture:
    fixture_id: str
    seed: str
    act: int
    encounter_id: str
    native_entry: str
    total_floor: int
    ascension: int = 0
    map_column: int = 3
    map_row: int = 1
    replay_opening_draw_hooks: bool = True
    other_chance_provider_id: str = OTHER_CHANCE_NATIVE_CODEC

    def __post_init__(self) -> None:
        if not all((self.fixture_id, self.seed, self.encounter_id, self.native_entry)):
            raise ValueError("Missing reproducible semantic combat identity")
        if self.act not in (1, 2, 3) or self.total_floor <= 0:
            raise ValueError("Invalid semantic combat fixture act/floor")


@dataclass(frozen=True, slots=True)
class CombatDecision:
    phase: str
    source_fingerprint: str
    support_identity: str
    selected_id: str
    resulting_fingerprint: str
    log_q: float | None = None

    def __post_init__(self) -> None:
        if self.phase not in ("nature:enemy_intent", "nature:draw_card", "player"):
            raise ValueError("Unrecognized P0 combat decision phase")
        if not all((self.source_fingerprint, self.support_identity,
                    self.selected_id, self.resulting_fingerprint)):
            raise ValueError("Incomplete source-backed combat decision")


@dataclass(frozen=True, slots=True)
class CombatTrace:
    fixture: CombatFixture
    emulator_revision: str
    policy_id: str
    decisions: tuple[CombatDecision, ...]
    outcome: str
    exit_hp: int
    terminal_fingerprint: str
    other_chance_provider: str
    probability_status: str = "UNVERIFIED"
    schema: str = TRACE_SCHEMA

    @property
    def digest(self) -> str:
        # Policy log_q is excluded to compare legal mechanical replays.
        return canonical_sha256([
            asdict(self.fixture), self.emulator_revision,
            [(x.phase, x.source_fingerprint, x.support_identity,
              x.selected_id, x.resulting_fingerprint) for x in self.decisions],
            self.outcome, self.exit_hp, self.terminal_fingerprint,
            self.other_chance_provider,
        ])

    def replay_choices(self) -> tuple[tuple[str, str, str], ...]:
        return tuple((x.phase, x.support_identity, x.selected_id)
                     for x in self.decisions)


@dataclass(frozen=True, slots=True)
class RealCombatEpisode:
    trace: CombatTrace
    player_return: float
    nature_episode: PilotEpisode

    def __post_init__(self) -> None:
        if not math.isfinite(self.player_return):
            raise ValueError("Nonfinite real combat reward")


def check_capabilities(session: SemanticSession, fixture: CombatFixture) -> None:
    h = session.capabilities
    if (h.get("wireSchemaId") != "prototype-ai-jsonl-v0"
            or h.get("semanticChanceSchemaId") != EMULATOR_SCHEMA
            or h.get("semanticPlayable") is not True
            or set(h.get("semanticControlledSites", []))
            != {"enemy_intent", "draw_card"}
            or h.get("semanticIntentSupportSchemaId") != INTENT_SCHEMA
            or h.get("semanticSeededOpeningProvider") != OPENING_PROVIDER
            or h.get("fairPolicyId") != "prototype-fair-v0"
            or h.get("semanticProbabilityStatus") != "UNVERIFIED"):
        raise P0SemanticInterfaceError("Unsupported P0 semantic Nature handshake")
    if fixture.other_chance_provider_id not in h.get("semanticOtherChanceProviders", []):
        raise P0SemanticInterfaceError("Requested engine-owned RNG provider unavailable")
    if (fixture.replay_opening_draw_hooks and
            h.get("semanticSeededOpeningDrawHooksId") !=
            "engine-owned-seeded-source-shaped-draw-hooks-unverified-v1"):
        raise P0SemanticInterfaceError("Original opening CardDrawn continuation unavailable")


def _pending_event(
    raw: Mapping[str, Any], fingerprint: str,
) -> tuple[SemanticEvent, tuple[str, ...], tuple[str, ...]]:
    if raw.get("probabilityStatus") != "UNVERIFIED":
        raise P0SemanticInterfaceError("Unexpected source probability-law status")
    kind = raw.get("kind")
    if kind == "enemy_intent":
        if (raw.get("semanticIntentSupportSchemaId") != INTENT_SCHEMA
                or raw.get("weightsVerified") is not False):
            raise P0SemanticInterfaceError("Missing post-filter intent support provenance")
        candidates = raw.get("semanticCandidates")
        if not isinstance(candidates, list) or not candidates:
            raise P0SemanticInterfaceError("No eligible semantic intents")
        ids = tuple(x["intentId"] for x in candidates)
        moves = tuple(x["moveId"] for x in candidates)
        weights = tuple(x["rawWeight"] for x in candidates)
        total = raw.get("totalRawWeight")
        if (not all(isinstance(x, str) and x for x in ids + moves)
                or not isinstance(total, int)
                or total != sum(weights)
                or not raw.get("semanticSupportFingerprint")):
            raise P0SemanticInterfaceError("Malformed eligible weighted intent support")
        epoch = f"intent:{raw['ownerId']}:{raw['ordinal']}"
        support = raw["semanticSupportFingerprint"]
        event_id = canonical_sha256([fingerprint, raw["siteId"],
                                     raw["ownerId"], raw["ordinal"], support])
        event = SemanticEvent(
            event_id=event_id, boundary_fingerprint=fingerprint,
            site_id=raw["siteId"], owner_id=raw["ownerId"],
            ordinal=raw["ordinal"], operation="enemy_intent",
            candidate_ids=ids, support_fingerprint=support,
            randomization_id=epoch, constraint_fingerprint=raw["supportFingerprint"],
            raw_weights=weights, weight_total=total,
        )
        return event, moves, ids
    if kind == "draw_card":
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise P0SemanticInterfaceError("No eligible physical card instances")
        ids = tuple(str(x["cardInstanceId"]) for x in candidates)
        models = tuple(x["cardModelId"] for x in candidates)
        if (not all(isinstance(x, str) and x for x in models)
                or not all(x["index"] == i for i, x in enumerate(candidates))
                or not raw.get("constraintFingerprint")):
            raise P0SemanticInterfaceError("Malformed physical-card draw support")
        event = SemanticEvent(
            event_id=raw["eventId"], boundary_fingerprint=fingerprint,
            site_id=raw["siteId"], owner_id=raw["ownerId"],
            ordinal=raw["ordinal"], operation="draw_card",
            candidate_ids=ids, support_fingerprint=raw["supportFingerprint"],
            randomization_id=f"deck:{raw['shuffleEpoch']}",
            constraint_fingerprint=raw["constraintFingerprint"],
        )
        return event, models, ids
    raise P0SemanticInterfaceError(f"Outside the two-site pilot: {kind}")


def first_legal_player(
    _observation: dict[str, Any], actions: list[dict[str, Any]],
) -> str:
    if not actions:
        raise P0SemanticInterfaceError("No legal public player actions")
    return actions[0]["actionId"]


def strike_first_player(
    _observation: dict[str, Any], actions: list[dict[str, Any]],
) -> str:
    """Structural test policy: prioritize an available Strike, then end turn.

    This is deliberately *not* the frozen v15 policy or a performance
    baseline. It operates only on a fair, source-issued action menu.
    """
    hand = {
        card["instance_id"]: card["card_id"]
        for card in _observation.get("combat", {}).get("hand", [])
    }
    for preferred in ("proto.silent.strike", "proto.silent.defend"):
        for action in actions:
            if action.get("kind") != "play_card":
                continue
            import json

            payload = json.loads(action.get("payloadJson", "{}"))
            if hand.get(payload.get("CardInstanceId")) == preferred:
                return action["actionId"]
    for action in actions:
        if action.get("kind") == "end_turn":
            return action["actionId"]
    return first_legal_player(_observation, actions)


def play_real_two_site(
    session: SemanticSession,
    fixture: CombatFixture, *,
    emulator_revision: str,
    nature: TwoSiteNature | None = None,
    player_choose: Callable[[dict[str, Any], list[dict[str, Any]]], str] = first_legal_player,
    scripted_choices: tuple[tuple[str, str, str], ...] | None = None,
    max_boundaries: int = 512,
    replay_terminal_snapshot: bool = True,
) -> RealCombatEpisode:
    """Return one *real* terminal combat, or fail without labeling a loss.

    For scripted replay, validate the ordered source support identity,
    selected card/intent IDs, every parent/result fingerprint, and the
    genuine terminal outcome. The caller may compare trace digests.
    """
    if max_boundaries <= 0 or not emulator_revision:
        raise ValueError("Missing combat budget/emulator revision")
    check_capabilities(session, fixture)
    state = session.begin_seeded_semantic_combat(
        fixture.seed, act=fixture.act, encounter_id=fixture.encounter_id,
        native_entry=fixture.native_entry, total_floor=fixture.total_floor,
        ascension=fixture.ascension,
        map_column=fixture.map_column, map_row=fixture.map_row,
        replay_opening_draw_hooks=fixture.replay_opening_draw_hooks,
        other_chance_provider_id=fixture.other_chance_provider_id,
    )
    handles: list[SemanticCombatState] = [state]
    decisions: list[CombatDecision] = []
    pilot_choices: list[PilotChoice] = []
    try:
        for turn in range(max_boundaries):
            if state.terminal:
                break
            prior = state.fingerprint
            raw = state.pending
            if state.boundary == "nature":
                if raw is None:
                    raise P0SemanticInterfaceError("Missing pending Nature event")
                event, features, identities = _pending_event(raw, prior)
                context = f"{event.site_id}|{event.owner_id}"
                if scripted_choices is not None:
                    if turn >= len(scripted_choices):
                        raise P0SemanticInterfaceError("Scripted replay exhausted")
                    phase, identity, chosen = scripted_choices[turn]
                    if (phase != f"nature:{event.operation}"
                            or identity != event.replay_identity
                            or chosen not in identities):
                        raise P0SemanticInterfaceError("Scripted support/constraint divergence")
                    index = identities.index(chosen)
                    event.check_index(index)
                    log_q = None
                else:
                    if nature is None:
                        raise P0SemanticInterfaceError("Nature policy required")
                    proposal, choice = nature.propose(
                        event, context_key=context,
                        candidate_features=features,
                        episode_key=f"{fixture.fixture_id}:{turn}",
                    )
                    index = proposal.selected_index
                    chosen = proposal.selected_id
                    log_q = proposal.log_q
                    pilot_choices.append(choice)
                child = (state.resolve_intent(chosen)
                         if event.operation == "enemy_intent"
                         else state.resolve(index))
                label = f"nature:{event.operation}"
                identity = event.replay_identity
            elif state.boundary == "player":
                if raw is not None:
                    raise P0SemanticInterfaceError("Hidden Nature support at player boundary")
                obs = state.observe()
                actions = state.actions()
                if scripted_choices is not None:
                    if turn >= len(scripted_choices):
                        raise P0SemanticInterfaceError("Player replay exhausted")
                    phase, identity, chosen = scripted_choices[turn]
                    if (phase != "player" or identity !=
                            canonical_sha256([obs, actions])):
                        raise P0SemanticInterfaceError("Player observation/menu drift")
                else:
                    chosen = player_choose(obs, actions)
                if chosen not in [a["actionId"] for a in actions]:
                    raise P0SemanticInterfaceError("Illegal fair player selection")
                child = state.step(chosen)
                label = "player"
                identity = canonical_sha256([obs, actions])
                log_q = None
            else:
                raise P0SemanticInterfaceError(f"Unknown semantic boundary: {state.boundary}")
            decisions.append(CombatDecision(
                label, prior, identity, chosen, child.fingerprint, log_q,
            ))
            handles.append(child)
            state = child
        if not state.terminal:
            raise P0SemanticInterfaceError("Combat boundary limit: censored, not a defeat")
        if (state.boundary != "terminal" or state.pending is not None
                or state.outcome not in ("victory", "defeat")
                or type(state.exit_hp) is not int):
            raise P0SemanticInterfaceError("Unverified semantic terminal outcome")
        if scripted_choices is not None and len(scripted_choices) != len(decisions):
            raise P0SemanticInterfaceError("Unconsumed scripted decisions")
        status = state.status() if hasattr(state, "status") else {}
        provider = status.get("otherChanceProvider", fixture.other_chance_provider_id)
        if provider != fixture.other_chance_provider_id:
            raise P0SemanticInterfaceError("Engine-owned RNG provider provenance drift")
        if replay_terminal_snapshot:
            restored = session.load_semantic_snapshot(state.save_snapshot())
            handles.append(restored)
            if restored.fingerprint != state.fingerprint:
                raise P0SemanticInterfaceError("Terminal snapshot replay diverged")
        trace = CombatTrace(
            fixture=fixture, emulator_revision=emulator_revision,
            policy_id=nature.policy_id if nature is not None else "scripted-replay",
            decisions=tuple(decisions), outcome=state.outcome,
            exit_hp=state.exit_hp,
            terminal_fingerprint=state.fingerprint,
            other_chance_provider=provider,
        )
        reward = (1.0 if state.outcome == "victory" else 0.0) + (
            max(0, state.exit_hp) / 100.0
        )
        return RealCombatEpisode(
            trace, reward, PilotEpisode(reward, tuple(pilot_choices)),
        )
    finally:
        session.release_semantic_many(handles)
