"""Independent Reference Nature posterior for *fixed* H8 combat fixtures.

This is a useful, explicitly scoped combat model. It is NOT the true native
full-run hidden-state posterior: fixture seed and all non-Reference-Nature
mechanics are held fixed. The sampled variable is an independent signed
63-bit Reference Nature seed, with deterministic event-scoped card draws,
intents and random-target decisions provided by the H8 emulator.

Conditioning replays the ENTIRE visible combat action/observation/legal-menu
history from the fixture's initial public player decision. It never forks a
source playthrough, copies its hidden Nature cursor, or accepts a mismatch.
Rejection exhaustion and unsupported stochastic sites fail loudly.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from sts2_ai.emulator.chance import (
    FAIR_CONTINUATION_CAPABILITY_ID,
    FairContinuationUnavailable,
    PublicHistoryStep,
)
from sts2_ai.emulator.jsonl_backend import FAIR_POLICY_ID
from sts2_ai.emulator.protocol import InformationPolicy, LegalAction, Observation, Transition

H8_REFERENCE_MODEL = "h15-h8-fixed-fixture-independent-reference-nature-v1"
H8_DECLARED_LAW_MODEL = "h15-h8-fixed-fixture-declared-chance-law-v1"
H8_RUNTIME_SCHEMA = "sts2-h8-versioned-combat-runtime-v1"


class H8SessionTransport(Protocol):
    def call(self, op: str, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class H8FixtureEntry:
    """Public research-fixture specification, pinned independently of Nature."""
    seed: str
    encounter_id: str
    native_entry: str
    total_floor: int
    act: int = 1
    ascension: int = 10
    first_monster_combat_id: int = 1
    map_column: int = 3
    map_row: int = 5
    fixture_deck_preset_id: str | None = None
    fixture_spec: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (not self.seed or not self.encounter_id or not self.native_entry
                or type(self.total_floor) is not int or self.total_floor < 1
                or self.act != 1 or self.ascension != 10
                or (self.fixture_spec is not None
                    and self.fixture_deck_preset_id is not None)):
            raise ValueError("H15 reference search requires pinned A10 combat fixture")

    def wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "seed": self.seed,
            "act": self.act,
            "encounter_id": self.encounter_id,
            "native_entry": self.native_entry,
            "total_floor": self.total_floor,
            "ascension": self.ascension,
            "first_monster_combat_id": self.first_monster_combat_id,
            "map_column": self.map_column,
            "map_row": self.map_row,
            "replay_opening_draw_hooks": True,
            "include_player_frame": True,
        }
        if self.fixture_deck_preset_id is not None:
            payload["fixture_deck_preset_id"] = self.fixture_deck_preset_id
        if self.fixture_spec is not None:
            payload["fixture_spec"] = dict(self.fixture_spec)
        return payload


@dataclass
class _LiveH8:
    status: dict[str, Any]
    nature_seed: int
    opening_max_hp: int
    chance_rng: random.Random | None = None


class H8ReferenceCombatBackend:
    """Mutable H8 JSONL sessions exposed through the H15 combat interface.

    The H8 runtime mutates the *same* session handle on each step. Search
    adapters must not release the parent when parent == transition.child.
    We retain no source hidden state for simulations; all proposed handles
    are separately created by begin_reference(nature_seed).
    """

    mutable_transition_capability_id = "h15-h8-inplace-player-step-v1"
    reference_model_id = H8_REFERENCE_MODEL
    binding_version = H8_RUNTIME_SCHEMA

    def __init__(self, session: H8SessionTransport, entry: H8FixtureEntry, *,
                 emulator_revision: str,
                 use_declared_chance_laws: bool = False):
        if not emulator_revision:
            raise ValueError("Pin the emulator revision")
        self._session = session
        self._entry = entry
        self.emulator_revision = emulator_revision
        self._use_declared_chance_laws = use_declared_chance_laws
        self.reference_model_id = (
            H8_DECLARED_LAW_MODEL if use_declared_chance_laws
            else H8_REFERENCE_MODEL
        )
        self._live: dict[str, _LiveH8] = {}

    @staticmethod
    def _actions(status: Mapping[str, Any]) -> tuple[LegalAction, ...]:
        kind = status.get("kind")
        if kind == "terminal":
            return ()
        if kind not in ("player", "player_choice"):
            raise FairContinuationUnavailable(
                "Reference Nature encountered unsupported player boundary: "
                + str(kind)
            )
        frame = status.get("playerFrame")
        if not isinstance(frame, dict) or not isinstance(frame.get("actions"), list):
            raise ValueError("Missing H8 inline source-public player menu")
        public_menu = frame["actions"]
        actions = tuple(LegalAction(
            a["actionId"], a["kind"], a["payloadJson"],
        ) for a in public_menu)
        if (not actions or len({a.action_id for a in actions}) != len(actions)
                or [a.action_id for a in actions] != status.get("playerActionIds")):
            raise ValueError("H8 player menu disagrees with boundary status")
        if kind == "player_choice":
            # The H8 status describes which source-legal choices are pending,
            # but ONLY actions from the public observation are returned to
            # the player. Reuse the source-verified legality filter.
            from sts2_ai.nature.draw_only_episode import fair_player_choice_actions

            pending = status.get("playerChoicePending")
            if not isinstance(pending, dict) or pending.get("kind") not in (
                    "player_card_selection", "draw_hook_action_choice"):
                raise FairContinuationUnavailable(
                    "H8 pending player selection is unsupported"
                )
            selected = fair_player_choice_actions(pending, public_menu)
            ids = {a["actionId"] for a in selected}
            actions = tuple(action for action in actions if action.action_id in ids)
            if not actions:
                raise ValueError("H8 nested player choice has no public actions")
        return actions

    @staticmethod
    def _observation(status: Mapping[str, Any], max_hp: int) -> Observation:
        if status.get("kind") == "terminal":
            result = status.get("outcome")
            hp = status.get("exitHp")
            if result not in ("victory", "defeat") or type(hp) is not int:
                raise ValueError("Incomplete H8 terminal combat status")
            public = {
                "combat": None, "terminal_outcome": result,
                "hp": hp, "max_hp": max_hp,
                "defeat_total_non_minion_enemy_hp":
                    status.get("defeatTotalNonMinionEnemyHp"),
                "defeat_remaining_non_minion_enemy_hp":
                    status.get("defeatRemainingNonMinionEnemyHp"),
            }
            # Exact enemy totals here come from H8 terminal public score,
            # and are exposed only AFTER combat has actually terminated.
            payload = json.dumps(public, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(payload.encode()).hexdigest()
            return Observation(FAIR_POLICY_ID, payload, digest)
        frame = status.get("playerFrame")
        if not isinstance(frame, dict) or not isinstance(frame.get("observation"), dict):
            raise ValueError("Missing H8 inline fair observation")
        public = frame["observation"]
        if not isinstance(public.get("combat"), dict):
            raise ValueError("H8 ordinary player decision outside combat")
        if status.get("kind") == "player_choice":
            pending = status.get("playerChoicePending")
            if not isinstance(pending, dict) or pending.get("kind") not in (
                    "player_card_selection", "draw_hook_action_choice"):
                raise FairContinuationUnavailable("Unsupported H8 nested player choice")
            # Public kind marker only. The source pending object (which can
            # contain private implementation details) is never forwarded.
            public = dict(public)
            public["h15_player_choice_kind"] = pending["kind"]
        payload = json.dumps(public, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        return Observation(FAIR_POLICY_ID, payload, digest)

    def _resolve_declared_laws(
        self, status: dict[str, Any], rng: random.Random,
    ) -> dict[str, Any]:
        """Sample only from the emulator-declared conditional laws.

        This exercises the same typed Nature interface needed by native H15.
        It remains a *fixed-fixture* stochastic model; it does not construct
        the posterior over arbitrary authentic native-run hidden states.
        """
        for _ in range(1200):
            kind = status.get("kind")
            if kind in ("player", "player_choice", "terminal"):
                return status
            if kind not in ("draw_card", "enemy_intent", "random_target"):
                raise FairContinuationUnavailable(
                    "No declared H8 Chance Law for boundary " + str(kind)
                )
            handle = status.get("h8Handle")
            token = {
                "h8_handle": handle,
                "runtime_id": status["runtimeId"],
                "boundary_version": status["boundaryVersion"],
            }
            pending = self._session.call("h8_reference_law", **token)["pending"]
            law = pending.get("chanceLaw")
            if (pending.get("kind") != kind or not isinstance(law, dict)
                    or law.get("eventId") != pending.get("eventId")):
                raise ValueError("Declared Chance Law does not match pending event")
            candidates = law.get("outcomes")
            total = law.get("totalWeight")
            if (not isinstance(candidates, list) or not candidates
                    or type(total) is not int or total <= 0):
                raise ValueError("Invalid H8 Chance Law support")
            weights = [candidate.get("weight") for candidate in candidates]
            if (any(type(w) is not int or w <= 0 for w in weights)
                    or sum(weights) != total):
                raise ValueError("Inconsistent declared Chance Law weights")
            ticket = rng.randrange(total)
            selected = None
            for candidate, weight in zip(candidates, weights, strict=True):
                ticket -= weight
                if ticket < 0:
                    selected = candidate.get("outcomeId")
                    break
            if not isinstance(selected, str) or not selected:
                raise ValueError("Malformed H8 Chance Law outcome")
            following = self._session.call(
                "h8_resolve_chance_law_outcome", **token,
                chance_law=law, outcome_id=selected,
                provider_id="h15-declared-law-reference-v1",
                include_player_frame=True,
            )["result"]
            if (following.get("h8Handle") != handle
                    or following.get("runtimeId") != status["runtimeId"]
                    or following.get("boundaryVersion", -1) <=
                    status["boundaryVersion"]):
                raise ValueError("H8 Chance Law resolution failed to advance")
            status = following
        raise FairContinuationUnavailable(
            "H8 declared Chance Law continuation exceeded 1200 events"
        )

    def _store(self, status: dict[str, Any], nature_seed: int,
               chance_rng: random.Random | None = None) -> str:
        handle = status.get("h8Handle")
        if not isinstance(handle, str) or not handle or handle in self._live:
            raise ValueError("Invalid or duplicate H8 session handle")
        if status.get("schemaId") != H8_RUNTIME_SCHEMA:
            raise ValueError("H8 runtime schema mismatch")
        if status.get("kind") not in ("player", "player_choice", "terminal"):
            raise FairContinuationUnavailable(
                "H8 reference start has unsupported boundary "
                + str(status.get("kind"))
            )
        opening = status.get("playerFrame")
        if status.get("kind") == "terminal":
            raise FairContinuationUnavailable("Combat ended before first player observation")
        if not isinstance(opening, dict) or not isinstance(opening.get("observation"), dict):
            raise ValueError("H8 public opening missing")
        max_hp = opening["observation"].get("max_hp")
        if type(max_hp) is not int or max_hp < 1:
            raise ValueError("H8 public opening max HP missing")
        self._actions(status)
        self._live[handle] = _LiveH8(status, nature_seed, max_hp, chance_rng)
        return handle

    def begin_reference(self, nature_seed: int) -> str:
        if type(nature_seed) is not int or not 0 <= nature_seed < (1 << 63):
            raise ValueError("Reference Nature seed must be a signed 63-bit nonnegative int")
        args: dict[str, Any] = {**self._entry.wire()}
        if not self._use_declared_chance_laws:
            args["local_reference_seed"] = nature_seed
        response = self._session.call("h8_begin_seeded", **args)
        status = response["result"]
        handle = status.get("h8Handle")
        chance_rng = (
            random.Random(nature_seed) if self._use_declared_chance_laws else None
        )
        try:
            if chance_rng is not None:
                status = self._resolve_declared_laws(status, chance_rng)
            return self._store(status, nature_seed, chance_rng)
        except BaseException:
            if isinstance(handle, str):
                self._session.call("h8_release_many", h8_handles=[handle])
            raise

    def observe(self, state: str, policy: InformationPolicy) -> Observation:
        if policy.policy_id != FAIR_POLICY_ID:
            raise ValueError("H8 search only exposes the fair player observation")
        live = self._live[state]
        return self._observation(live.status, live.opening_max_hp)

    def legal_actions(self, state: str) -> tuple[LegalAction, ...]:
        return self._actions(self._live[state].status)

    def is_terminal(self, state: str) -> bool:
        return self._live[state].status.get("kind") == "terminal"

    def step(self, state: str, action: LegalAction) -> Transition:
        live = self._live[state]
        if action not in self.legal_actions(state):
            raise ValueError("Illegal H8 player action")
        token = {
            "h8_handle": state,
            "runtime_id": live.status["runtimeId"],
            "boundary_version": live.status["boundaryVersion"],
        }
        args: dict[str, Any] = {
            **token, "action_id": action.action_id, "include_player_frame": True,
        }
        if live.chance_rng is None:
            args["local_reference_seed"] = live.nature_seed
        nxt = self._session.call("h8_step", **args)["result"]
        if live.chance_rng is not None:
            nxt = self._resolve_declared_laws(nxt, live.chance_rng)
        if nxt.get("h8Handle") != state:
            raise ValueError("H8 changed an in-place session handle")
        if nxt.get("boundaryVersion", -1) <= live.status["boundaryVersion"]:
            raise ValueError("H8 reference step failed to advance")
        if nxt.get("kind") not in ("player", "player_choice", "terminal"):
            raise FairContinuationUnavailable(
                "H8 reference future encountered unsupported boundary "
                + str(nxt.get("kind"))
            )
        if nxt.get("kind") in ("player", "player_choice"):
            self._actions(nxt)
        live.status = nxt
        return Transition(state, action, state, self.is_terminal(state))

    def release_many(self, states: Sequence[str]) -> int:
        ids = list(dict.fromkeys(states))
        owned = [h for h in ids if h in self._live]
        if owned:
            self._session.call("h8_release_many", h8_handles=owned)
            for h in owned:
                del self._live[h]
        return len(owned)

    def close(self) -> None:
        self.release_many(tuple(self._live))

    @property
    def live_handles(self) -> int:
        return len(self._live)


@dataclass(frozen=True)
class H8ReferenceConditioningStats:
    candidates: int
    accepted: int
    mismatched: int


class H8ReferenceNaturePosterior:
    """Rejection conditional sampler for the declared fixed-fixture H8 law.

    Each accepted handle comes from its OWN new Reference Nature seed.
    The source's actual nature seed or hidden cursor is never read.
    """

    fair_continuation_capability_id = FAIR_CONTINUATION_CAPABILITY_ID
    prior_id = H8_REFERENCE_MODEL

    def __init__(self, backend: H8ReferenceCombatBackend, *,
                 max_candidates: int = 4096):
        if type(max_candidates) is not int or max_candidates < 1:
            raise ValueError("Invalid rejection budget")
        self.backend = backend
        self.prior_id = backend.reference_model_id
        self.max_candidates = max_candidates
        self.last_stats = H8ReferenceConditioningStats(0, 0, 0)

    def sample_fair_continuations(
        self, history: Sequence[PublicHistoryStep], *,
        search_rng: random.Random, count: int,
    ) -> tuple[str, ...]:
        transcript = tuple(history)
        if not transcript or count <= 0 or type(count) is not int:
            raise ValueError("Expected nonempty public combat history and count")
        if any(record.legal_actions is None or
               record.observation.policy_id != FAIR_POLICY_ID for record in transcript):
            raise ValueError("Incomplete or wrong-policy H8 public conditioning history")
        if any(record.chosen_action is None for record in transcript[:-1]
               ) or transcript[-1].chosen_action is not None:
            raise ValueError("H8 history must end at the uncommitted public decision")
        # This backend only represents combat-local histories starting at the
        # very first player decision of the fixture. Macro/Neow prefixes are
        # outside this conditional model and therefore explicitly unsupported.
        accepted: list[str] = []
        attempted = 0
        try:
            while len(accepted) < count and attempted < self.max_candidates:
                attempted += 1
                candidate = self.backend.begin_reference(search_rng.randrange(1 << 63))
                try:
                    matched = True
                    for step in transcript:
                        current = self.backend.observe(
                            candidate, InformationPolicy(FAIR_POLICY_ID),
                        )
                        menu = self.backend.legal_actions(candidate)
                        if (current.payload_json != step.observation.payload_json
                                or menu != step.legal_actions):
                            matched = False
                            break
                        if step.chosen_action is not None:
                            if step.chosen_action not in menu:
                                matched = False
                                break
                            self.backend.step(candidate, step.chosen_action)
                    if matched:
                        accepted.append(candidate)
                        candidate = ""
                finally:
                    if candidate:
                        self.backend.release_many((candidate,))
            self.last_stats = H8ReferenceConditioningStats(
                attempted, len(accepted), attempted - len(accepted),
            )
            if len(accepted) != count:
                raise FairContinuationUnavailable(
                    "H8 fixed-fixture Reference Nature conditioning exhausted: "
                    f"{len(accepted)}/{attempted} accepted, "
                    f"history_frames={len(transcript)}. No fair root was returned."
                )
            return tuple(accepted)
        except BaseException:
            if accepted:
                self.backend.release_many(accepted)
            raise
