"""Exact factored initial-map posterior under a SPECIFIED alternate RNG prior.

At the RunStart -> MapChoice boundary ONLY, the pinned prototype's visible
map and map-choice menu use the `map` SplitMix64 stream, while the visible
Act 1 boss uses `combat`. Four other named streams are not touched by
RunStart. This versioned model uses *independent initial 64-bit stream
states*, not the prototype's ordinary hashed one-seed prior or native STS2.

The declared prior has finite, uniform, fully enumerated supports for the
map and combat initial states, and full 64-bit uniform independent priors
for the other four streams. Conditioning at this exact boundary therefore
factorizes: inspect |map_support| + |combat_support| states, never the full
Cartesian product. Sample any surviving pair and uniformly chosen other
streams, then replay the actual engine's StartRun to retain every RNG cursor.

It deliberately refuses longer histories: after gameplay starts, mechanics
may couple previously independent streams through outcomes and actions.
"""
from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .chance import (
    FAIR_CONTINUATION_CAPABILITY_ID,
    FairContinuationUnavailable,
    PublicHistoryStep,
)
from .protocol import InformationPolicy, LegalAction, Observation, StateHandle, Transition
from .rejection import FairHistoryRejectionSampler

FACTORIZED_RUNSTART_PRIOR_ID = "independent-streams-factorized-runstart-v1"
FACTORIZED_RUNSTART_SCHEMA_ID = "prototype-independent-initial-streams-v1"
_EXTRA_STREAMS = ("combat_targets", "event", "reward", "shop")


class FactorizedRunStartUnavailable(FairContinuationUnavailable):
    """No compatible start-of-run state in the explicitly enumerated prior."""


class FactorizedRunStartBackend(Protocol):
    @property
    def factorized_initial_stream_schema(self) -> str | None: ...

    def reset_factorized_hypothetical(
        self, initial_streams: Mapping[str, int], ascension: int = 0
    ) -> StateHandle: ...

    def step(self, state: StateHandle, action: LegalAction) -> Transition: ...

    def observe(self, state: StateHandle, policy: InformationPolicy) -> Observation: ...

    def legal_actions(self, state: StateHandle) -> Sequence[LegalAction]: ...

    def release_many(self, states: Sequence[StateHandle]) -> int: ...


@dataclass(frozen=True, slots=True)
class FactorizedRunStartStats:
    map_candidates: int
    combat_candidates: int
    surviving_map_states: int
    surviving_combat_states: int
    inspected_run_starts: int

    @property
    def posterior_pairs(self) -> int:
        return self.surviving_map_states * self.surviving_combat_states

    @property
    def evidence_probability(self) -> float:
        if self.map_candidates == 0 or self.combat_candidates == 0:
            return 0.0
        return (
            self.surviving_map_states / self.map_candidates
            * self.surviving_combat_states / self.combat_candidates
        )


def _seed_support(values: Sequence[int], name: str, max_support: int) -> tuple[int, ...]:
    atoms = tuple(values)
    if not atoms or len(atoms) > max_support:
        raise ValueError(f"{name} support must be nonempty and within max_support")
    if any(type(n) is not int or n < 0 or n >= 1 << 64 for n in atoms):
        raise ValueError(f"{name} states must be unsigned 64-bit integers")
    if len(set(atoms)) != len(atoms):
        raise ValueError(f"{name} support has repeated atoms")
    return atoms


def _frame_minus_boss(payload_json: str) -> object:
    decoded = json.loads(payload_json)
    if not isinstance(decoded, dict):
        raise ValueError("Player observation must be a JSON object")
    if "act_one_boss_encounter_id" not in decoded:
        raise ValueError("Required visible Act 1 boss field is missing")
    return {
        key: value for key, value in decoded.items()
        if key != "act_one_boss_encounter_id"
    }


def _boss(payload_json: str) -> str:
    decoded = json.loads(payload_json)
    if not isinstance(decoded, dict):
        raise ValueError("Player observation must be a JSON object")
    boss = decoded.get("act_one_boss_encounter_id")
    if not isinstance(boss, str) or not boss:
        raise ValueError("A revealed Act 1 boss is required at MapChoice")
    return boss


class FactorizedRunStartPosteriorSampler:
    """Exact start-of-run posterior under an independent-stream research prior.

    The caller passes player-visible `RunStart -> MapChoice` history, never
    a live hidden-state handle or seed. All returned states are entirely
    independently created hypothetical runs; the caller owns their handles.
    """

    fair_continuation_capability_id = FAIR_CONTINUATION_CAPABILITY_ID
    seed_prior_id = FACTORIZED_RUNSTART_PRIOR_ID

    def __init__(
        self,
        backend: FactorizedRunStartBackend,
        *,
        map_states: Sequence[int],
        combat_states: Sequence[int],
        ascension: int = 0,
        max_support: int = 4096,
    ) -> None:
        if backend.factorized_initial_stream_schema != FACTORIZED_RUNSTART_SCHEMA_ID:
            raise FactorizedRunStartUnavailable(
                "Backend lacks independently initialized factorized RNG streams"
            )
        if type(max_support) is not int or max_support <= 0:
            raise ValueError("max_support must be positive")
        if type(ascension) is not int or ascension < 0:
            raise ValueError("ascension must be a nonnegative integer")
        self._backend = backend
        self._map_prior = _seed_support(map_states, "Map", max_support)
        self._combat_prior = _seed_support(combat_states, "Combat", max_support)
        self._ascension = ascension
        self._history: tuple[PublicHistoryStep, ...] | None = None
        self._accepted_maps: tuple[int, ...] = ()
        self._accepted_combats: tuple[int, ...] = ()
        self._inspected = 0
        self._closed = False

    @property
    def stats(self) -> FactorizedRunStartStats:
        return FactorizedRunStartStats(
            map_candidates=len(self._map_prior),
            combat_candidates=len(self._combat_prior),
            surviving_map_states=len(self._accepted_maps),
            surviving_combat_states=len(self._accepted_combats),
            inspected_run_starts=self._inspected,
        )

    @property
    def public_history(self) -> tuple[PublicHistoryStep, ...] | None:
        return self._history

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("Factorized posterior is closed")

    @staticmethod
    def _validate_history(history: Sequence[PublicHistoryStep]) -> (
        tuple[PublicHistoryStep, PublicHistoryStep]
    ):
        transcript = tuple(history)
        FairHistoryRejectionSampler._validate_history(transcript)
        if len(transcript) != 2:
            raise ValueError(
                "Factorized sampler only supports RunStart -> MapChoice, "
                "not longer coupled histories"
            )
        first, second = transcript
        if first.chosen_action is None or first.chosen_action.kind != "start_run":
            raise ValueError("StartRun action is required")
        if first.chosen_action not in (first.legal_actions or ()):
            raise ValueError("StartRun action must be in the public legal menu")
        if first.observation.policy_id != "prototype-fair-v0":
            raise ValueError("Factorized prior requires the prototype fair policy")
        state = json.loads(second.observation.payload_json)
        if not isinstance(state, dict) or state.get("phase") != 2:
            raise ValueError("Last frame must be a post-StartRun MapChoice")
        _boss(second.observation.payload_json)
        _frame_minus_boss(second.observation.payload_json)
        return first, second

    @staticmethod
    def _streams(map_state: int, combat_state: int,
                 additional: Mapping[str, int] | None = None) -> dict[str, int]:
        streams: dict[str, int] = {
            "map": map_state, "combat": combat_state
        }
        streams.update(additional or {key: 0 for key in _EXTRA_STREAMS})
        return streams

    def _replay_start(
        self, streams: Mapping[str, int], history: tuple[PublicHistoryStep, PublicHistoryStep]
    ) -> tuple[Observation, tuple[LegalAction, ...]]:
        first, _ = history
        root = self._backend.reset_factorized_hypothetical(
            streams, ascension=self._ascension
        )
        self._inspected += 1
        child: StateHandle | None = None
        try:
            policy = InformationPolicy(first.observation.policy_id)
            if (self._backend.observe(root, policy) != first.observation
                    or tuple(self._backend.legal_actions(root)) != first.legal_actions):
                raise FactorizedRunStartUnavailable(
                    "RunStart public frame is not shared by factorized prior atoms"
                )
            assert first.chosen_action is not None
            transition = self._backend.step(root, first.chosen_action)
            child = transition.child  # Engine transition, never hand-edit an RNG cursor.
            return (
                self._backend.observe(child, policy),
                tuple(self._backend.legal_actions(child)),
            )
        finally:
            if child is not None:
                self._backend.release_many((child,))
            self._backend.release_many((root,))

    def initialize(self, history: Sequence[PublicHistoryStep]) -> None:
        self._assert_open()
        if self._history is not None:
            raise RuntimeError("Posterior already initialized")
        first, second = self._validate_history(history)
        transcript = (first, second)
        target_rest = _frame_minus_boss(second.observation.payload_json)
        target_boss = _boss(second.observation.payload_json)
        target_menu = second.legal_actions
        assert target_menu is not None
        maps: list[int] = []
        combats: list[int] = []
        try:
            for map_state in self._map_prior:
                observed, menu = self._replay_start(
                    self._streams(map_state, self._combat_prior[0]), transcript
                )
                if _frame_minus_boss(observed.payload_json) == target_rest and menu == target_menu:
                    maps.append(map_state)
            for combat_state in self._combat_prior:
                observed, _ = self._replay_start(
                    self._streams(self._map_prior[0], combat_state), transcript
                )
                if _boss(observed.payload_json) == target_boss:
                    combats.append(combat_state)
            if not maps or not combats:
                raise FactorizedRunStartUnavailable(
                    "No atom of the independently factorized research prior "
                    "can produce the observed starting map and boss"
                )
            self._accepted_maps = tuple(maps)
            self._accepted_combats = tuple(combats)
            self._history = transcript
        except BaseException:
            self.close()
            raise

    def sample_fair_continuations(
        self,
        history: Sequence[PublicHistoryStep],
        *,
        search_rng: random.Random,
        count: int,
    ) -> tuple[StateHandle, ...]:
        self._assert_open()
        if self._history is None or tuple(history) != self._history:
            raise ValueError("Requested history differs from initialized factorized posterior")
        if type(count) is not int or count <= 0:
            raise ValueError("Sample count must be positive")
        transcript = self._history
        first, second = transcript
        policy = InformationPolicy(first.observation.policy_id)
        results: list[StateHandle] = []
        try:
            for _ in range(count):
                streams = self._streams(
                    search_rng.choice(self._accepted_maps),
                    search_rng.choice(self._accepted_combats),
                    {name: search_rng.getrandbits(64) for name in _EXTRA_STREAMS},
                )
                root = self._backend.reset_factorized_hypothetical(
                    streams, ascension=self._ascension
                )
                try:
                    if (self._backend.observe(root, policy) != first.observation
                            or tuple(self._backend.legal_actions(root)) != first.legal_actions):
                        raise FactorizedRunStartUnavailable(
                            "A sampled prior atom disagrees at RunStart"
                        )
                    assert first.chosen_action is not None
                    child = self._backend.step(root, first.chosen_action).child
                finally:
                    self._backend.release_many((root,))
                try:
                    actual = self._backend.observe(child, policy)
                    legal = tuple(self._backend.legal_actions(child))
                    if actual != second.observation or legal != second.legal_actions:
                        raise FactorizedRunStartUnavailable(
                            "Factorized independence audit failed: accepted map/boss "
                            "streams produced a different public frame"
                        )
                    results.append(child)
                except BaseException:
                    self._backend.release_many((child,))
                    raise
            return tuple(results)
        except BaseException:
            if results:
                self._backend.release_many(tuple(results))
            raise

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> FactorizedRunStartPosteriorSampler:
        self._assert_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
