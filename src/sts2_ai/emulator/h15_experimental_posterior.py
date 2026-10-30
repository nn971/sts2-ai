"""Experimental public-history conditioning of real-entry H15 ReferenceNature.

The starting RunState is an exactly replayed *authentic native-structure /
prototype-combat-RNG* combat entry. Each candidate independently replaces
its hidden future with the explicitly UNVERIFIED H15 research prior. All
chance events are resolved by the emulator's ReferenceNatureV111.

Conditioning checks the *complete* visible player observations and ordered
menus after replaying the prescribed public actions. This is a posterior for
the independent experimental candidate prior ONLY, not the source run's
hidden-state posterior. It deliberately does not implement FairContinuationSampler,
advertise a fair capability, or supply search/training targets.
"""
from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sts2_ai.emulator.chance import PublicHistoryStep
from sts2_ai.emulator.h15_public_protocol import (
    NATURE_KINDS,
    choose_reference_at_pending,
    public_player_frame,
)
from sts2_ai.emulator.jsonl_backend import FAIR_POLICY_ID
from sts2_ai.emulator.protocol import InformationPolicy, LegalAction

H15_EXPERIMENTAL_CONDITIONING_ID = (
    "h15-native-structure-prototype-rng-reference-conditioning-unverified-v1"
)
H15_CANDIDATE_MODEL_ID = (
    "h15-native-structure-prototype-rng-candidate-unverified-v1"
)
SUPPORTED_PLAYER_ACTIONS = frozenset({"play_card", "use_potion", "end_turn"})


class ExperimentalConditioningUnavailable(RuntimeError):
    """Posterior exhausted or mechanics cannot represent the given history."""


@dataclass(frozen=True)
class ExperimentalConditioningStats:
    attempted: int = 0
    accepted: int = 0
    mismatched: int = 0
    sampled_nature_events: int = 0


class H15ExperimentalReferencePosterior:
    """Rejection conditioning under the emulator's independent candidate prior.

    Accepted live handles belong to the caller until release_many(). Unsupported
    mechanics abort the *whole* request: dropping those candidates would silently
    condition on emulator coverage as well as the visible player history.
    """

    model_id = H15_EXPERIMENTAL_CONDITIONING_ID
    probability_status = "UNVERIFIED"
    fair_continuation_capability_id = "unavailable"

    def __init__(self, backend: Any, source: str, *,
                 max_candidates: int = 4096, max_nature_events: int = 1200):
        if not isinstance(source, str) or not source:
            raise ValueError("A live, exact-replayed combat-entry source is required")
        if (type(max_candidates) is not int or max_candidates < 1
                or type(max_nature_events) is not int or max_nature_events < 1):
            raise ValueError("Positive conditioning and Nature budgets are required")
        self.backend = backend
        self.source = source
        self.max_candidates = max_candidates
        self.max_nature_events = max_nature_events
        self.last_stats = ExperimentalConditioningStats()

    @staticmethod
    def _validate_history(
        history: Sequence[PublicHistoryStep],
    ) -> tuple[PublicHistoryStep, ...]:
        steps = tuple(history)
        if not steps or steps[-1].chosen_action is not None:
            raise ValueError("Experimental history must end at an uncommitted player turn")
        for i, step in enumerate(steps):
            if (step.observation.policy_id != FAIR_POLICY_ID
                    or not step.observation.observation_hash
                    or not isinstance(step.observation.payload_json, str)
                    or step.legal_actions is None or not step.legal_actions):
                raise ValueError("Incomplete public H15 observation/menu history")
            if (len({a.action_id for a in step.legal_actions}) !=
                    len(step.legal_actions)):
                raise ValueError("Duplicate legal actions in H15 history")
            if i < len(steps) - 1:
                action = step.chosen_action
                if (action is None or action not in step.legal_actions
                        or action.kind not in SUPPORTED_PLAYER_ACTIONS):
                    raise ValueError("Unsupported or illegal H15 history action")
        return steps

    def _settle_reference(
        self, handle: str, frame: dict[str, Any], remaining: int,
    ) -> tuple[dict[str, Any], int]:
        resolved = 0
        while frame.get("boundaryKind") in NATURE_KINDS:
            if resolved >= remaining:
                raise ExperimentalConditioningUnavailable(
                    "H15 candidate exceeded its Nature-event conditioning budget"
                )
            old_count = frame.get("referenceNatureSampleCount")
            frame, _ = choose_reference_at_pending(self.backend, handle, frame)
            if (type(old_count) is not int
                    or frame.get("referenceNatureSampleCount") != old_count + 1):
                raise ValueError("H15 ReferenceNature cursor count did not advance once")
            resolved += 1
        if frame.get("boundaryKind") not in ("player", "combat_exit"):
            raise ExperimentalConditioningUnavailable(
                "H15 candidate hit unsupported mechanics boundary: " +
                str(frame.get("boundaryKind"))
            )
        return frame, resolved

    def _matches(
        self, handle: str, frame: dict[str, Any],
        steps: tuple[PublicHistoryStep, ...],
    ) -> tuple[bool, int]:
        nature_events = 0
        for step in steps:
            frame, count = self._settle_reference(
                handle, frame, self.max_nature_events - nature_events,
            )
            nature_events += count
            if frame["boundaryKind"] != "player":
                return False, nature_events
            observation, actions = public_player_frame(frame)
            if (observation != step.observation or
                    actions != step.legal_actions):
                return False, nature_events
            if step.chosen_action is None:
                return True, nature_events
            action = step.chosen_action
            try:
                if action.kind == "end_turn":
                    frame = self.backend.h15_experimental_end_turn(
                        handle, frame["version"], action.action_id,
                    )
                else:
                    frame = self.backend.h15_experimental_player_action(
                        handle, frame["version"], action.action_id,
                    )
            except Exception as exc:
                if "UNSUPPORTED_" in str(exc):
                    raise ExperimentalConditioningUnavailable(
                        "H15 mechanics cannot replay the public history: " +
                        str(exc)
                    ) from exc
                raise
        raise AssertionError("H15 final public decision was not processed")

    def sample_conditioned(
        self, history: Sequence[PublicHistoryStep], *,
        search_rng: random.Random, count: int,
    ) -> tuple[str, ...]:
        steps = self._validate_history(history)
        if type(count) is not int or count < 1:
            raise ValueError("Positive candidate count required")
        accepted: list[str] = []
        attempted = 0
        mismatched = 0
        nature_events = 0
        self.last_stats = ExperimentalConditioningStats()
        try:
            while len(accepted) < count and attempted < self.max_candidates:
                # This seed proposes an independent candidate *prior*, never
                # an outcome of a Chance Law. The latter is sampled in C#.
                independent_seed = f"{search_rng.getrandbits(256):064x}"
                attempted += 1
                frame = self.backend.h15_experimental_begin(
                    self.source, independent_seed=independent_seed,
                )
                handle = frame.get("h15Handle")
                if not isinstance(handle, str) or not handle:
                    raise ValueError("H15 begin returned no owned session handle")
                try:
                    if (frame.get("candidateModelId") != H15_CANDIDATE_MODEL_ID
                            or frame.get("fairContinuationCapability") != "unavailable"):
                        raise ValueError("H15 candidate lacks unverified provenance")
                    matched, count_nature = self._matches(handle, frame, steps)
                    nature_events += count_nature
                    if matched:
                        accepted.append(handle)
                        handle = ""
                    else:
                        mismatched += 1
                finally:
                    if handle:
                        self.backend.h15_experimental_release_many((handle,))
            self.last_stats = ExperimentalConditioningStats(
                attempted, len(accepted), mismatched, nature_events,
            )
            if len(accepted) < count:
                raise ExperimentalConditioningUnavailable(
                    "H15 experimental ReferenceNature conditioning exhausted: "
                    f"{len(accepted)}/{attempted} accepted, "
                    f"public_frames={len(steps)}; no posterior samples returned"
                )
            return tuple(accepted)
        except BaseException:
            self.last_stats = ExperimentalConditioningStats(
                attempted, len(accepted), mismatched, nature_events,
            )
            if accepted:
                self.backend.h15_experimental_release_many(accepted)
            raise


def capture_source_public_history(
    backend: Any, source: str, action_ids: Sequence[str],
) -> tuple[PublicHistoryStep, ...]:
    """Capture authentic emulator public evidence without leaking source RNG.

    Exact source state is used only in this offline observer. Returned history
    contains fair observations, source-legal menus and committed *public*
    player actions. Stops at another active combat player decision; the source
    and its hidden cursor remain untouched.
    """
    if not source:
        raise ValueError("Missing source combat-entry handle")
    current = backend.fork(source)
    steps: list[PublicHistoryStep] = []
    try:
        for selected_id in (*action_ids, None):
            observation = backend.observe(
                current, InformationPolicy(FAIR_POLICY_ID),
            )
            public = json.loads(observation.payload_json)
            if not isinstance(public, dict) or not isinstance(
                    public.get("combat"), dict):
                raise ExperimentalConditioningUnavailable(
                    "Source prefix exited combat before a final public decision"
                )
            menu = tuple(backend.legal_actions(current))
            if not menu:
                raise ValueError("Source player boundary omitted legal actions")
            selected: LegalAction | None = None
            if selected_id is not None:
                selected = next(
                    (a for a in menu if a.action_id == selected_id), None,
                )
                if selected is None or selected.kind not in SUPPORTED_PLAYER_ACTIONS:
                    raise ValueError("Source prefix contains an unsupported action")
            steps.append(PublicHistoryStep(observation, selected, menu))
            if selected is not None:
                next_state = backend.step(current, selected).child
                backend.release_many((current,))
                current = next_state
        return H15ExperimentalReferencePosterior._validate_history(steps)
    finally:
        backend.release_many((current,))
