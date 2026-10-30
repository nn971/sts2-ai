"""Bridge fair public-history continuation samplers to H15 combat PUCT.

The ONLY provider of hidden states is a certified, independently seeded
FairContinuationSampler. The adapter never forks a live root state for search.
It closes a combat on the first public noncombat frame, before card/relic
reward selection, using the authoritative H9 terminal score.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from typing import Any

from sts2_ai.emulator.chance import (
    FairContinuationSampler,
    PublicHistoryStep,
    require_fair_sampler,
)
from sts2_ai.emulator.protocol import EmulatorBackend, InformationPolicy, LegalAction
from sts2_ai.nature.h4_reward import encounter_total_hp_score

from .combat_puct import FAIR_COMBAT_CAPABILITY, CombatFrame


class H15PublicCombatAdapter:
    fair_combat_capability_id = FAIR_COMBAT_CAPABILITY

    def __init__(self, backend: EmulatorBackend, *, sampler: FairContinuationSampler):
        self._backend = backend
        self._sampler = require_fair_sampler(sampler)
        self._histories: dict[str, tuple[PublicHistoryStep, ...]] = {}
        self._active_handle: str | None = None
        self._active_key: str | None = None

    @staticmethod
    def _key(history: Sequence[PublicHistoryStep]) -> str:
        rows = [
            [step.observation.policy_id, step.observation.payload_json,
             None if step.legal_actions is None else [
                 [a.action_id, a.kind, a.payload_json] for a in step.legal_actions
             ],
             None if step.chosen_action is None else [
                 step.chosen_action.action_id, step.chosen_action.kind,
                 step.chosen_action.payload_json,
             ]]
            for step in history
        ]
        return "h15-public-history:" + hashlib.sha256(
            json.dumps(rows, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _combat_result(public: dict[str, Any]) -> float | None:
        if isinstance(public.get("combat"), dict):
            if public.get("terminal_outcome") == "defeat":
                raise ValueError("Contradictory defeated public combat frame")
            return None
        hp, max_hp = public.get("hp"), public.get("max_hp")
        if type(hp) is not int or type(max_hp) is not int:
            raise ValueError("Postcombat public player HP missing")
        defeat = public.get("terminal_outcome") == "defeat" or hp == 0
        return encounter_total_hp_score(
            "defeat" if defeat else "victory",
            player_exit_hp=hp, player_max_hp=max_hp,
            total_enemy_hp=public.get("defeat_total_non_minion_enemy_hp"),
            remaining_enemy_hp=public.get("defeat_remaining_non_minion_enemy_hp"),
        )

    def root(self, history: Sequence[PublicHistoryStep]) -> CombatFrame:
        transcript = tuple(history)
        if not transcript or any(step.legal_actions is None for step in transcript):
            raise ValueError("Search needs complete public observation and legal-menu history")
        last = transcript[-1]
        if last.chosen_action is not None or not last.legal_actions:
            raise ValueError("Search root must end in an uncommitted legal choice")
        if any(step.chosen_action is not None and step.chosen_action not in step.legal_actions
               for step in transcript):
            raise ValueError("History contains an action outside its public menu")
        public = json.loads(last.observation.payload_json)
        if not isinstance(public, dict) or not isinstance(public.get("combat"), dict):
            raise ValueError("Search root must be an observable combat state")
        key = self._key(transcript)
        existing = self._histories.get(key)
        if existing is not None and existing != transcript:
            raise ValueError("Public history hash collision")
        self._histories[key] = transcript
        return CombatFrame(key, last.observation, last.legal_actions)

    def begin_simulation(self, root: CombatFrame, search_rng: random.Random) -> None:
        if self._active_handle is not None:
            raise RuntimeError("Previous fair continuation was not released")
        transcript = self._histories.get(root.information_key)
        if transcript is None or transcript[-1].observation != root.observation:
            raise ValueError("Unregistered search root")
        handles = self._sampler.sample_fair_continuations(
            transcript, search_rng=search_rng, count=1,
        )
        if len(handles) != 1:
            raise ValueError("Fair sampler must return one hidden continuation")
        self._active_handle = handles[0]
        self._active_key = root.information_key

    def step(self, frame: CombatFrame, action: LegalAction,
             search_rng: random.Random) -> CombatFrame:
        del search_rng  # randomness was independently sampled at begin_simulation
        handle = self._active_handle
        if handle is None or self._active_key != frame.information_key:
            raise RuntimeError("Transition requested outside the active sampled path")
        policy = InformationPolicy(frame.observation.policy_id)
        actual = self._backend.observe(handle, policy)
        if actual.payload_json != frame.observation.payload_json:
            raise ValueError("Conditioned continuation disagrees with public observation")
        legal = tuple(self._backend.legal_actions(handle))
        if tuple(frame.legal_actions) != legal or action not in legal:
            raise ValueError("Conditioned continuation disagrees with legal public menu")
        transition = self._backend.step(handle, action)
        self._active_handle = transition.child
        if transition.child != handle:
            self._backend.release_many([handle])
        elif getattr(self._backend, "mutable_transition_capability_id", None) != (
                "h15-h8-inplace-player-step-v1"):
            raise ValueError("Backend returned same handle without H8 mutable-step capability")
        observed = self._backend.observe(transition.child, policy)
        public = json.loads(observed.payload_json)
        if not isinstance(public, dict):
            raise ValueError("Malformed poststep public observation")
        score = self._combat_result(public)
        child_legal = (
            () if score is not None
            else tuple(self._backend.legal_actions(transition.child))
        )
        # The complete public transcript is required only to condition the
        # independently sampled ROOT. During a rollout the active hidden
        # continuation persists; a hash of the preceding information-set key,
        # selected public action, and newly visible frame uniquely identifies
        # the child history without copying the entire run prefix.
        transition_public = [
            frame.information_key,
            [action.action_id, action.kind, action.payload_json],
            observed.policy_id,
            observed.payload_json,
            [[a.action_id, a.kind, a.payload_json] for a in child_legal],
        ]
        key = "h15-child-history:" + hashlib.sha256(
            json.dumps(transition_public, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._active_key = key
        return CombatFrame(key, observed, child_legal, score)

    def end_simulation(self) -> None:
        handle = self._active_handle
        self._active_handle = None
        self._active_key = None
        if handle is not None:
            self._backend.release_many([handle])


class H15RootConditionedCombatAdapter(H15PublicCombatAdapter):
    """PUCT with the fixed source's hidden realization, labelled as biased.

    Branch mechanics and ReferenceNature cursor are both cloned by H8's
    existing h8_fork operation. Only public frames reach policy/value
    inference. These counterfactuals are *not* independently sampled from
    the public-history posterior and must never be exported as fair targets.
    """
    fair_combat_capability_id = "h15-h8-root-conditioned-combat-v1"

    def __init__(self, backend: EmulatorBackend, root_handle: str):
        self._backend = backend
        self._sampler = None
        self._histories = {}
        self._active_handle = None
        self._active_key = None
        self._root_handle = root_handle

    def begin_simulation(self, root: CombatFrame, search_rng: random.Random) -> None:
        del search_rng
        if self._active_handle is not None:
            raise RuntimeError("Previous H8 root-conditioned fork was not released")
        if root.information_key not in self._histories:
            raise ValueError("Unregistered public root-conditioned decision")
        public = self._backend.observe(
            self._root_handle, InformationPolicy(root.observation.policy_id),
        )
        if (public.payload_json != root.observation.payload_json
                or tuple(self._backend.legal_actions(self._root_handle))
                != root.legal_actions):
            raise ValueError("H8 root-conditioned public frame changed")
        child = self._backend.fork_reference(self._root_handle)
        self._active_handle = child
        self._active_key = root.information_key

