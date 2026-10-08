"""Adapt public-history rejection samples to single-player stochastic PUCT.

Each PUCT simulation starts from a *new* synthetic seed conditioned on the
observed root transcript. That hypothetical hidden state is advanced along
the entire simulation, preserving correlations among random events. Tree
selection receives only public observations and aggregated statistics.

This is exact only for the declared prototype seed ensemble; it has poor
acceptance for rare public histories. It is a correctness pilot rather than
an efficient native STS2 belief-state sampler.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from typing import Any, cast

from sts2_ai.emulator.chance import PublicHistoryStep
from sts2_ai.emulator.protocol import EmulatorBackend, InformationPolicy, LegalAction
from sts2_ai.emulator.rejection import FairHistoryRejectionSampler

from .puct import (
    FAIR_TRANSITION_CAPABILITY_ID,
    PublicSearchNode,
    TerminalReturn,
)


class FairReplayPuctAdapter:
    """Replay-conditioned PUCT transitions without using the live run seed."""

    fair_transition_capability_id = FAIR_TRANSITION_CAPABILITY_ID

    def __init__(
        self,
        backend: EmulatorBackend,
        *,
        goal: str = "prototype-full-victory-v1",
        max_candidates: int = 256,
    ) -> None:
        if goal not in {"prototype-full-victory-v1", "prototype-act1-clear-v1"}:
            raise ValueError("Unsupported or unversioned terminal objective")
        self._backend = backend
        self._goal = goal
        self._sampler = FairHistoryRejectionSampler(
            backend, max_candidates=max_candidates
        )
        self._histories: dict[str, tuple[PublicHistoryStep, ...]] = {}
        self._active_handle: str | None = None
        self._active_key: str | None = None
        self.transition_count = 0

    @property
    def sampler(self) -> FairHistoryRejectionSampler:
        return self._sampler

    @staticmethod
    def _key(history: Sequence[PublicHistoryStep]) -> str:
        # Full observation/action history prevents merging distinct information
        # sets with the same current screenshot and different observed reveals.
        rows = [
            (
                step.observation.policy_id,
                step.observation.payload_json,
                None if step.chosen_action is None else (
                    step.chosen_action.action_id,
                    step.chosen_action.kind,
                    step.chosen_action.payload_json,
                ),
            )
            for step in history
        ]
        canonical = json.dumps(rows, ensure_ascii=True, separators=(",", ":"))
        return "public-history-v1:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _visible_outcome(observation_json: str, goal: str) -> TerminalReturn | None:
        raw = json.loads(observation_json)
        if not isinstance(raw, dict):
            raise ValueError("Public observation must be an object")
        state = cast(dict[str, Any], raw)
        outcome = state.get("terminal_outcome")
        if outcome not in (None, "victory", "defeat"):
            raise ValueError("Cannot label an unknown terminal outcome")
        act = state.get("act")
        completed_act1 = isinstance(act, int) and act >= 2
        if goal == "prototype-act1-clear-v1" and (completed_act1 or outcome == "victory"):
            success = True
        elif outcome == "defeat":
            success = False
        elif goal == "prototype-full-victory-v1" and outcome == "victory":
            success = True
        else:
            return None
        hp = state.get("hp")
        max_hp = state.get("max_hp")
        hp_fraction = (
            max(0.0, min(1.0, hp / max_hp))
            if isinstance(hp, int) and isinstance(max_hp, int) and max_hp > 0
            else None
        )
        # Progress targets require validated, episode-specific final bounds.
        return TerminalReturn(success=success, hp=hp_fraction)

    def root(
        self,
        history: Sequence[PublicHistoryStep],
        legal_actions: Sequence[LegalAction],
    ) -> PublicSearchNode:
        self._sampler._validate_history(history)
        transcript = tuple(history)
        key = self._key(transcript)
        terminal = self._visible_outcome(transcript[-1].observation.payload_json, self._goal)
        if terminal is not None:
            if legal_actions:
                raise ValueError("Declared goal terminal must have no search actions")
            legal = ()
        else:
            legal = tuple(legal_actions)
        prior = self._histories.get(key)
        if prior is not None and prior != transcript:
            raise ValueError("Public history key collision")
        self._histories[key] = transcript
        return PublicSearchNode(key, transcript[-1].observation, legal, terminal)

    def begin_simulation(self, root: PublicSearchNode, search_rng: random.Random) -> None:
        if self._active_handle is not None:
            raise RuntimeError("Previous simulated hidden state has not been released")
        transcript = self._histories.get(root.information_key)
        if transcript is None or transcript[-1].observation != root.observation:
            raise ValueError("Search root must belong to the registered public history")
        samples = self._sampler.sample_fair_continuations(
            transcript, search_rng=search_rng, count=1,
        )
        self._active_handle = samples[0]
        self._active_key = root.information_key

    def end_simulation(self) -> None:
        handle = self._active_handle
        self._active_handle = None
        self._active_key = None
        if handle is not None:
            self._backend.release_many([handle])

    def sample_transition(
        self,
        node: PublicSearchNode,
        action: LegalAction,
        search_rng: random.Random,
    ) -> PublicSearchNode:
        # The sampled latent continuation already has independent search RNG
        # from begin_simulation; keep it throughout this simulation to preserve
        # correlations between all later random outcomes.
        del search_rng
        state = self._active_handle
        if state is None or self._active_key != node.information_key:
            raise RuntimeError("Transition requested outside its sampled public path")
        transcript = self._histories.get(node.information_key)
        if transcript is None:
            raise ValueError("Unknown public information key")
        policy = InformationPolicy(node.observation.policy_id)
        actual = self._backend.observe(state, policy)
        if actual.payload_json != node.observation.payload_json:
            raise ValueError("Conditioned state no longer matches its public history")
        legal = self._backend.legal_actions(state)
        matched = next((
            option for option in legal
            if option.action_id == action.action_id
            and option.kind == action.kind
            and option.payload_json == action.payload_json
        ), None)
        if matched is None:
            raise ValueError("Search action is illegal in conditioned hidden state")
        transition = self._backend.step(state, matched)
        self._active_handle = transition.child
        self._backend.release_many([state])
        self.transition_count += 1
        observed = self._backend.observe(transition.child, policy)
        expanded = transcript[:-1] + (
            PublicHistoryStep(transcript[-1].observation, action),
            PublicHistoryStep(observed, None),
        )
        terminal = self._visible_outcome(observed.payload_json, self._goal)
        actions = () if terminal is not None else tuple(self._backend.legal_actions(transition.child))
        key = self._key(expanded)
        prior = self._histories.get(key)
        if prior is not None and prior != expanded:
            raise ValueError("Public history key collision")
        self._histories[key] = expanded
        self._active_key = key
        return PublicSearchNode(key, observed, actions, terminal)
