"""End-to-end joint decisions for the legacy controlled *playable fixture*.

This pilot intentionally uses primitive indexed tickets. It cannot certify
native conditional probabilities or replace semantic deferred-draw Nature.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sts2_ai.emulator.controlled_chance import IsolatedNatureBridge
from sts2_ai.emulator.playable_chance import FairPlayerFrame, PlayableNatureBridge
from sts2_ai.nature.protocol import NaturePolicy, canonical_sha256


@dataclass(frozen=True, slots=True)
class PlayableDecision:
    kind: str
    fingerprint_before: str
    identity: str
    selection: str
    fingerprint_after: str


@dataclass(frozen=True, slots=True)
class PlayableTrajectory:
    emulator_revision: str
    nature_policy_id: str
    fixture_id: str
    decisions: tuple[PlayableDecision, ...]
    terminal_fingerprint: str

    @property
    def replay_id(self) -> str:
        return canonical_sha256([
            self.emulator_revision, self.fixture_id,
            [(x.kind, x.fingerprint_before, x.identity,
              x.selection, x.fingerprint_after) for x in self.decisions],
            self.terminal_fingerprint,
        ])


def play_controlled_fixture(
    isolated: IsolatedNatureBridge,
    playable: PlayableNatureBridge,
    nature: NaturePolicy,
    *,
    seed: str,
    act: int,
    encounter_id: str,
    native_entry: str,
    total_floor: int,
    player_choose: Callable[[FairPlayerFrame], str],
    max_boundaries: int = 512,
    on_decision: Callable[[PlayableDecision], None] | None = None,
) -> PlayableTrajectory:
    """Use the actual HP->intent continuation and mechanically execute actions.

    Every decision is externally selected. Source fixture restrictions and
    unsupported RNG sites propagate as explicit errors; decision budgets
    raise rather than producing a synthetic defeat label.
    """
    if max_boundaries <= 0:
        raise ValueError("Positive boundary limit required")
    if isolated.emulator_revision != playable.emulator_revision:
        raise ValueError("Nature bridges must use identical emulator revision")
    hp = isolated.begin(
        kind="initial_hp", seed=seed, act=act, encounter_id=encounter_id,
        native_entry=native_entry, total_floor=total_floor,
    )
    isolated_handles = [hp]
    playable_handles = []
    decisions: list[PlayableDecision] = []

    def append(kind: str, before: str, identity: str, choice: str, after: str) -> None:
        if len(decisions) >= max_boundaries:
            raise RuntimeError("Controlled fixture boundary budget exhausted: censored")
        row = PlayableDecision(kind, before, identity, choice, after)
        decisions.append(row)
        if on_decision is not None:
            on_decision(row)

    try:
        state = hp
        while True:
            while not state.complete:
                if len(decisions) >= max_boundaries:
                    raise RuntimeError("Controlled fixture boundary budget exhausted: censored")
                if state.pending is None:
                    raise RuntimeError("Missing initial chance event")
                event = state.pending
                proposal = nature.propose(event)
                proposal.validate(event)
                child = isolated.resolve(state, proposal.selected_index)
                append("nature", state.fingerprint, event.event_id,
                       str(proposal.selected_index), child.fingerprint)
                state = child
                isolated_handles.append(state)
            if state.kind == "intents":
                break
            state = isolated.advance(state)
            isolated_handles.append(state)

        frame = playable.begin(state, seed=seed)
        playable_handles.append(frame)
        while frame.phase != "terminal":
            if len(decisions) >= max_boundaries:
                raise RuntimeError("Controlled fixture boundary budget exhausted: censored")
            if frame.phase == "chance":
                assert frame.event is not None and frame.native_event_id is not None
                event = frame.event
                proposal = nature.propose(event)
                proposal.validate(event)
                child = playable.resolve(frame, proposal.selected_index)
                append("nature", frame.fingerprint, event.event_id,
                       str(proposal.selected_index), child.fingerprint)
            else:
                player_frame = playable.observe(frame)
                action = player_choose(player_frame)
                child = playable.step(frame, action)
                append("player", frame.fingerprint,
                       canonical_sha256([player_frame.observation_hash,
                                         player_frame.action_ids]),
                       action, child.fingerprint)
            frame = child
            playable_handles.append(frame)
        return PlayableTrajectory(
            isolated.emulator_revision, nature.policy_id,
            f"{seed}:{act}:{encounter_id}:{native_entry}:{total_floor}",
            tuple(decisions), frame.fingerprint,
        )
    finally:
        playable.release(*playable_handles)
        isolated.release(*isolated_handles)
