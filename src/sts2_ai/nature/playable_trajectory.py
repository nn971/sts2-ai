"""Joint player/Nature smoke trajectories for v111 *legacy playable fixture*.

This does not run PPO, reinterpret a precommitted shuffle as deferred card
draws, or claim original-game chance probabilities. Exhausting the decision
budget is CENSORED, never a loss; unsupported sites propagate verbatim.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sts2_ai.emulator.legacy_playable_chance import (
    LegacyFrame,
    LegacyPlayableBridge,
)
from sts2_ai.nature.protocol import NaturePolicy, canonical_sha256


@dataclass(frozen=True, slots=True)
class LegacyDecision:
    kind: str
    parent_fingerprint: str
    identity: str
    chosen: str
    child_fingerprint: str


@dataclass(frozen=True, slots=True)
class LegacyPilot:
    schema: str
    emulator_revision: str
    seed: str
    nature_policy_id: str
    decisions: tuple[LegacyDecision, ...]
    initial_fingerprint: str
    final_fingerprint: str
    status: str
    final_transcript_sha256: str

    @property
    def fingerprint(self) -> str:
        from dataclasses import asdict
        return canonical_sha256(asdict(self))

    def replay_choices(self) -> tuple[tuple[str, str, str], ...]:
        return tuple((d.kind, d.identity, d.chosen) for d in self.decisions)


def collect_legacy_pilot(
    bridge: LegacyPlayableBridge, *, completed_intents_handle: str,
    seed: str, nature_policy: NaturePolicy,
    player_choose: Callable[[dict, tuple[str, ...]], str],
    max_decisions: int = 64,
) -> LegacyPilot:
    """Exercise real controlled_* mechanics until terminal or explicit cap.

    The caller owns and releases the completed isolated HP/intent handles.
    Handles originating here are released even on an unsupported RNG site.
    """
    if type(max_decisions) is not int or max_decisions < 1:
        raise ValueError("Positive decision budget required")
    frame = bridge.begin(completed_intents_handle=completed_intents_handle, seed=seed)
    initial = frame.fingerprint
    handles: list[LegacyFrame] = [frame]
    decisions: list[LegacyDecision] = []
    try:
        while not frame.terminal and len(decisions) < max_decisions:
            before = frame
            if frame.pending is not None:
                event = frame.pending
                proposal = nature_policy.propose(event.event)
                proposal.validate(event.event)
                frame = bridge.resolve(frame, proposal.selected_index)
                decision = LegacyDecision(
                    "nature", before.fingerprint, event.event_id,
                    str(proposal.selected_index), frame.fingerprint,
                )
            else:
                fair = bridge.observe(frame)
                action = player_choose(fair.observation, fair.action_ids)
                frame = bridge.step(frame, action, decision=fair)
                decision = LegacyDecision(
                    "player", before.fingerprint, fair.observation_hash,
                    action, frame.fingerprint,
                )
            decisions.append(decision)
            handles.append(frame)
        trace = bridge.trace(frame)
        return LegacyPilot(
            schema="sts2-ai-legacy-playable-pilot-v1",
            emulator_revision=bridge.emulator_revision,
            seed=seed, nature_policy_id=nature_policy.policy_id,
            decisions=tuple(decisions),
            initial_fingerprint=initial, final_fingerprint=frame.fingerprint,
            status="TERMINAL" if frame.terminal else "CENSORED_BOUNDARY_LIMIT",
            final_transcript_sha256=canonical_sha256(trace),
        )
    finally:
        bridge.release(*handles)


def replay_legacy_pilot(
    bridge: LegacyPlayableBridge, *, completed_intents_handle: str,
    pilot: LegacyPilot,
) -> LegacyPilot:
    """Replay exact selected outcomes and player actions from fresh origin."""
    # ChanceEvent.event_id is intentionally *different* from the emulator's
    # authoritative eventId. Replay by checking the wire-level IDs instead.
    script = list(pilot.decisions)
    frame = bridge.begin(
        completed_intents_handle=completed_intents_handle, seed=pilot.seed,
    )
    handles: list[LegacyFrame] = [frame]
    try:
        if frame.fingerprint != pilot.initial_fingerprint:
            raise ValueError("Controlled replay initial state diverged")
        for step in script:
            if frame.fingerprint != step.parent_fingerprint:
                raise ValueError("Controlled replay parent state diverged")
            if step.kind == "nature":
                if frame.pending is None or frame.pending.event_id != step.identity:
                    raise ValueError("Controlled replay chance identity diverged")
                frame = bridge.resolve(frame, int(step.chosen))
            elif step.kind == "player":
                fair = bridge.observe(frame)
                if fair.observation_hash != step.identity:
                    raise ValueError("Controlled replay public observation diverged")
                frame = bridge.step(frame, step.chosen, decision=fair)
            else:
                raise ValueError("Unknown controlled replay decision type")
            handles.append(frame)
            if frame.fingerprint != step.child_fingerprint:
                raise ValueError("Controlled replay mechanics diverged")
        if frame.fingerprint != pilot.final_fingerprint:
            raise ValueError("Controlled replay final fingerprint mismatch")
        expected_status = "TERMINAL" if frame.terminal else "CENSORED_BOUNDARY_LIMIT"
        if expected_status != pilot.status:
            raise ValueError("Controlled replay terminal/censor status diverged")
        if canonical_sha256(bridge.trace(frame)) != pilot.final_transcript_sha256:
            raise ValueError("Controlled replay event transcript diverged")
        return pilot
    finally:
        bridge.release(*handles)
