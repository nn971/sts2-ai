"""Joint player/Nature semantic trajectory protocol and strict replay.

No game state or deck mutation is performed here: an emulator-owned,
versioned bridge must author all support, constraints and transitions.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from sts2_ai.emulator.semantic_chance import SemanticNatureBridge

from .protocol import canonical_sha256
from .semantic import PROPOSED_WIRE_SCHEMA, SemanticEvent, SemanticProposal

SEMANTIC_TRAJECTORY_SCHEMA = "sts2-ai-semantic-trajectory-v1"


class SemanticPolicy(Protocol):
    policy_id: str

    def propose(self, event: SemanticEvent) -> SemanticProposal: ...


@dataclass(frozen=True, slots=True)
class SemanticDecision:
    kind: str
    parent_fingerprint: str
    identity: str
    selected_id: str
    result_fingerprint: str
    log_q: float | None

    def __post_init__(self) -> None:
        if self.kind not in ("player", "nature"):
            raise ValueError("Unknown joint semantic decision kind")
        if not all((self.parent_fingerprint, self.identity,
                    self.selected_id, self.result_fingerprint)):
            raise ValueError("Incomplete semantic decision record")


@dataclass(frozen=True, slots=True)
class SemanticTrajectory:
    emulator_revision: str
    fixture_id: str
    policy_id: str
    decisions: tuple[SemanticDecision, ...]
    terminal_fingerprint: str
    terminal_outcome: str
    schema: str = SEMANTIC_TRAJECTORY_SCHEMA
    chance_schema: str = PROPOSED_WIRE_SCHEMA

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(asdict(self))

    def nature_script(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (step.identity, step.selected_id)
            for step in self.decisions if step.kind == "nature"
        )

    def player_script(self) -> tuple[str, ...]:
        return tuple(step.selected_id for step in self.decisions if step.kind == "player")

    def mechanical_signature(self) -> tuple[tuple[str, str, str, str, str], ...]:
        """Compare scripted replay across differently named Nature policies."""
        return tuple(
            (step.kind, step.parent_fingerprint, step.identity,
             step.selected_id, step.result_fingerprint)
            for step in self.decisions
        )


def collect_semantic_trajectory(
    bridge: SemanticNatureBridge, policy: SemanticPolicy, *,
    fixture_id: str, seed: str,
    player_choose: Callable[[dict[str, Any], tuple[str, ...]], str],
    max_boundaries: int = 512,
) -> SemanticTrajectory:
    if max_boundaries <= 0:
        raise ValueError("Positive semantic boundary budget required")
    frame = bridge.begin(fixture_id=fixture_id, seed=seed)
    handles = [frame]
    rows: list[SemanticDecision] = []
    seen_identity: dict[str, set[str]] = {}
    try:
        while frame.phase != "terminal":
            if len(rows) >= max_boundaries:
                raise RuntimeError("Semantic trajectory boundary limit reached")
            prior = frame.fingerprint
            if frame.phase == "chance":
                assert frame.event is not None
                event = frame.event
                proposal = policy.propose(event)
                proposal.validate(event)
                # Duplicate selected *physical* instances within a single
                # deferred shuffle or without-replacement pool are invalid.
                if event.operation in ("draw_card", "reveal_card", "select_distinct"):
                    used = seen_identity.setdefault(event.randomization_id, set())
                    if proposal.selected_id in used:
                        raise ValueError(
                            "Repeated semantic instance within unresolved randomization"
                        )
                    used.add(proposal.selected_id)
                child = bridge.resolve(frame, proposal)
                row = SemanticDecision(
                    "nature", prior, event.replay_identity, proposal.selected_id,
                    child.fingerprint, proposal.log_q,
                )
            else:
                assert frame.phase == "player" and frame.observation is not None
                action_id = player_choose(frame.observation, frame.legal_actions)
                child = bridge.step(frame, action_id)
                row = SemanticDecision(
                    "player", prior,
                    canonical_sha256([frame.observation, frame.legal_actions]),
                    action_id, child.fingerprint, None,
                )
            rows.append(row)
            frame = child
            handles.append(frame)
        assert frame.terminal is not None
        return SemanticTrajectory(
            emulator_revision=bridge.emulator_revision,
            fixture_id=fixture_id, policy_id=policy.policy_id,
            decisions=tuple(rows),
            terminal_fingerprint=frame.fingerprint,
            terminal_outcome=frame.terminal,
        )
    finally:
        bridge.release(*handles)
