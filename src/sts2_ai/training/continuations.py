"""Independent exact-state continuation labels for sampled MCTS cutoff states.

These records measure the outcome of a specified *continuation policy* from
forked hidden states, never the value model used by MCTS. They are explicitly
oracle-exact data, and truncations remain censored.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

from sts2_ai.emulator import EmulatorBackend, InformationPolicy, LegalAction, Observation
from sts2_ai.emulator.protocol import StateHandle
from sts2_ai.search.mcts import sts2_value

SCHEMA = "sts2-ai-cutoff-continuation-v1"


class _Decision(Protocol):
    @property
    def action(self) -> LegalAction: ...


class ContinuationPolicy(Protocol):
    def choose(
        self, observation: Observation, legal_actions: Sequence[LegalAction]
    ) -> _Decision: ...


@dataclass(frozen=True, slots=True)
class ContinuationRecord:
    observation_hash: str
    observation_json: str
    information_policy: str
    source_run_seed: str
    source_exact_hash: str
    cutoff_reason: str
    continuation_policy_id: str
    continuation_max_decisions: int
    continuation_decisions: int
    outcome: str
    # A genuine terminal win/loss is ±1. Censored continuations have no label.
    terminal_value: float | None
    terminal_act: int | None
    terminal_floor: int | None


class CutoffContinuationCollector:
    """Sample nonterminal MCTS cutoffs and fork each exact state for a rollout.

    We select before hashing/forking. Sampling, state exact-hash deduplication,
    and the continuation policy use no MCTS RNG. Thus this observer consumes
    extra compute without changing the tree's decisions. The continuation
    remains deterministic for the given hidden state and policy.
    """

    def __init__(
        self,
        backend: EmulatorBackend,
        *,
        policy: InformationPolicy,
        continuation_policy: ContinuationPolicy,
        every: int = 64,
        max_unique: int = 64,
        max_per_seed: int | None = None,
        max_decisions: int = 512,
    ) -> None:
        if min(every, max_unique, max_decisions) <= 0 or (
            max_per_seed is not None and max_per_seed <= 0
        ):
            raise ValueError("Sampling and continuation limits must be positive")
        self.backend = backend
        self.policy = policy
        self.continuation_policy = continuation_policy
        self.continuation_policy_id = getattr(
            continuation_policy,
            "policy_id",
            f"{type(continuation_policy).__module__}."
            f"{type(continuation_policy).__qualname__}",
        )
        self.every = every
        self.max_unique = max_unique
        self.max_per_seed = max_per_seed
        self.max_decisions = max_decisions
        self.seen = 0
        self.sampled = 0
        self._records: dict[tuple[str, str], ContinuationRecord] = {}

    def record(
        self,
        observation: Observation,
        handle: StateHandle,
        reason: str,
        *,
        run_seed: str,
    ) -> None:
        if observation.policy_id != self.policy.policy_id:
            raise ValueError("Continuation observation policy mismatch")
        if reason not in {"depth", "combat-exit", "no-actions"}:
            raise ValueError(f"Unknown cutoff reason {reason!r}")
        self.seen += 1
        if self.seen % self.every != 0 or len(self._records) >= self.max_unique:
            return
        if self.max_per_seed is not None and sum(
            seed == run_seed for seed, _ in self._records
        ) >= self.max_per_seed:
            return
        self.sampled += 1
        exact_hash = self.backend.exact_hash(handle)
        key = (run_seed, exact_hash)
        if key in self._records:
            previous = self._records[key]
            if previous.observation_hash != observation.observation_hash:
                raise RuntimeError("One exact state yielded conflicting observations")
            return
        continuation = self._continue_fork(handle, observation)
        raw = json.loads(continuation[0].payload_json)
        if not isinstance(raw, dict):
            raise ValueError("Terminal continuation observation must be an object")
        outcome, decisions = continuation[1], continuation[2]
        act = raw.get("act")
        floor = raw.get("floor")
        self._records[key] = ContinuationRecord(
            observation_hash=observation.observation_hash,
            observation_json=observation.payload_json,
            information_policy=observation.policy_id,
            source_run_seed=run_seed,
            source_exact_hash=exact_hash,
            cutoff_reason=reason,
            continuation_policy_id=self.continuation_policy_id,
            continuation_max_decisions=self.max_decisions,
            continuation_decisions=decisions,
            outcome=outcome,
            terminal_value=(
                1.0 if outcome == "victory" else
                -1.0 if outcome == "defeat" else None
            ),
            terminal_act=act if type(act) is int else None,
            terminal_floor=floor if type(floor) is int else None,
        )

    def _continue_fork(
        self, original: StateHandle, observation: Observation
    ) -> tuple[Observation, str, int]:
        current = self.backend.fork(original)
        latest = observation
        decisions = 0
        try:
            if self.backend.is_terminal(current):
                raise RuntimeError("A terminal state reached the nonterminal cutoff observer")
            actions = tuple(self.backend.legal_actions(current))
            while decisions < self.max_decisions:
                if not actions:
                    return latest, "stuck", decisions
                chosen = self.continuation_policy.choose(latest, actions).action
                if chosen.action_id not in {a.action_id for a in actions}:
                    raise RuntimeError("Continuation policy chose an illegal action")
                (frame,) = self.backend.batch_rollout_step_frame(
                    ((current, chosen),), self.policy
                )
                previous = current
                current = frame.transition.child
                if current != previous:
                    self.backend.release_many((previous,))
                decisions += 1
                latest = frame.observation
                if frame.transition.terminal:
                    terminal = json.loads(latest.payload_json)
                    if not isinstance(terminal, dict):
                        raise ValueError("Terminal observation must be an object")
                    outcome = terminal.get("terminal_outcome")
                    if outcome not in {"victory", "defeat"}:
                        raise RuntimeError(
                            f"Terminal continuation has unknown outcome: {outcome!r}"
                        )
                    return latest, outcome, decisions
                actions = frame.legal_actions
            return latest, "truncated", decisions
        finally:
            self.backend.release_many((current,))

    @property
    def records(self) -> tuple[ContinuationRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def write_jsonl(self, path: Path, *, provenance: dict[str, Any]) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for record in self.records:
                payload = {
                    "schema": SCHEMA,
                    **asdict(record),
                    "sampling_every": self.every,
                    "provenance": provenance,
                }
                stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        return len(self._records)


def load_continuations(path: Path) -> tuple[ContinuationRecord, ...]:
    records: list[ContinuationRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
                raise ValueError(f"Invalid continuation schema on line {line_number}")
            fields = {
                field: raw[field]
                for field in ContinuationRecord.__dataclass_fields__
            }
            record = ContinuationRecord(**fields)
            expected = {"victory": 1.0, "defeat": -1.0}
            if record.outcome not in {"victory", "defeat", "truncated", "stuck"}:
                raise ValueError("Unknown continuation outcome")
            if record.terminal_value != expected.get(record.outcome):
                raise ValueError("Censored or incorrect continuation target")
            if not 0 <= record.continuation_decisions <= record.continuation_max_decisions:
                raise ValueError("Invalid continuation decision count")
            if record.continuation_max_decisions < 1:
                raise ValueError("Invalid continuation horizon")
            records.append(record)
    return tuple(records)


def continuation_report(
    records: tuple[ContinuationRecord, ...],
    *,
    model: Any = None,
) -> dict[str, Any]:
    """Held-out-style descriptive RMSE against observed heuristic outcomes.

    The input itself is not an independent holdout; users must supply records
    generated on unseen run seeds when evaluating a trained model.
    """
    if not records:
        raise ValueError("Continuation dataset is empty")
    labeled = tuple(record for record in records if record.terminal_value is not None)
    report: dict[str, Any] = {
        "records": len(records),
        "terminal_labels": len(labeled),
        "truncated": sum(r.outcome == "truncated" for r in records),
        "stuck": sum(r.outcome == "stuck" for r in records),
        "victories": sum(r.outcome == "victory" for r in records),
        "defeats": sum(r.outcome == "defeat" for r in records),
        "mean_continuation_decisions": fmean(
            r.continuation_decisions for r in records
        ),
        "source_run_seeds": len({r.source_run_seed for r in records}),
        "warning": (
            "Labels are deterministic outcomes under the named continuation policy "
            "from oracle-exact hidden states. Censored records are excluded from RMSE; "
            "a policy with all defeats gives weak value-training supervision."
        ),
    }
    if labeled:
        manual_errors = []
        model_errors = []
        for record in labeled:
            observation = Observation(
                policy_id=record.information_policy,
                payload_json=record.observation_json,
                observation_hash=record.observation_hash,
            )
            assert record.terminal_value is not None
            manual_errors.append((sts2_value(observation) - record.terminal_value) ** 2)
            if model is not None:
                prediction = model.evaluate(observation, ()).value
                if not math.isfinite(prediction):
                    raise ValueError("Model produced nonfinite prediction")
                model_errors.append((prediction - record.terminal_value) ** 2)
        report["handcrafted_terminal_rmse"] = math.sqrt(fmean(manual_errors))
        if model is not None:
            report["learned_terminal_rmse"] = math.sqrt(fmean(model_errors))
            report["model_id"] = model.model_id
    else:
        report["handcrafted_terminal_rmse"] = None
        if model is not None:
            report["learned_terminal_rmse"] = None
            report["model_id"] = model.model_id
    return report
