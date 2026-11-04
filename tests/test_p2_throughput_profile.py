"""P2.0 diagnostic profiling contract: no teacher or stochastic drift."""
from __future__ import annotations

import io
import json
import random

import pytest

from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.emulator.jsonl_backend import JsonlEmulatorBackend
from sts2_ai.search.combat_puct import (
    FAIR_COMBAT_CAPABILITY,
    H16_EXPERIMENTAL_ROOT_CONDITIONED_CAPABILITY,
    CombatFrame,
    CombatPuct,
    CombatSearchProfile,
)

A = LegalAction("a", "end_turn")
B = LegalAction("b", "end_turn")
ADVANCE = LegalAction("advance", "end_turn")


class FiniteSampler:
    fair_combat_capability_id = H16_EXPERIMENTAL_ROOT_CONDITIONED_CAPABILITY

    def __init__(self, length: int = 4) -> None:
        self.length = length
        self.begun = 0
        self.ended = 0

    def frame(self, n: int, branch: str = "root") -> CombatFrame:
        return CombatFrame(
            information_key=f"{branch}/{n}",
            observation=Observation("prototype-fair-v0", "{}", f"{branch}-{n}"),
            legal_actions=(
                () if n == self.length
                else (A, B) if n == 0 else (ADVANCE,)
            ),
            terminal_reward=2.0 if n == self.length else None,
        )

    def begin_simulation(self, frame: CombatFrame, rng: random.Random) -> None:
        self.begun += 1

    def step(self, frame: CombatFrame, action: LegalAction,
             rng: random.Random) -> CombatFrame:
        branch, depth = frame.information_key.rsplit("/", 1)
        return self.frame(int(depth) + 1, action.action_id if depth == "0" else branch)

    def end_simulation(self) -> None:
        self.ended += 1


def evaluator(obs: Observation, actions: tuple[LegalAction, ...]):
    return tuple(0.0 for _ in actions), 0.25


@pytest.mark.parametrize("budget", [8, 16, 64])
@pytest.mark.parametrize("depth", [2, None])
def test_profiled_search_is_identical_to_unprofiled(budget: int, depth: int | None):
    control = FiniteSampler()
    measured = FiniteSampler()
    settings = dict(
        seed=121, max_depth=depth, tree_expansion_mode="full_depth",
        max_simulation_steps=32,
    )
    plain = CombatPuct(control, evaluator, **settings).search(
        control.frame(0), simulations=budget,
    )
    counters = CombatSearchProfile()
    traced = CombatPuct(measured, evaluator, **settings).search(
        measured.frame(0), simulations=budget, profile=counters,
    )
    assert traced == plain
    assert plain.policy == traced.policy
    assert measured.begun == measured.ended == budget
    assert counters.begin_calls == counters.end_calls == budget
    assert counters.tree_step_calls == traced.transitions
    assert len(counters.simulation_path_lengths) == budget
    assert sum(counters.simulation_path_lengths) == traced.transitions
    assert counters.repeated_public_edges > 0
    data = counters.snapshot(traced)
    assert data["root_visits"] == list(plain.visits)
    assert data["teacher_policy"] == list(plain.policy)
    assert data["chance_capability"] == H16_EXPERIMENTAL_ROOT_CONDITIONED_CAPABILITY
    assert data["public_replay_hit_fraction"] > 0
    json.dumps(data)  # JSON-safe report


def test_profiled_fair_sampler_cannot_certify_exact_state_reuse():
    sampler = FiniteSampler()
    sampler.fair_combat_capability_id = FAIR_COMBAT_CAPABILITY
    stats = CombatSearchProfile()
    result = CombatPuct(sampler, evaluator, seed=3).search(
        sampler.frame(0), simulations=8, profile=stats,
    )
    assert stats.snapshot(result)["chance_capability"] == FAIR_COMBAT_CAPABILITY
    assert "cannot certify" in stats.snapshot(result)["note"]


def test_profiled_search_releases_simulation_after_sampler_exception():
    class BrokenSampler(FiniteSampler):
        def step(self, frame: CombatFrame, action: LegalAction,
                 rng: random.Random) -> CombatFrame:
            raise RuntimeError("forced exception")

    sampler = BrokenSampler()
    stats = CombatSearchProfile()
    with pytest.raises(RuntimeError, match="forced exception"):
        CombatPuct(sampler, evaluator).search(
            sampler.frame(0), simulations=1, profile=stats,
        )
    assert sampler.begun == sampler.ended == 1
    assert stats.begin_calls == stats.end_calls == stats.tree_step_calls == 1


class _FakeProcess:
    def __init__(self, response: str) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(response)
        self.stderr = io.StringIO()
        self.poll = lambda: None


def make_backend(*, profile_wire: bool):
    """In-memory JSONL transport; tests framing independently of .NET."""
    backend = object.__new__(JsonlEmulatorBackend)
    backend._process = _FakeProcess(
        json.dumps({"requestId": "1", "ok": True, "message": "μ"},
                   ensure_ascii=False) + "\n",
    )
    backend._next_request_id = 1
    backend._closed = False
    backend._operation_profile = {}
    backend._profile_wire = profile_wire
    backend._wire_profile = {}
    return backend


@pytest.mark.parametrize("instrumented", [False, True])
def test_bridge_wire_profile_preserves_jsonl_output(instrumented: bool):
    backend = make_backend(profile_wire=instrumented)
    result = backend._request("probe", data={"key": "μ"})
    assert result["message"] == "μ"
    assert backend._next_request_id == 2
    request = json.loads(backend._process.stdin.getvalue())
    assert request == {"request_id": "1", "op": "probe", "data": {"key": "μ"}}
    assert backend.operation_profile()["probe"].calls == 1
    if instrumented:
        stat = backend.wire_operation_profile()["probe"]
        assert stat.calls == 1
        assert stat.request_bytes == len(backend._process.stdin.getvalue().encode())
        assert stat.response_bytes == len(
            (json.dumps({"requestId": "1", "ok": True, "message": "μ"},
                        ensure_ascii=False) + "\n").encode()
        )
        assert stat.encode_seconds >= 0
        assert stat.pipe_seconds >= 0
        assert stat.decode_seconds >= 0
    else:
        assert backend.wire_operation_profile() == {}
    backend.reset_operation_profile()
    assert backend.operation_profile() == {}
    assert backend.wire_operation_profile() == {}
