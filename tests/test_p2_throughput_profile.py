"""P2.0 diagnostic profiling contract: no teacher or stochastic drift."""
from __future__ import annotations

import ast
import io
import json
import random
from pathlib import Path

import pytest

from sts2_ai.emulator.jsonl_backend import JsonlEmulatorBackend
from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.training.h16_public_codec import encode_h16_decision
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


def test_p2_native_profiler_protocol_and_wire_profile_toggle():
    backend = make_backend(profile_wire=False)
    response = [
        {
            "requestId": "1", "ok": True,
            "schemaId": "sts2-p2-jsonl-performance-v1",
        },
        {
            "requestId": "2", "ok": True,
            "profile": {
                "schemaId": "sts2-p2-jsonl-performance-v1",
                "enabled": False,
                "operations": {
                    "p2.fork.run_state": {
                        "count": 3, "totalSeconds": 0.1,
                        "allocatedBytes": 2048,
                    },
                },
            },
            "liveStateHandles": 0, "peakStateHandles": 4,
        },
    ]
    backend._process.stdout = io.StringIO(
        "".join(json.dumps(row) + "\n" for row in response),
    )
    backend.p2_native_profile_reset()
    stats = backend.p2_native_profile_snapshot(stop=True)
    assert stats["live_state_handles"] == 0
    assert stats["peak_state_handles"] == 4
    assert stats["profile"]["operations"]["p2.fork.run_state"]["count"] == 3
    assert backend.operation_profile()["p2_perf_stop"].calls == 1
    assert backend.wire_operation_profile() == {}
    backend.set_p2_wire_profile(True)
    assert backend.wire_operation_profile() == {}


def test_p2_native_profile_rejects_malformed_handle_counts():
    backend = make_backend(profile_wire=False)
    backend._process.stdout = io.StringIO(json.dumps({
        "requestId": "1", "ok": True,
        "profile": {
            "schemaId": "sts2-p2-jsonl-performance-v1",
            "operations": {},
        },
        "liveStateHandles": 5, "peakStateHandles": 4,
    }) + "\n")
    with pytest.raises(Exception, match="Malformed C# P2"):
        backend.p2_native_profile_snapshot(stop=True)


def test_generated_map_probe_never_enables_fixed_route_only_compact_selection():
    """Lightweight source contract; the profiler imports torch only at runtime.

    A previously inverted `not fixed_route` passed compact selection to the
    generated-map reset. The pinned H16 emulator correctly rejects that flag.
    """
    profiler = Path(__file__).resolve().parents[1] / "tools/profile_p2_joint_throughput.py"
    module = ast.parse(profiler.read_text(encoding="utf-8"))
    runner = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_episode"
    )
    episode_call = next(
        node for node in ast.walk(runner)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "collect_act_episode"
    )
    compact = next(
        keyword.value for keyword in episode_call.keywords
        if keyword.arg == "compact_card_selection"
    )
    assert isinstance(compact, ast.Constant)
    assert compact.value is False


def test_generated_map_draw_pile_selection_uses_public_prompt_candidates():
    """H16 must resolve cards revealed in a selection dialog, not hidden draw order."""
    public = {
        "hp": 70, "max_hp": 70, "gold": 0, "deck": [],
        "combat": {
            "hand": [], "discard_pile": [], "exhaust_pile": [],
            "draw_pile_count": 3, "pending_choice_id": "select_cards",
            "selection_candidates": [
                {
                    "instance_id": 15, "card_id": "proto.silent.strike",
                    "upgrade_level": 0, "is_temporary": False, "state": {},
                },
            ],
        },
    }
    action = LegalAction("choose-15", "select_cards", '{"CardInstanceIds":[15]}')
    encoded = encode_h16_decision(public, (action,))
    assert len(encoded.actions) == 1
    (member,) = encoded.actions[0].members
    assert encoded.objects[member].zone == 10  # public reward/choice zone
    assert member >= 0
    # The dialog is the disclosure boundary: without a candidate record,
    # the action must fail rather than silently alias another card.
    public["combat"]["selection_candidates"] = []
    with pytest.raises(ValueError, match="candidate missing"):
        encode_h16_decision(public, (action,))
