"""P2.1 bounded, private-receipt-certified Native prefix retention (CPU-only)."""
from __future__ import annotations

import hashlib

import pytest

from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.search.combat_puct import (
    H16_EXPERIMENTAL_ROOT_CONDITIONED_CAPABILITY,
    CombatFrame,
)
from sts2_ai.search.p2_exact_prefix_cache import ExactRootPrefixCache


class StubBackend:
    def __init__(self):
        self.hidden = {
            "root": ("state-root", "rng-0"),
            "c1": ("state-1", "rng-1"),
            "c2": ("state-2", "rng-2"),
        }
        self.present_flags = {}
        self.released = []

    def exact_continuation_receipt(self, handle):
        public, private = self.hidden[handle]
        def h(text):
            return hashlib.sha256(text.encode()).hexdigest()
        return {
            "exactHash": h(public),
            "runStochasticHash": h(private),
            "worldStochasticHash": h("world-0"),
            "combatNativeRngHash": None,
            "referenceNaturePresent": self.present_flags.get(handle) == "nature",
            "semanticOtherChancePresent": self.present_flags.get(handle) == "other",
            "semanticRewardAuditPresent": self.present_flags.get(handle) == "reward",
        }

    def release_many(self, handles):
        for handle in handles:
            assert handle in self.hidden
            assert handle != "root"
            del self.hidden[handle]
            self.released.append(handle)
        return len(handles)


def cache(backend, *, max_handles=2):
    return ExactRootPrefixCache(
        backend, "root", max_handles=max_handles,
        capability=H16_EXPERIMENTAL_ROOT_CONDITIONED_CAPABILITY,
    )


def frames():
    start = CombatFrame(
        "root-key", Observation("prototype-fair-v0", "{}", "hash-root"),
        (LegalAction("a", "play_card"), LegalAction("b", "end_turn")),
    )
    child = CombatFrame(
        "root-key/a", Observation("prototype-fair-v0", "{}", "hash-child"),
        (LegalAction("end", "end_turn"),),
    )
    end = CombatFrame(
        "root-key/a/end",
        Observation("prototype-fair-v0", "{}", "hash-end"),
        (), 1.0,
    )
    return start, child, end


def test_repeated_action_path_reuses_only_exact_receipt_and_full_history():
    backend = StubBackend()
    c = cache(backend)
    parent, child, final = frames()
    initial = [{"combat": {"turn": 1}, "hp": 70}]
    after = [*initial, {"combat": {"turn": 1}, "hp": 69}]
    c.begin_simulation()
    assert c.lookup(parent, "a", initial, "root") is None
    assert c.record(parent, "a", child, "c1", initial, after, "root")
    assert c.lookup(child, "end", after, "c1") is None
    assert c.record(child, "end", final, "c2", after, after, "c1")
    c.begin_simulation()
    entry = c.lookup(parent, "a", initial, "root")
    assert entry is not None and entry.handle == "c1"
    assert list(entry.child_history) == after
    assert c.lookup(child, "end", after, "c1").handle == "c2"
    assert c.metrics()["hits"] == 2
    assert c.metrics()["retained_handles"] == 2
    c.close()
    c.close()
    assert backend.released == ["c1", "c2"]
    assert set(backend.hidden) == {"root"}
    assert c.metrics()["retained_handles"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        c.begin_simulation()


def test_distinct_action_path_and_capacity_limit_never_steal_handle():
    backend = StubBackend()
    c = cache(backend, max_handles=1)
    start, child, _ = frames()
    initial = [{"combat": {"turn": 1}}]
    c.begin_simulation()
    assert c.record(start, "a", child, "c1", initial, initial, "root")
    assert not c.record(child, "end", child, "c2", initial, initial, "c1")
    assert c.metrics()["skipped_capacity"] == 1
    c.begin_simulation()
    assert c.lookup(start, "b", initial, "root") is None
    assert c.lookup(start, "a", initial, "root").handle == "c1"
    c.close()
    assert backend.released == ["c1"]
    assert "c2" in backend.hidden  # caller still owns unretained child


@pytest.mark.parametrize("provider", ["nature", "other", "reward"])
def test_private_provider_guard_bypasses_entire_cache(provider):
    backend = StubBackend()
    backend.present_flags["root"] = provider
    c = cache(backend)
    parent, child, _ = frames()
    initial = [{"combat": {}}]
    c.begin_simulation()
    assert not c.record(parent, "a", child, "c1", initial, initial, "root")
    assert c.metrics()["retained_handles"] == 0
    assert c.metrics()["skipped_provider"] == 1
    c.close()
    assert backend.released == []


def test_new_private_provider_state_cannot_be_cached():
    backend = StubBackend()
    backend.present_flags["c1"] = "nature"
    c = cache(backend)
    parent, child, _ = frames()
    c.begin_simulation()
    assert not c.record(parent, "a", child, "c1", [], [], "root")
    assert c.metrics()["retained_handles"] == 0
    assert c.metrics()["enabled"] is False
    # Even if a later child no longer exposes the transient provider,
    # it must never inherit the old deterministic-root cache permission.
    assert not c.record(child, "end", child, "c2", [], [], "c1")
    assert c.metrics()["retained_handles"] == 0
    c.close()


def test_record_validates_live_root_receipt_after_simulation_begins():
    backend = StubBackend()
    c = cache(backend)
    parent, child, _ = frames()
    c.begin_simulation()
    backend.hidden["root"] = ("state-root", "changed-before-action")
    with pytest.raises(ValueError, match="changed private search root"):
        c.record(parent, "a", child, "c1", [], [], "root")
    c.close()
    assert "c1" in backend.hidden  # unretained caller-owned child



def test_private_rng_receipt_mutation_fails_closed_even_if_public_same():
    backend = StubBackend()
    c = cache(backend)
    parent, child, _ = frames()
    c.begin_simulation()
    assert c.record(parent, "a", child, "c1", [], [], "root")
    c.begin_simulation()
    backend.hidden["c1"] = ("state-1", "rng-mutated")
    assert backend.exact_continuation_receipt("c1")["exactHash"] == hashlib.sha256(
        b"state-1"
    ).hexdigest()
    with pytest.raises(ValueError, match="changed stochastic continuation"):
        c.lookup(parent, "a", [], "root")
    c.close()
    assert backend.released == ["c1"]


def test_public_history_and_frame_mismatch_fail_closed():
    backend = StubBackend()
    c = cache(backend)
    parent, child, _ = frames()
    c.begin_simulation()
    assert c.record(parent, "a", child, "c1", [], [], "root")
    c.begin_simulation()
    with pytest.raises(ValueError, match="conflicts"):
        c.lookup(parent, "a", [{"different": True}], "root")
    with pytest.raises(ValueError, match="conflicts"):
        c.lookup(child, "a", [], "c1")
    c.close()


def test_rejects_unsupported_sampler_capability_and_bad_handle_limit():
    backend = StubBackend()
    with pytest.raises(ValueError, match="capability"):
        ExactRootPrefixCache(
            backend, "root", max_handles=8, capability="fair-history-sampler",
        )
    for limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive handle cap"):
            cache(backend, max_handles=limit)


def test_current_parent_rng_drift_fails_even_with_unchanged_cached_child():
    backend = StubBackend()
    c = cache(backend)
    parent, child, _ = frames()
    c.begin_simulation()
    assert c.record(parent, "a", child, "c1", [], [], "root")
    c.begin_simulation()
    backend.hidden["root"] = ("state-root", "rng-diverged")
    with pytest.raises(ValueError, match="search root changed private"):
        c.begin_simulation()
    # An already-started simulation also checks the actual live parent
    # rather than trusting a public key and a retained child receipt.
    with pytest.raises(ValueError, match="current parent changed private"):
        c.lookup(parent, "a", [], "root")
    c.close()
    assert backend.released == ["c1"]
