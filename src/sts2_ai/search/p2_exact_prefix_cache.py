"""P2.1 experimental exact Native root-conditioned prefix retention.

Only H16's explicitly privileged, deterministic root-conditioned sampler may
use this cache. Public-history keys alone never authorize reuse. The cache is
per combat-search root, keeps complete action-path identity, verifies versioned
private stochastic receipts on insertion/hit, and releases every retained
handle when the root search ends. Independent Nature/fair-history samplers are
outside this contract. This is opt-in and does not alter search policy logic.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from sts2_ai.emulator.protocol import StateHandle
from sts2_ai.search.combat_puct import (
    H16_EXPERIMENTAL_ROOT_CONDITIONED_CAPABILITY,
    CombatFrame,
)

PRIVATE_PROVIDER_FLAGS = (
    "referenceNaturePresent", "semanticOtherChancePresent",
    "semanticRewardAuditPresent",
)


@dataclass(frozen=True)
class CachedPrefix:
    handle: StateHandle
    parent_frame: CombatFrame
    child_frame: CombatFrame
    parent_history: tuple[dict[str, Any], ...]
    child_history: tuple[dict[str, Any], ...]
    parent_receipt: dict[str, str | bool | None]
    receipt: dict[str, str | bool | None]


class ExactRootPrefixCache:
    """Bounded private state handles; fail-closed equality and cleanup."""

    def __init__(
        self, backend: Any, root: StateHandle, *, max_handles: int,
        capability: str,
    ) -> None:
        if type(max_handles) is not int or max_handles < 1:
            raise ValueError("P2 exact-prefix cache must have a positive handle cap")
        if capability != H16_EXPERIMENTAL_ROOT_CONDITIONED_CAPABILITY:
            raise ValueError("P2 cache requires the explicit Native root capability")
        self.backend = backend
        self.root = root
        self.max_handles = max_handles
        self._root_receipt: dict[str, str | bool | None] | None = None
        self._enabled = False
        self._path: tuple[str, ...] = ()
        self._entries: dict[tuple[str, ...], CachedPrefix] = {}
        self._closed = False
        self.hits = 0
        self.misses = 0
        self.skipped_capacity = 0
        self.skipped_provider = 0
        self.receipt_calls = 0
        self.receipt_seconds = 0.0

    @staticmethod
    def _allowed(receipt: dict[str, str | bool | None]) -> bool:
        return (
            isinstance(receipt.get("exactHash"), str)
            and isinstance(receipt.get("runStochasticHash"), str)
            and isinstance(receipt.get("worldStochasticHash"), str)
            and all(receipt.get(flag) is False for flag in PRIVATE_PROVIDER_FLAGS)
        )

    def _receipt(self, handle: StateHandle) -> dict[str, str | bool | None]:
        started = perf_counter()
        try:
            return self.backend.exact_continuation_receipt(handle)
        finally:
            self.receipt_seconds += perf_counter() - started
            self.receipt_calls += 1

    def begin_simulation(self) -> None:
        if self._closed:
            raise RuntimeError("P2 cache was closed")
        receipt = self._receipt(self.root)
        if self._root_receipt is None:
            self._root_receipt = receipt
            self._enabled = self._allowed(receipt)
        elif receipt != self._root_receipt:
            raise ValueError(
                "P2 search root changed private stochastic continuation"
            )
        self._path = ()

    def lookup(
        self, parent: CombatFrame, action_id: str,
        history: list[dict[str, Any]], parent_handle: StateHandle,
    ) -> CachedPrefix | None:
        if self._closed:
            raise RuntimeError("P2 cache was closed")
        key = (*self._path, action_id)
        entry = self._entries.get(key) if self._enabled else None
        if entry is None:
            self.misses += 1
            return None
        if entry.parent_frame != parent or entry.parent_history != tuple(history):
            raise ValueError("P2 retained prefix conflicts with public parent/history")
        if (
            self._receipt(parent_handle)
            != entry.parent_receipt
        ):
            raise ValueError(
                "P2 current parent changed private stochastic continuation"
            )
        if self._receipt(entry.handle) != entry.receipt:
            raise ValueError("P2 retained prefix changed stochastic continuation")
        self._path = key
        self.hits += 1
        return entry

    def record(
        self, parent: CombatFrame, action_id: str,
        child: CombatFrame, handle: StateHandle,
        parent_history: list[dict[str, Any]],
        child_history: list[dict[str, Any]],
        parent_handle: StateHandle,
    ) -> bool:
        """Return True iff handle ownership was transferred into the cache."""
        if self._closed:
            raise RuntimeError("P2 cache was closed")
        key = (*self._path, action_id)
        if self._enabled and key in self._entries:
            raise ValueError("P2 attempted to replace a retained prefix")
        at_root = not self._path
        self._path = key
        if not self._enabled:
            self.skipped_provider += 1
            return False
        if len(self._entries) >= self.max_handles:
            self.skipped_capacity += 1
            return False
        parent_receipt = self._receipt(parent_handle)
        receipt = self._receipt(handle)
        if not self._allowed(parent_receipt) or not self._allowed(receipt):
            # Once an incompatible private provider appears, descendant
            # prefixes cannot inherit the root-conditioned authorization.
            self._enabled = False
            self.skipped_provider += 1
            return False
        if at_root and parent_receipt != self._root_receipt:
            raise ValueError(
                "P2 recorded child from changed private search root"
            )
        self._entries[key] = CachedPrefix(
            handle, parent, child,
            tuple(deepcopy(parent_history)), tuple(deepcopy(child_history)),
            parent_receipt, receipt,
        )
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        handles = [entry.handle for entry in self._entries.values()]
        self._entries.clear()
        self._root_receipt = None
        if handles:
            released = self.backend.release_many(handles)
            if released != len(handles):
                raise RuntimeError(
                    f"P2 retained-prefix handle leak: {released}/{len(handles)}"
                )

    def metrics(self) -> dict[str, int | bool | float]:
        return {
            "enabled": self._enabled and not self._closed,
            "hits": self.hits, "misses": self.misses,
            "emulator_step_calls_avoided": self.hits,
            "receipt_calls": self.receipt_calls,
            "receipt_seconds": self.receipt_seconds,
            "retained_handles": len(self._entries),
            "handle_limit": self.max_handles,
            "skipped_capacity": self.skipped_capacity,
            "skipped_provider": self.skipped_provider,
        }
