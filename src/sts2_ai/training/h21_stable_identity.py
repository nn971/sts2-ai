"""H21 stable public concept identities with explicit neural-alias auditing.

A 4,096-bucket ID is too small for a full act card vocabulary: different
cards can collide. H21 uses 63-bit deterministic identities and decomposes
each identity into four independent 12-bit embedding lookups. A check on
their effective 48-bit embedding key prevents silent aliases. The source
label is public (card/item/event/attribute kind), never an instance ID.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

H21_IDENTITY_COMPONENT_BUCKETS = 4096
H21_IDENTITY_COMPONENTS = 4
H21_EMBEDDING_MODULUS = H21_IDENTITY_COMPONENT_BUCKETS ** H21_IDENTITY_COMPONENTS
H21_FULL_ID_MODULUS = (1 << 63) - 1


@lru_cache(maxsize=65536)
def stable_id(label: str) -> int:
    """Deterministic, positive int64-safe public concept identity."""
    if not label:
        return 0
    raw = int.from_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()[:8], "big",
    )
    return 1 + raw % (H21_FULL_ID_MODULUS - 1)


class H21StableIdentityHasher:
    """Detect both full-hash and composed-embedding collisions per decision."""

    def __init__(self) -> None:
        self._full: dict[int, str] = {}
        self._embedded: dict[int, str] = {}

    def __call__(self, label: str) -> int:
        hashed = stable_id(label)
        if not label:
            return hashed
        previous = self._full.get(hashed)
        if previous is not None and previous != label:
            raise ValueError(
                "H21 stable full identity collision: "
                + repr((previous, label)))
        embedded = hashed % H21_EMBEDDING_MODULUS
        if embedded == 0:
            raise ValueError("H21 nonempty identity mapped to padding")
        other = self._embedded.get(embedded)
        if other is not None and other != label:
            raise ValueError(
                "H21 stable neural identity collision: "
                + repr((other, label)))
        self._full[hashed] = label
        self._embedded[embedded] = label
        return hashed
