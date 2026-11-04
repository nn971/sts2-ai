#!/usr/bin/env python3
"""H18: Silent A0 training on seeded, branching Overgrowth maps.

Keeps H17's model, reward, replay and full-combat search settings.
P2 training staging opts into the separately validated fixed-shape-4 shared
actor and preserves the existing fused emulator transport. The checkpoint,
map generator, ascension and search curriculum remain H18-specific.
With no --execute, print the complete experiment fingerprint and configuration.
"""
from __future__ import annotations

import sys


def h18_default_arguments() -> tuple[str, ...]:
    """The staged H18 defaults, inspectable without importing torch or the game."""
    return (
        "--h17-adaptive",
        "--initial-h16", "results/h16-v5-combat-credit-r150/models/joint-r0079.pt",
        "--ascension", "0",
        "--map-profile", "generated",
        "--runs-per-round", "64",
        "--heldout-runs", "16",
        "--workers", "15",
        "--actor-inference", "shared",
        "--actor-device", "cuda",
        # P2: accepted 15-worker actor optimization, teacher-parity checked.
        "--actor-max-batch", "16",
        "--actor-fixed-shape-batch-size", "4",
        "--actor-max-delay-ms", "2",
        # P2: already-fused emulator step/frame transport; cache stays disabled.
        "--rollout-transport", "fused",
        "--search-depth", "0",
        "--search-probability", "1",
        "--macro-prior-weight", "0.5",
        "--h17-macro-decay-rounds", "120",
        "--h17-horizon-initial", "0.4113950303216448",
        "--h17-horizon-target", "0.77",
        "--h17-horizon-rounds", "200",
        "--h17-hp-reference", "100",
        "--h17-search-p16", "0.15",
        "--h17-search-p64", "0.02",
        "--study-seed", "20261120",
        "--rounds", "40",
    )


if __name__ == "__main__":
    from train_h16_joint_act_pilot import main

    sys.argv = [sys.argv[0], *h18_default_arguments(), *sys.argv[1:]]
    main()
