#!/usr/bin/env python3
"""H18: Silent A0 training on seeded, branching Overgrowth maps.

Keeps H17's model, reward, replay and full-combat search settings. Only the
ascension, default map generator, experiment seed namespace and run output
change. With no --execute, prints the complete run fingerprint/configuration.
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    from train_h16_joint_act_pilot import main

    defaults = [
        "--h17-adaptive",
        "--initial-h16", "results/h16-v5-combat-credit-r150/models/joint-r0079.pt",
        "--ascension", "0",
        "--map-profile", "generated",
        "--runs-per-round", "64",
        "--heldout-runs", "16",
        "--workers", "15",
        "--actor-inference", "shared",
        "--actor-device", "cuda",
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
    ]
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    main()
