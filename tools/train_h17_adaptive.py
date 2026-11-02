#!/usr/bin/env python3
"""H17 training entrypoint: adaptive full-combat PUCT, learnable macro and HP horizon.

The shared H16 trainer owns checkpoint/optimizer compatibility and the joint
neural network. H17 supplies carefully chosen defaults; explicit user options
override them. Passing no --execute prints the complete configuration.
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    from train_h16_joint_act_pilot import main

    defaults = [
        "--h17-adaptive",
        "--initial-h16", "results/h16-v5-combat-credit-r150/models/joint-r0079.pt",
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
        "--h17-search-p16", "0.15",
        "--h17-search-p64", "0.02",
        "--study-seed", "20261117",
        "--rounds", "40",
    ]
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    main()
