#!/usr/bin/env python3
"""H22: opt-in H21 grounded public macro representation in joint Act 1 training.

This is a NEW model/codec checkpoint family. Inference, root-conditioned search,
joint optimization and H17 replay all use H21 v4 representations. H16 R79
contributes only shape-compatible initialization parameters, never an H21
checkpoint or claims of numerically equivalent policy behavior.
"""
from __future__ import annotations

import sys


def h22_default_arguments() -> tuple[str, ...]:
    """Generated Silent A0 defaults; inspect without launching the emulator."""
    from train_h18_generated_a0 import h18_default_arguments

    baseline = list(h18_default_arguments())
    # H21 relations are dynamically collated. The validated P2 fixed-shape
    # H16 actor bucket must remain off until graph-aware batching is certified.
    baseline.extend((
        "--representation", "h21",
        "--actor-fixed-shape-batch-size", "0",
        "--actor-max-batch", "4",
        "--minibatch", "8",
        "--rounds", "40",
    ))
    return tuple(baseline)


if __name__ == "__main__":
    from train_h16_joint_act_pilot import main

    sys.argv = [sys.argv[0], *h22_default_arguments(), *sys.argv[1:]]
    main()
