#!/usr/bin/env python3
"""H24 opt-in hard resource-monotone value experiment on generated Silent A0.

Requires a locally available H22 H21-v4 checkpoint (R95 is the proposed
selection control). The first H24 arm constrains HP and gold only; the potion
identity whitelist is empty; relic hard monotonicity is disabled. Existing
H22 model training remains unchanged without --h24.
"""
from __future__ import annotations

import sys


def h24_cli_arguments(overrides: list[str]) -> tuple[str, ...]:
    if not any(
        arg == "--initial-h22" or arg.startswith("--initial-h22=")
        for arg in overrides
    ):
        raise ValueError("H24 requires --initial-h22 pointing to the H22 R95 baseline")
    try:
        from tools.train_h22_h21_generated_a0 import h22_cli_arguments
    except ModuleNotFoundError:
        from train_h22_h21_generated_a0 import h22_cli_arguments
    return (
        *h22_cli_arguments(overrides),
        "--h24",
        "--rounds", "5",
    )


if __name__ == "__main__":
    try:
        from tools.train_h16_joint_act_pilot import main
    except ModuleNotFoundError:
        from train_h16_joint_act_pilot import main

    overrides = sys.argv[1:]
    try:
        resolved = h24_cli_arguments(overrides)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    # The user can override trial length by passing --rounds; argparse's
    # last occurrence wins, so put explicit user flags after defaults.
    resolved = (*resolved, *overrides)
    sys.argv = [sys.argv[0], *resolved]
    main()
