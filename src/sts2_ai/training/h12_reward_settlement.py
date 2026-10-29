"""Derive after-reward resource labels from existing H12 full-run journals.

Combat survival and exit HP are observed at the immediate combat boundary.
Persistent resource labels may instead use the last *public* same-floor
noncombat state before the next combat or next floor. This reflects the
frozen H6-C reward-choice policy. No hidden state or branch is queried.
"""
from __future__ import annotations

from typing import Any


def settled_public_resources(
    run: dict[str, Any], combat_index: int,
) -> tuple[dict[str, Any], str]:
    combats = run["combats"]
    combat = combats[combat_index]
    immediate = combat["exit"]
    if not combat["survived"]:
        return immediate, "defeat_immediate"
    start = combat["exit_observation_index"]
    stop = (
        combats[combat_index + 1]["entry_observation_index"]
        if combat_index + 1 < len(combats) else float("inf")
    )
    floor, act = immediate.get("floor"), immediate.get("act")
    # All macro states are public observations. They include reward menus,
    # resolved choices and, eventually, subsequent map-room entry.
    best = immediate
    found = False
    previous_index = start - 1
    for frame in run["macro_states"]:
        index = frame["observation_index"]
        if index < start:
            continue
        if index >= stop:
            break
        if index <= previous_index:
            raise ValueError("Nonmonotone H12 public macro frame ordering")
        previous_index = index
        state = frame["state"]
        if state.get("floor") != floor or state.get("act") != act:
            break
        best = state
        found = True
    return best, "settled" if found else "immediate_only"
