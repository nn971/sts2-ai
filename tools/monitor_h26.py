#!/usr/bin/env python3
"""Read-only monitor for H26 forecast-enabled, search-free long training."""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _pct(value: Any) -> str:
    return f"{100 * value:.0f}%" if (
        isinstance(value, (int, float)) and math.isfinite(value)
    ) else "-"


def _number(value: Any, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}" if (
        isinstance(value, (int, float)) and math.isfinite(value)
    ) else "-"


def render_h26(root: Path, *, now: float | None = None) -> str:
    """Render a snapshot without modifying running jobs or saved checkpoints."""
    timestamp = time.time() if now is None else now
    progress = _json(root / "progress.json") or {}
    live = _json(root / "live.json") or {}
    rows = progress.get("rounds", [])
    if not isinstance(rows, list):
        rows = []
    identity = progress.get("experiment", {})
    if not isinstance(identity, dict):
        identity = {}
    lines = [
        f"H26 forecast training   {datetime.fromtimestamp(timestamp):%Y-%m-%d %H:%M:%S}",
        f"Root: {root}",
    ]
    if identity:
        if (identity.get("next_damage_risk_enabled") is not True
                or not identity.get("intent_catalog_sha256")):
            lines.append("WARNING: committed experiment lacks complete forecast features!")
        mode = (
            "damage + intent"
            if identity.get("next_damage_risk_enabled") else "intent only"
        )
        lines.append(
            f"Mode: {mode}"
            f" | A{identity.get('ascension', '?')}"
            f" | workers {identity.get('workers', '?')}"
            f" | train/heldout {identity.get('runs_per_round', '?')}/"
            f"{identity.get('heldout_runs', '?')}"
        )
    elapsed = progress.get("elapsed_wall_seconds")
    budget = progress.get("wall_budget_seconds")
    if isinstance(elapsed, (int, float)) and math.isfinite(elapsed):
        headline = f"Committed: {len(rows)} rounds | active time {elapsed / 3600:.2f}h"
        if isinstance(budget, (int, float)) and math.isfinite(budget):
            headline += f" / {budget / 3600:.2f}h"
            headline += f" | remaining at last commit {max(0, budget - elapsed) / 3600:.2f}h"
        lines.append(headline)

    if rows:
        lines.append(
            "Round  Train clear  Held clear  Held floor  Combat/Macro  "
            "Intent%  Damage%  HUD errors  Support errors  Dec/s"
        )
        for row in rows[-8:]:
            if not isinstance(row, dict):
                continue
            tr, ev = row.get("train", {}), row.get("heldout", {})
            tr, ev = (tr if isinstance(tr, dict) else {},
                      ev if isinstance(ev, dict) else {})
            combat, macro = row.get("combat_update") or {}, row.get("macro_update") or {}
            coverage = row.get("intent_law_coverage") or {}
            risk = row.get("damage_risk_coverage") or {}
            held_law = coverage.get("heldout") or {}
            held_risk = risk.get("heldout") or {}
            parity = held_risk.get("current_hud_preview_parity") or {}
            support = held_law.get("next_turn_support_audit") or {}
            floor = (ev.get("floor_progress") or {}).get("mean_max_floor_reached")
            seconds = (row.get("timing") or {}).get("collection_seconds")
            decisions = tr.get("decisions")
            rate = (
                decisions / seconds if isinstance(decisions, (int, float))
                and isinstance(seconds, (int, float)) and seconds > 0 else None
            )
            lines.append(
                f"{row.get('round', '?'):>5}  "
                f"{tr.get('cleared', 0):>3}/{tr.get('completed', 0):<6}     "
                f"{ev.get('cleared', 0):>3}/{ev.get('completed', 0):<6}      "
                f"{_number(floor):>5}     "
                f"{combat.get('steps', 0):>3}/{macro.get('steps', 0):<4}     "
                f"{_pct(held_law.get('living_availability_rate')):>6}   "
                f"{_pct(held_risk.get('availability_rate')):>6}   "
                f"{parity.get('mismatched', '-'):>5}         "
                f"{support.get('out_of_support', '-'):>5}      "
                f"{_number(rate):>5}"
            )
            search = row.get("combat_search") or {}
            if search.get("roots", 0) or search.get("transitions", 0):
                lines.append("WARNING: unexpected H26 combat search activity!")
            if parity.get("mismatched", 0) or support.get("out_of_support", 0):
                lines.append(
                    "WARNING: intent/damage fidelity disagreement in committed round!"
                )
        last = rows[-1]
        if isinstance(last, dict):
            combat = last.get("combat_update") or {}
            hp = (last.get("heldout") or {}).get("mean_terminal_hp_fraction")
            lines.append(
                f"Latest combat: examples={combat.get('examples','?')}"
                f"  loss={_number(combat.get('loss'),4)}"
                f"  heldout terminal HP={_pct(hp)}"
            )
            damage = (last.get("damage_risk_coverage") or {}).get("train") or {}
            reason = damage.get("unavailable_reasons") or {}
            if reason:
                lines.append("Training damage unavailable reasons: "
                             + json.dumps(reason, sort_keys=True))
    else:
        lines.append("No committed round yet.")
    if live:
        age = timestamp - live.get("timestamp_unix", timestamp)
        if not isinstance(age, (int, float)) or not math.isfinite(age):
            age = float("nan")
        lines.append(
            f"Current: round {live.get('round','?')}"
            f" | {live.get('stage','?')}"
            f" | episodes {live.get('completed_episodes','-')}/"
            f"{live.get('total_episodes','-')}"
            f" | heartbeat age {_number(age,0)}s"
        )
        if age > 120:
            lines.append("WARNING: no heartbeat for more than 120 seconds.")
    else:
        lines.append("Awaiting live.json heartbeat.")
    lines.append("Read-only monitor; Ctrl+C exits this display.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.interval <= 3600:
        parser.error("--interval must be between 1 and 3600 seconds")
    try:
        while True:
            if not args.once:
                print("\033[2J\033[H", end="")
            print(render_h26(args.root), flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
