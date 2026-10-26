# H6-C public replay of macro decisions

Read-only evaluation tooling on top of the H6-C checkpoint format. It does not
change training, policy execution, emulator mechanics, RNG or the inspector.

## What you can inspect

- A **sticky complete Act 1 map** is drawn from public `map` nodes, their
  `next_node_ids` and room types. Both agents' chosen map nodes are outlined
  in green / amber. All published nodes are included, not just the next choice.
- The **publicly reported boss identity**, permanently shown in the header.
  If a particular emulator observation doesn't publish a boss key, the report
  says "not exposed"; it never queries privileged state or guesses identity.
- A full chronological trace of **every decision**, including combat, public
  observation, legal actions and logits. The HTML report collapses ordinary
  combat decisions into floor counts; full decisions remain in JSON / JSONL.
- Expandable floor-by-floor macro decisions, including the selected legal
  action, action alternatives, scores (logits, **not** predicted win chances),
  value estimate, HP/gold and final outcome.
- A first-divergence report that distinguishes a policy disagreement on the
  same observation from differing observations. Same source seed does not
  guarantee equal stochastic trajectories after actions diverge.

## Reproduce selected runs (fish shell)

Use the pinned `sts2-emulator` revision and existing checkpoints. Example:

```fish
cd ~/projects/sts2-ai
source .venv/bin/activate.fish
git switch research/h6c-decision-trace-map-boss-20261026

python -u tools/replay_h6_macro.py \
  --player a_h6c80=results/nature/h6c-macro-attention-r80.json \
  --player b_h6c160=results/nature/h6c-macro-attention-a10.json \
  --comparison results/nature/h6c-r80-r160-fourway-confirmation.json \
  --sample 10 \
  --output-dir results/nature/h6c-decision-replays
```

Outputs:

- `manifest.json`: seeds, model SHA-256s, boss identity, outcomes.
- `<seed>.html`: self-contained side-by-side viewer with sticky complete map.
- `<seed>.md`: compact floor-by-floor reading version.
- `<seed>.<model>.json`: full structured public decision trace.
- `<seed>.<model>.jsonl`: metadata, map snapshots and individual decisions.

To inspect a particular previously evaluated source:

```fish
python -u tools/replay_h6_macro.py \
  --player a_h6c80=results/nature/h6c-macro-attention-r80.json \
  --player b_h6c160=results/nature/h6c-macro-attention-a10.json \
  --seed h6c-r80-r160-fresh-confirm-v2-confirmation-a10-000224 \
  --output-dir results/nature/h6c-specific-replay
```

Open the `<seed>.html` file locally in a browser. The header stays pinned as
you scroll the decisions, so the map and boss remain available.

## Provenance and limitations

The data comes solely from the **fair player-facing observation** and legal
actions at each live run boundary. Public source-state snapshots are captured,
but the `map` list is stored separately when changed to avoid unnecessary
duplication. Raw legal-action IDs/payloads are retained only for audit / labels.
The replay reports do not include arbitrary emulator exact-state hashes,
unrevealed map details, hidden RNG streams, committed shuffled order or Nature
private information.

The run hook `play_run(..., decision_observer=...)` already exists. Record
only explicitly replayed runs, not all training / large-scale evaluation runs.
No changes to `sts2-emulator`, its native mod or inspector are necessary.

The map layout in the HTML is a **graph visualization**, not necessarily the
native game's exact horizontal coordinates. It uses the public floor and
available columns, and otherwise a stable per-floor layout. If the boss
identifier is unavailable in the pinned public schema, address that as a
separate observation-interface request; it must not become a fidelity gate.
