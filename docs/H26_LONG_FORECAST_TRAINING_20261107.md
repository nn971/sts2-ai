# H26 long forecast training — single six-hour run

This run is a **single feature configuration**, not an A/B experiment.
The Silent at Ascension 0 trains over native-generated Overgrowth Act 1
using the **source-exact conditional next-intent distribution** together
with **estimated conditional gross incoming attack damage**. Search is
disabled. Macro and combat actors are independent, using H21 grounded
macro representation, H24 monotone Act critic and H26 combat actor-critic.

## Verified before training

The paired H26-D native 4-train/2-heldout engineering smokes completed:
- Source damage-risk coverage: 169/173 (97.7%) training and 51/51 (100%)
  held-out **living enemy-turn observations**.
- Current displayed attack HUD parity: 0 discrepancies in 152 checked
  public attacks (117 training + 35 held-out).
- Next-intent support audit: zero out-of-support observations in 97
  checked transitions.
- Zero search transitions; combat and macro checkpoints committed.

The test is **not** proof of all future combat damage or improved clear
rate. Damage forecasts are conditional on the player's observed current
powers, enemy powers, ascension, and announced intent executing normally;
future block, hidden stored damage and intervening effects are masked or
treated as changed states requiring recalculation.

## Configuration

| Setting | Value |
| --- | --- |
| Training | One run, no separate A/B or optimization pilot |
| Character / difficulty | Silent / A0 |
| Generated map | native-overgrowth-generated-default-v111 |
| Intent | public, v3 exact Native grid (supported conditions) |
| Estimated incoming damage | source v3 conditional gross pre-Block |
| Initialization | H22 R95 reference; H25 v2 R3 macro and combat |
| Worker processes | 15 |
| Training episodes per round | 64 |
| Fixed held-out episodes per round | 16 |
| Combat training sample cap per round | 4096 |
| Minibatch / epochs / learning rate | 8 / 1 / 1e-5 |
| Policy sampling temperature / combat discount | 0.85 / 0.99 |
| Wall-time cap | 21,600 seconds total active training |
| Maximum rounds | 200 (wall cap normally stops earlier) |
| Checkpoints | each committed round; keep last 5 and every 20th |
| Search | none: no PUCT, zero roots/transitions |
| Experiment output | results/h26-a0-forecast-6h-v1 |

A completed round may extend beyond the nominal six-hour budget.
The stopping check executes **after** an atomic progress commit.

## Preparation (does not start training)

In a fish shell with the project virtual environment active:

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch research/h26-search-free-tactical-20261107
git pull --ff-only

set emu ~/projects/sts2-emulator
git -C $emu fetch origin
git -C $emu switch research/h26-public-intent-law-20261107
git -C $emu pull --ff-only

fish -n scripts/h26_forecast_long.fish
fish scripts/h26_forecast_long.fish prepare
```

The default `prepare` mode verifies committed source checkouts,
initial model paths, CUDA, the v3 source catalog, and the exact dry-run
experiment fingerprint. **It performs no training or evaluation.**
The source code hash is committed into the experiment fingerprint, so a
subsequent code change requires a fresh output root.

## Starting or resuming (explicit user actions)

Start the single run only when ready:

```fish
fish scripts/h26_forecast_long.fish start
```

Observe it in a separate terminal:

```fish
cd ~/projects/sts2-ai
python tools/monitor_h26.py \
  --root results/h26-a0-forecast-6h-v1 --interval 30
```

After a crash, cancellation, or other interruption, resume the latest
verified round:

```fish
fish scripts/h26_forecast_long.fish resume
```

The runner verifies progress fingerprint, both model checkpoint hashes
and optimizer checkpoint hash, loads optimizer and Torch RNG state, and
continues at the next round. A partly collected round is replayed.
The original catalog must be retained unchanged. The six-hour wall
budget counts previously committed active training time on resume.

The launcher writes the console log to
`results/h26-a0-forecast-6h-v1.log`; one `progress.json` and
`live.json` are written under the output root. No extra A/B run is
created by the launcher.

## Acceptance and monitoring

Track trends in held-out Act clears, max floor reached, post-room HP,
combat wins and optimizer steps. Because the same held-out seeds repeat,
report clear rates as **held-out on the specified fixed seeds**, not as
general population win rate.

The monitor also displays:
- exact intent and damage coverage for living enemy observations;
- authoritative current-intent HUD parity and next-turn support audit;
- public decisions per second (collection stage), heartbeat age and
  cumulative committed wall budget;
- warnings if H26 executes searches or if public-law checks disagree.

Investigate any nonzero `current_hud_preview_parity.mismatched` or
`next_turn_support_audit.out_of_support`, lost checkpoints, or persistent
heartbeat age. The existing v3 smoke provided zero mismatches on its
checked sample, though full native fidelity remains conditional.

## Reproducibility

The experiment identity includes SHA-256 hashes of model initialization,
the exact source catalog bytes and AI inference/training files; it also
includes the pinned emulator Git revision, study seed, worker count,
hyperparameters and number of training/evaluation episodes. A different
AI code revision or catalog cannot be resumed under the same fingerprint.

Do not edit source code or source catalog during the long run. Retain
the independent held-out metrics, complete round summaries and at least
the periodic immutable model/optimizer snapshots for later analysis.
