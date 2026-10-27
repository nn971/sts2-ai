# Act 1 model selection: independent 256 + 512 seed cohorts (v11)

## Why

The 120-round continuation showed held-out clears of 34/128 (round 40),
44/128 (round 60), 33/128 (round 80), 41/128 (round 100), and 33/128
(round 120). Round 60 is the **provisional** champion, not a confirmed
population improvement: it was selected after repeatedly looking at the
same historical 128 evaluation seeds.

This protocol avoids using the historical cohort to claim a winner.

## Two new cohorts, three different roles

- Historical **128 seeds** (`ppo-v7-v8-slyfix-eval`): debugging/learning
  curve only. Do not use them for final model selection or p-values.
- **Selection 256 seeds** (`ppo-v8-fresh-selection-v11`):
  evaluate checkpoints 40, 60, 100, and 120 *paired on exactly the same
  seeds* in the same pinned native Act-1 emulator. Pick the highest
  certified clear count, then greatest mean frontier progress, then
  prefer round 40 for an exact tie. This rule is determined in advance.
- **Confirmation 512 seeds** (`ppo-v8-fresh-confirmation-v11`):
  evaluate **only** the winner and the fixed round-40 reference. Report
  their clear rates, 95% Wilson intervals, and the exact two-sided paired
  McNemar test. Treat the test as one predeclared confirmation, not
  an opportunity to select a different model.

Do not change training hyperparameters or choose another checkpoint after
seeing confirmation outcomes. No estimate from the selection 256 should
be treated as an independent final-model estimate.

## Run (Fish, Ubuntu WSL)

The existing `results/ppo-v8-batched-40/models/` directory must contain
`stage-0040.json`, `stage-0060.json`, `stage-0100.json`, and
`stage-0120.json`. The original `ppo-v8-batched-40 (2).zip` archive
contains all these files.

```fish
git fetch origin
git switch --track origin/agent/act1-checkpoint-selection-v11
git submodule update --init --recursive
python -m pytest -q tests/test_act1_checkpoint_selection.py
fish tools/validate_v8_checkpoints.fish
```

The script uses 12 isolated .NET emulator workers, reuses the checkpoint
model JSON files unmodified, and prints progress every ten paired seeds.
These are **greedy** evaluations under the certified
`native-act1-boss-v1` goal, not exploratory PPO rollouts.

On fresh WSL builds where the emulator CLI has not already been built,
run `dotnet build emulator/src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj
-c Release` once before the evaluation, or call the underlying Python
selection evaluator with `--build`. There is no hidden fallback to
the six-floor legacy environment.

Results are written to:

- `results/ppo-v8-checkpoint-selection-v11/selection-256.json`
- `results/ppo-v8-checkpoint-selection-v11/confirmation-512.json`
- Corresponding `.partial.jsonl` journals. If interrupted,
  **run the same Fish command again**. Journal identity includes model
  absolute paths + SHA256 bytes, emulator commit, seed cohort, goal,
  decision limit, and selected report SHA. Completed seed groups are
  replayed neither on resume nor after successful finalization.
  If a completed report already exists, the runner stops deliberately
  rather than overwriting it.

The final evaluator loads the selection report, enforces different seed
prefixes, checks selected checkpoint SHA256 hashes, and refuses to
substitute a model silently. With a round-40 selection winner, the 512
fresh seeds evaluate round 40 once and report no artificial paired gain.

The older `tools/evaluate_phase_split.py` now accepts
`--episode-goal` and defaults to native certified Act 1;
historical three-act mode remains selectable explicitly. Its summary
uses `act1_cleared`, not a floor-number heuristic.

## How to interpret results

The exact paired McNemar p-value tests a zero marginal difference between
the two already-locked deterministic greedy policies, conditional on
discordant outcomes on that particular final cohort. A low p-value does
not address robustness to new training initializations or emulator fidelity.
The Wilson intervals describe individual model clear rates.

Near-kill counts describe a boss *initial-roster* HP proxy only; The Kin
can have summons, so do not mistake it for a phase-aware ground truth.

## Next after confirmation

Run at least two independent batched-v8 training replicates from the same
warm-start model with new distinct training prefixes and RNG seeds, then
compare a hidden-width control (64 or 128) before investing in attention.
Do not use the final 512-seed confirmation cohort for iterative
hyperparameter tuning.
