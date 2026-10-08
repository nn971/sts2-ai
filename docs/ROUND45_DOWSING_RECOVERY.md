# Recover the round-45 native self-play run after Dowsing autoplay

## What failed

The 300-round run stopped at zero-based **round 45**, episode
`neural-train-45-28`, decision 49, on Act 1 floor 7.
Distilled Chaos (potion slot 2) attempted to autoplay the unplayable
Neow quest card Dowsing. Emulator revision
`cc78c367a80f88ecef72cc72899892298f20ef90` checked
`MechanicsImplemented` too early and threw:

```text
NotSupportedException: Autoplay cannot resolve unsupported card 'Dowsing'.
```

The AI correctly aborted instead of assigning a synthetic defeat reward.
The 45 **completed** rounds (1,440 episodes) should remain in the atomic
checkpoint `results/native-tempered-300.pt`. The last incomplete
cohort is NOT scored. The training log contains no round summaries
because the previously invoked script had not yet gained live logging.

The fix is committed in the separate emulator project; this AI revision
pins the corrected emulator. The fix lets the already existing
unplayable-card no-op/autoplay-discard path execute *before* checking
whether a card has supported gameplay effects. Unsupported **playable**
effects still abort; we do not mask other missing emulator mechanics.

## Restore learned weights without forging a cross-revision resume

The old checkpoint was created under the older emulator revision. The
trainer deliberately refuses `--resume` when the emulator pin changes:
the meaning of a saved trajectory may have changed. Do NOT manually edit
the checkpoint revision, fingerprint or submodule pin to circumvent this.

Instead, safely export only the neural parameters from the last completed
round. This preserves the learned policy and value head but intentionally
starts a **fresh** optimizer, seed set, round counter and reward
curriculum. Original checkpoint and failure report should be archived.

In fish, from `sts2-ai`:

```fish
git pull --ff-only
git submodule update --init --recursive
source .venv/bin/activate.fish
mkdir -p results

python tools/export_selfplay_checkpoint_model.py \
  --checkpoint results/native-tempered-300.pt \
  --output results/native-tempered-round45-model.json
```

The export prints exactly how many completed rounds, completed
episodes and wins are present **in your checkpoint**. This information
is not available from the six-line failure log alone; if the checkpoint
is missing or unreadable, do not fabricate those statistics.

Then run a new **255 × 32 = 8,160 episode** experiment, which,
combined with the 45 previously completed rounds, totals the original
9,600-episode research budget (but is **not one optimizer-continuous
experiment**). The more conservative temperature starts at 0.08, close
to the previous schedule's round-45 temperature, and decays to 0.035
over 75 new rounds:

```fish
set -gx OMP_NUM_THREADS 1
set -gx MKL_NUM_THREADS 1

python -u tools/train_selfplay.py \
  --build \
  --environment native-overgrowth \
  --initialize-from-model results/native-tempered-round45-model.json \
  --workers 15 --rounds 255 --episodes 32 \
  --max-decisions 4096 --dimension 128 --hidden 32 \
  --seed 44 --learning-rate 0.001 \
  --temperature-start 0.08 \
  --temperature-end 0.035 \
  --temperature-decay-rounds 75 \
  --entropy-weight 0.002 \
  --evaluate-seeds 128 \
  --checkpoint results/native-tempered-recovery-255.pt \
  --output results/native-tempered-recovery-255-model.json \
  --report results/native-tempered-recovery-255-report.json \
  2>&1 | tee results/native-tempered-recovery-255.log
```

With 32 episodes per cohort, 15 process-isolated workers can run up to
15 emulator episodes concurrently. This is deliberately aggressive for a
Ryzen 9700X (8 cores / 16 threads) and 24 GB RAM: monitor memory pressure,
swapping and elapsed seconds per round. More processes are not guaranteed to
reduce round time. The worker count is scheduling only, does not change
episode seeds or the on-policy cohort, and is not included in the checkpoint
fingerprint. If necessary, decrease `--workers` without restarting the
learning experiment; preserve all other training arguments.

The updated trainer emits a flushed live progress report for **every
round** and evaluation milestones, and `tee` retains a complete log.
This does not recover old optimizer momentum, but no learned neural
weights are discarded.

If this new experiment is interrupted without an emulator revision
change, rerun exactly the same command adding `--resume`, keeping
the exported source model and all configuration options unchanged.
When the emulator changes again, do not bypass the revision lock:
use the same explicit checkpoint extraction procedure.
