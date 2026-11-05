# H22 long training, checkpoint continuation and independent audit

## Design and acceptance scope

- Silent A0, native Overgrowth generated maps, H21 grounded macro graph and packed deterministic CUDA actor, 15 emulator workers.
- **Trained starting policy**: `results/h22-packed-paired16-two-rounds-v1/models/joint-r0002.pt`. New `--initial-h22` loads the schema-verified H21 checkpoint with a **fresh optimizer and replay buffer**. The checkpoint's SHA256 is part of the immutable experiment fingerprint.
- **64 training episodes per round**, 16 fixed search-free heldout seeds. Initial evaluation, round 1, every fifth round, and the planned final round: `--heldout-every-rounds 5`. Non-evaluated progress rows contain `heldout: null` and `heldout_evaluated: false`.
- **150 planned maximum rounds**. Only `--stop-after-round` and `--max-wall-seconds` vary between resumes; both are excluded from the training fingerprint. The round plan, model source, settings and emulator revision must stay identical.
- Per-round checkpoint and optimizer snapshot `models/joint-rNNNN.pt` and `models/optimizer-rNNNN.pt` plus H17 replay and `progress.json`. Resume verifies model digest and uses the optimizer snapshot corresponding to the last **committed** progress round; in-progress work may be repeated after interruption.
- PUCT is still 16 simulations, depth 8, up to one root per episode with adaptive probability .15. This is a validated, bounded search setting. Full-combat PUCT is a separate experiment, not yet demonstrated at 64 episodes/round.
- A **32-seed independent audit** is a separate read-only script and seed namespace. Run it once on a checkpoint after completing the long experiment, without using it to select hyperparameters.

## Fish commands — stage four rounds

```fish
cd ~/projects/sts2-ai
git switch perf/p2-h22-packed-collator-20261105
git pull --ff-only

set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
set initial results/h22-packed-paired16-two-rounds-v1/models/joint-r0002.pt
set out results/h22-h21-packed-a0-long-v1

test -f $initial; or begin
    echo "Missing trained H22 checkpoint $initial" >&2
    exit 1
end

python -u tools/train_h22_h21_generated_a0.py \
    --emulator-root $emu --emulator-revision $rev \
    --initial-h22 $initial --macro-prior-weight 0 \
    --device cuda --actor-inference shared --actor-device cuda \
    --actor-deterministic-cuda --actor-h21-collator packed \
    --actor-max-batch 4 --actor-fixed-shape-batch-size 4 \
    --rounds 150 --runs-per-round 64 --heldout-runs 16 \
    --heldout-every-rounds 5 --workers 15 \
    --study-seed 20261106 --search-depth 8 \
    --h17-search-p16 0.15 --h17-search-p64 0 \
    --max-searches 1 --max-decisions 4096 --minibatch 8 \
    --evaluation-horizon-lambda 0.4113950303216448 \
    --stop-after-round 4 \
    --output-root $out --execute
```

The staged four rounds test optimizer snapshots, sparse evaluation and progress monitoring on local CUDA before a longer run.

## Continue for a six-hour session

Reuse the **exact same command**, replacing the last two lines of controls with:

```fish
    --max-wall-seconds 21000 \
    --output-root $out --resume --execute
```

Remove `--stop-after-round 4`. The initial `--rounds 150` must remain fixed. The 21,000-second budget is **soft**: the trainer stops only after committing an entire round, so a long last round may exceed it. A later session uses the same `--resume` command and its own wall budget.

## Monitoring and independent audit

```fish
set out results/h22-h21-packed-a0-long-v1
python tools/monitor_h22_long_training.py \
    --progress $out/progress.json --latest 12

# Only after selecting the final checkpoint by the original training and
# repeatedly heldout evaluations: one fresh independent cohort.
python tools/audit_h22_independent.py \
    --progress $out/progress.json --emulator-root $emu \
    --audit-count 32 --workers 15 \
    --audit-tag independent-final-v1 --execute
```

The monitor prints committed training floor, clears, search roots, transitions, TPS, update count, replay steps and sparse heldout means. The audit validates the checkpoint digest and training provenance, excludes seeds already used for training/heldout, and writes `independent-audit-rNNNN.json`. It protects an existing output file from overwrites. Preserve the independent audit as a single honest generalization measurement.

## Stop/go assessment

The two-round matched pilot gave heldout mean maximum floor 1.0 → 6.4375 → 6.625 and no Act clears in 16 seeds. The initial floor-1 deadlock resolved, but the +0.1875 gain between trained rounds is weak. Track actual Act clears, new-map coverage, censored episodes, distribution of floors, training-vs-heldout divergence, replay stability and the cost share of each stage. A successful four-round CUDA pilot is the next gate before allocating a multihour session.
