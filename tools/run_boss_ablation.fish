#!/usr/bin/env fish
# Run a paired 50-round boss-health ablation; every stage logs progress live.
# Execute from any directory: fish tools/run_boss_ablation.fish
set -l root (dirname (status --current-filename))/..
cd $root; or exit 1

if not test -f .venv/bin/activate.fish
    echo "[boss] Missing .venv/bin/activate.fish; create the project's virtualenv first" >&2
    exit 2
end
source .venv/bin/activate.fish
mkdir -p results

set -l source_checkpoint results/native-tempered-recovery-255.pt
set -l initial_model results/native-tempered-round131-model.json
if not test -f $source_checkpoint
    echo "[boss] Missing $source_checkpoint (checkpoint after 86 completed recovery rounds)" >&2
    exit 2
end
if not test -f $initial_model
    python tools/export_selfplay_checkpoint_model.py \
        --checkpoint $source_checkpoint \
        --output $initial_model
    or exit 1
end

set -gx OMP_NUM_THREADS 1
set -gx MKL_NUM_THREADS 1

# Identical initial weights, learner seed, training seeds, emulator revision,
# training schedule and fixed evaluation seeds. Only boss reward differs.
for arm in control shaped
    set -l weight 0
    if test $arm = shaped
        set weight 1
    end

    set -l prefix results/native-boss-$arm-50
    if test -f $prefix-report.json
        echo "[boss] $arm completed already; retaining $prefix-report.json"
        continue
    end

    set -l resume_flags
    if test -f $prefix.pt
        set -a resume_flags --resume
    end

    echo "[boss] starting $arm: weight=$weight rounds=50 episodes=32 workers=15"
    python -u tools/train_selfplay.py \
        --build --environment native-overgrowth \
        --initialize-from-model $initial_model \
        --workers 15 --rounds 50 --episodes 32 \
        --max-decisions 4096 --dimension 128 --hidden 32 \
        --seed 45 --learning-rate 0.001 \
        --temperature-start 0.05 --temperature-end 0.035 \
        --temperature-decay-rounds 30 --entropy-weight 0.002 \
        --boss-damage-weight $weight \
        --monitor-every 10 --monitor-seeds 64 \
        --evaluate-seeds 64 \
        --checkpoint $prefix.pt \
        --output $prefix-model.json \
        --report $prefix-report.json \
        $resume_flags \
        2>&1 | tee $prefix.log

    # Fish gives pipeline component statuses; tee succeeding alone is not enough.
    if test $pipestatus[1] -ne 0
        echo "[boss] $arm failed; preserve the checkpoint, log and .failure.json" >&2
        exit 1
    end
    echo "[boss] completed $arm; monitor=$prefix-report.monitor.jsonl"
end
echo "[boss] paired training completed. Compare fixed-seed monitor histories."
