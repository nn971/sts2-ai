#!/usr/bin/env fish
# Continue the already-completed batched v8 run without reinitializing optimizer.
# Usage: fish tools/continue_v8_batched.fish [80|100|120]
set -l rounds 120
if test (count $argv) -gt 0
    set rounds $argv[1]
end
if not contains -- $rounds 60 80 100 120
    echo "Choose a total round count: 60, 80, 100, or 120" >&2
    exit 2
end

set -l output results/ppo-v8-batched-40
set -l warm results/ppo-pr51-large/models/stage-0160.json
if not test -f $output/checkpoint.pt
    echo "Missing original checkpoint: $output/checkpoint.pt" >&2
    echo "Do not warm-start from stage-0040.json if you want exact PPO continuation." >&2
    exit 1
end
if not test -f $output/reports/stage-0040.json
    echo "Missing original stage-0040 report; cannot verify training history." >&2
    exit 1
end
if not test -f $warm
    echo "Missing original warm-start model: $warm" >&2
    exit 1
end

python -u tools/train_longrun.py \
    --warm-start $warm \
    --tactical-state-encoding enemy_instances \
    --ppo-backend batched \
    --rounds $rounds \
    --stage-size 20 \
    --episodes 32 \
    --workers 15 \
    --eval-seeds 128 \
    --win-anneal-threshold 128 \
    --train-seed-prefix ppo-v7-v8-slyfix-train \
    --eval-seed-prefix ppo-v7-v8-slyfix-eval \
    --output-dir $output
