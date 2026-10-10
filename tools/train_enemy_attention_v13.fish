#!/usr/bin/env fish
# Matched v13 enemy-attention vs v12 width32/64 controls.
# fish tools/train_enemy_attention_v13.fish pilot r1
# fish tools/train_enemy_attention_v13.fish full all
set -l phase pilot
if test (count $argv) -ge 1
    set phase $argv[1]
end
set -l total 40
switch $phase
    case pilot
        set total 40
    case full
        set total 80
    case '*'
        echo "Usage: fish tools/train_enemy_attention_v13.fish [pilot|full] [r1|r2|r3|all]" >&2
        exit 2
end
set -l requested r1
if test (count $argv) -ge 2
    set requested $argv[2]
end
set -l replicas $requested
if test $requested = all
    set replicas r1 r2 r3
else if not contains -- $requested r1 r2 r3
    echo "Use r1, r2, r3, or all" >&2
    exit 2
end

set -l champion results/ppo-v8-batched-40/models/stage-0120.json
if not test -f $champion
    echo "Missing original v8 round-120 champion: $champion" >&2
    exit 1
end
for rep in $replicas
    set -l seed 29
    switch $rep
        case r1
            set seed 29
        case r2
            set seed 53
        case r3
            set seed 71
    end
    set -l output results/ppo-enemy-attention-v13/$rep/attn
    echo "[v13] $rep attention32 through round $total; training seed=$seed"
    python -u tools/train_longrun.py \
        --warm-start $champion \
        --tactical-state-encoding enemy_attention \
        --ppo-backend batched \
        --rounds $total \
        --stage-size 40 \
        --episodes 32 \
        --workers 15 \
        --seed $seed \
        --dimension 128 \
        --hidden 32 \
        --win-anneal-threshold 128 \
        --train-seed-prefix ppo-capacity-v12-$rep-train \
        --eval-seed-prefix ppo-capacity-v12-monitor \
        --eval-seeds 16 \
        --output-dir $output
    or exit $status
end
echo "[v13] completed: $phase $replicas"
