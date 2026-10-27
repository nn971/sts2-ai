#!/usr/bin/env fish
# Matched v8 width32 vs width64 capacity experiment from the round-120 champion.
# fish tools/train_capacity_v12.fish pilot r1
# fish tools/train_capacity_v12.fish full all
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
        echo "Usage: fish tools/train_capacity_v12.fish [pilot|full] [r1|r2|r3|all]" >&2
        exit 2
end

set -l selection r1
if test (count $argv) -ge 2
    set selection $argv[2]
end
set -l replicas $selection
if test $selection = all
    set replicas r1 r2 r3
else if not contains -- $selection r1 r2 r3
    echo "Invalid replicate; choose r1, r2, r3, or all" >&2
    exit 2
end

set -l root results/ppo-capacity-v12
set -l champion results/ppo-v8-batched-40/models/stage-0120.json
set -l wide $root/warm-start-width64.json
if not test -f $champion
    echo "Missing round-120 champion: $champion" >&2
    exit 1
end
python -u tools/widen_phase_split.py \
    --source $champion --hidden 64 --output $wide
or exit $status

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
    for hidden in 32 64
        set -l warm $champion
        if test $hidden = 64
            set warm $wide
        end
        set -l output $root/$rep/h$hidden
        echo "[capacity] $rep width=$hidden through $total rounds; training seed=$seed"
        python -u tools/train_longrun.py \
            --warm-start $warm \
            --tactical-state-encoding enemy_instances \
            --ppo-backend batched \
            --rounds $total \
            --stage-size 40 \
            --episodes 32 \
            --workers 15 \
            --seed $seed \
            --dimension 128 \
            --hidden $hidden \
            --win-anneal-threshold 128 \
            --train-seed-prefix ppo-capacity-v12-$rep-train \
            --eval-seed-prefix ppo-capacity-v12-monitor \
            --eval-seeds 16 \
            --output-dir $output
        or exit $status
    end
end
echo "[capacity] $phase training completed for: $replicas"
