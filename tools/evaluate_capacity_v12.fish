#!/usr/bin/env fish
# Unbiased architectural comparison: three pairs on identical fresh seed cohorts.
# Run after fish tools/train_capacity_v12.fish full all
set -l root results/ppo-capacity-v12
set -l output $root/evaluation
set -l models
for rep in r1 r2 r3
    for hidden in 32 64
        set -l file $root/$rep/h$hidden/models/stage-0080.json
        if not test -f $file
            echo "Missing full capacity checkpoint: $file" >&2
            exit 1
        end
        set -a models --model h$hidden-$rep=$file
    end
end

if test -f $output/comparison-256.json
    echo "[capacity] reusing completed 256-seed comparison"
else
    python -u tools/evaluate_act1_checkpoints.py \
        --mode comparison \
        --reference-label h32-r1 \
        $models \
        --seed-prefix ppo-capacity-v12-selection-fresh \
        --seeds 256 \
        --workers 12 \
        --output $output/comparison-256.json
    or exit $status
end

python -u tools/summarize_capacity_v12.py \
    --comparison $output/comparison-256.json \
    --training-root $root \
    --output $output/summary-256.json
or exit $status

# Predeclared second cohort (same six policies, never select lucky individual).
if test -f $output/comparison-512.json
    echo "[capacity] reusing completed 512-seed comparison"
else
    python -u tools/evaluate_act1_checkpoints.py \
        --mode comparison \
        --reference-label h32-r1 \
        $models \
        --seed-prefix ppo-capacity-v12-confirmation-fresh \
        --seeds 512 \
        --workers 12 \
        --output $output/comparison-512.json
    or exit $status
end
python -u tools/summarize_capacity_v12.py \
    --comparison $output/comparison-512.json \
    --training-root $root \
    --output $output/summary-512.json
or exit $status
echo "[capacity] finished paired 256- and 512-seed capacity analysis"
