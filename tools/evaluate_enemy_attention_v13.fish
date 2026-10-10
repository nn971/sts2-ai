#!/usr/bin/env fish
# Compare three matched v8-width32 / v12-width64 / v13-attention32 replicas.
# All nine checkpoints are fixed before reading new evaluation outcomes.
set -l output results/ppo-enemy-attention-v13/evaluation
set -l models
for rep in r1 r2 r3
    for width in 32 64
        set -l baseline results/ppo-capacity-v12/$rep/h$width/models/stage-0080.json
        if not test -f $baseline
            echo "Missing v12 control checkpoint: $baseline" >&2
            exit 1
        end
        set -a models --model h$width-$rep=$baseline
    end
    set -l candidate results/ppo-enemy-attention-v13/$rep/attn/models/stage-0080.json
    if not test -f $candidate
        echo "Missing v13 attention checkpoint: $candidate" >&2
        exit 1
    end
    set -a models --model attn-$rep=$candidate
end

for cohort in 256 512
    set -l report $output/comparison-$cohort.json
    if not test -f $report
        python -u tools/evaluate_act1_checkpoints.py \
            --mode comparison \
            --reference-label h32-r1 \
            $models \
            --seed-prefix ppo-v13-attention-comparison-$cohort \
            --seeds $cohort \
            --workers 12 \
            --output $report
        or exit $status
    else
        echo "[v13] comparison-$cohort already completed"
    end
    python -u tools/summarize_enemy_attention_v13.py \
        --comparison $report \
        --output $output/summary-$cohort.json
    or exit $status
end
echo "[v13] fresh 256/512-seed attention capacity comparisons completed."
