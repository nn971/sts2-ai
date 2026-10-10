#!/usr/bin/env fish
# Paired 256-seed selection, then 512 distinct seeds for locked final comparison.
# Execute at sts2-ai root with ppo-v8-batched-40 stage JSON models installed.
set -l base results/ppo-v8-batched-40/models
set -l out results/ppo-v8-checkpoint-selection-v11
for stage in 0040 0060 0100 0120
    if not test -f $base/stage-$stage.json
        echo "Missing checkpoint: $base/stage-$stage.json" >&2
        exit 1
    end
end

if not python -u tools/evaluate_act1_checkpoints.py \
    --mode selection \
    --model round40=$base/stage-0040.json \
    --model round60=$base/stage-0060.json \
    --model round100=$base/stage-0100.json \
    --model round120=$base/stage-0120.json \
    --seed-prefix ppo-v8-fresh-selection-v11 \
    --seeds 256 \
    --workers 12 \
    --output $out/selection-256.json
    exit 1
end

# Final model is already locked by the selection report.
# Only the winner and round-40 control are evaluated here.
if not python -u tools/evaluate_act1_checkpoints.py \
    --mode final \
    --selection-report $out/selection-256.json \
    --seed-prefix ppo-v8-fresh-confirmation-v11 \
    --seeds 512 \
    --workers 12 \
    --output $out/confirmation-512.json
    exit 1
end

echo "Completed checkpoint selection and independent confirmation:"
echo "  $out/selection-256.json"
echo "  $out/confirmation-512.json"
