#!/usr/bin/env fish
# Regenerate public-only run traces for curated held-out cases.
# Run from the sts2-ai repository root after initializing its emulator.
set -l output results/ppo-v8-batched-40
if not test -f $output/models/stage-0020.json
    echo "Missing round-20 model at $output" >&2
    exit 1
end
if not test -f $output/models/stage-0040.json
    echo "Missing round-40 model at $output" >&2
    exit 1
end

python -u tools/replay_review_runs.py \
    --manifest experiments/v8_review_seeds.json \
    --model round20=$output/models/stage-0020.json \
    --model round40=$output/models/stage-0040.json \
    --output-dir results/v8-review-traces
