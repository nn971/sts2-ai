#!/usr/bin/env fish
if test (count $argv) -lt 2
    echo "usage: ./scripts/write-manifest.fish EXPERIMENT_ID OUTPUT.json" >&2
    exit 2
end

python -m sts2_ai.cli manifest $argv[2] --experiment-id $argv[1]
