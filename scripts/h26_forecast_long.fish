#!/usr/bin/env fish
# H26-C/D: one six-hour Silent A0 run with source-derived intent + damage.
# By default, ONLY validates the setup and prints a dry-run fingerprint.
# Invoke with "start" to train, or "resume" after an interrupted round.
set -l mode prepare
if test (count $argv) -gt 1
    echo 'Usage: fish scripts/h26_forecast_long.fish [prepare|start|resume]' >&2
    exit 2
end
if set -q argv[1]
    set mode $argv[1]
end
if not contains -- $mode prepare start resume
    echo 'Usage: fish scripts/h26_forecast_long.fish [prepare|start|resume]' >&2
    exit 2
end

set -l ai (realpath (dirname (status filename))/..)
set -l emu $HOME/projects/sts2-emulator
if set -q STS2_EMULATOR_ROOT
    set emu $STS2_EMULATOR_ROOT
end

if not test -d $emu
    echo "Missing emulator directory: $emu" >&2
    exit 2
end
if not test (git -C $ai branch --show-current) = research/h26-search-free-tactical-20261107
    echo 'Expected AI branch research/h26-search-free-tactical-20261107' >&2
    exit 2
end
if not test (git -C $emu branch --show-current) = research/h26-public-intent-law-20261107
    echo 'Expected emulator branch research/h26-public-intent-law-20261107' >&2
    exit 2
end
set -l ai_dirty (git -C $ai status --porcelain --untracked-files=no)
set -l emu_dirty (git -C $emu status --porcelain --untracked-files=no)
if test (count $ai_dirty) -gt 0
    echo 'AI sources contain uncommitted changes; commit or restore before a long run.' >&2
    exit 2
end
if test (count $emu_dirty) -gt 0
    echo 'Emulator sources contain uncommitted changes; commit or restore before a long run.' >&2
    exit 2
end

set -l rev (git -C $emu rev-parse HEAD)
set -l ai_rev (git -C $ai rev-parse HEAD)
set -l catalog $ai/results/h26-public-intent-catalog-v3.json
set -l out $ai/results/h26-a0-forecast-6h-v1
set -l h22 $ai/results/h22-h21-packed-a0-long-v1/models/joint-r0095.pt
set -l prior $ai/results/h25-a0-combat-first-8h-v2-p64/models
set -l log $ai/results/h26-a0-forecast-6h-v1.log
for required in $h22 $prior/macro-r0003.pt $prior/combat-r0003.pt
    if not test -f $required
        echo "Missing initial model: $required" >&2
        exit 2
    end
end

cd $ai
or exit 2
mkdir -p results
if test $mode = resume
    if not test -f $catalog
        echo "Cannot resume without the original source catalog: $catalog" >&2
        exit 2
    end
    if not test -f $out/progress.json
        echo "Cannot resume without committed progress: $out/progress.json" >&2
        exit 2
    end
else
    # Generate v3 source rules once. Preserve this file unchanged for resume.
    set -l pending $catalog.pending
    if not dotnet run --project $emu/src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj -c Release -- public-intent-catalog-v111 > $pending
        rm -f $pending
        echo 'Could not export the v3 public source rule catalog.' >&2
        exit 2
    end
    mv -f $pending $catalog
    or exit 2
end

if not python -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x.get("schema") == "sts2-h26-public-intent-rules-v3"; assert x.get("enemies") and x.get("damage_powers")' $catalog
    echo 'Invalid v3 intent + damage catalog.' >&2
    exit 2
end
if not python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
    echo 'PyTorch CUDA check failed.' >&2
    exit 2
end

set -l args \
    --initial-h22 $h22 \
    --initial-h25-macro $prior/macro-r0003.pt \
    --initial-h25-combat $prior/combat-r0003.pt \
    --intent-catalog $catalog \
    --require-forecast-features \
    --emulator-root $emu \
    --emulator-revision $rev \
    --output-root $out \
    --rounds 200 \
    --runs-per-round 64 \
    --heldout-runs 16 \
    --workers 15 \
    --study-seed 20261120 \
    --max-wall-seconds 21600 \
    --max-combat-samples 4096 \
    --minibatch 8 \
    --epochs 1 \
    --lr 0.00001 \
    --temperature 0.85 \
    --keep-last-rounds 5 \
    --archive-every-rounds 20 \
    --device cuda

echo "AI commit:       $ai_rev"
echo "Emulator commit: $rev"
echo "Intent catalog:  $catalog"
echo "Output root:     $out"
echo 'Budget: 6 hours active, stopping after a committed round.'
echo 'Features: public exact-conditional intent + source gross-damage forecast.'
echo 'Search: disabled. Runs: 64 training + 16 held-out, 15 workers per round.'

if test $mode = prepare
    echo 'PREPARE ONLY: verifying fingerprint; no episodes will run.'
    python -u tools/train_h26_searchfree_a0.py $args
    exit $status
end
if test $mode = start
    if test -e $out/progress.json
        echo 'Output already contains committed progress. Use resume.' >&2
        exit 2
    end
    python -u tools/train_h26_searchfree_a0.py $args --execute 2>&1 | tee $log
    set -l exit_code $pipestatus[1]
    exit $exit_code
else
    python -u tools/train_h26_searchfree_a0.py $args --resume --execute 2>&1 | tee -a $log
    set -l exit_code $pipestatus[1]
    exit $exit_code
end
