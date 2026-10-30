#!/usr/bin/env fish
# A10 combat PPO on authentic native-structure source runs.
# Run locally in the CUDA workstation's sts2-ai checkout.
# H15 fair PUCT and experimental Nature samples remain separate.

set -l script_dir (dirname (status --current-filename))
set -l ai (git -C $script_dir rev-parse --show-toplevel)
or exit 1
set -l emu (realpath $ai/../sts2-emulator)

if not test -d $emu
    echo "Expected emulator checkout alongside sts2-ai: $emu" >&2
    exit 1
end
if test (git -C $ai branch --show-current) != research/h15-combat-search-guided-20261029
    echo "Switch sts2-ai to research/h15-combat-search-guided-20261029" >&2
    exit 1
end
if test (git -C $emu branch --show-current) != research/h15-native-chance-law-20261030
    echo "Switch sts2-emulator to research/h15-native-chance-law-20261030" >&2
    exit 1
end
for repository in $ai $emu
    # A dirty tracked C# source could produce a DLL different from HEAD.
    # The PPO fingerprint pins DLL bytes; the source provenance must also
    # identify an exact committed source tree.
    if not git -C $repository diff --quiet
        echo "Tracked working-tree edits in $repository; commit or stash before a pinned PPO study" >&2
        exit 1
    end
    if not git -C $repository diff --cached --quiet
        echo "Staged edits in $repository; commit or stash before a pinned PPO study" >&2
        exit 1
    end
end

set -l rev (git -C $emu rev-parse HEAD)
or exit 1
set -l ai_rev (git -C $ai rev-parse HEAD)
or exit 1
set -l tag (string sub -s 1 -l 12 $rev)
set -l output $ai/results/h15-a10-native-ppo-$tag
set -l train $output/train.jsonl
set -l heldout $output/heldout.jsonl
set -l experiment $output/ppo-r120-n64-s32
set -l model $ai/results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt
set -l macro $ai/results/nature/h6c-macro-attention-r80.json
set -l dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set -l bindings $emu/bindings/python
set -l ai_pin $output/ai-revision.txt

for required in $model $macro $bindings/sts2_emulator/__init__.py
    if not test -e $required
        echo "Required warm start, macro model or bindings missing: $required" >&2
        exit 1
    end
end

cd $ai
or exit 1
mkdir -p $output
if test -f $ai_pin
    set -l original (string trim (cat $ai_pin))
    if test "$original" != "$ai_rev"
        echo "AI source revision changed since this experiment began." >&2
        echo "Original=$original current=$ai_rev" >&2
        echo "Use the original AI revision or a new study directory; do not silently resume." >&2
        exit 1
    end
else
    echo $ai_rev > $ai_pin
end

echo "[h15-native-ppo] ai_revision=$ai_rev emulator_revision=$rev"
echo "[h15-native-ppo] train=$train heldout=$heldout"
echo "[h15-native-ppo] experiment=$experiment rounds=120 workers=15"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available"'
or exit 1

# A focused gate. The previous 17/18 failure was an unrelated test
# using an unregistered RNG site, now fixed on the pinned research branch.
dotnet test $emu/tests/Sts2Emulator.Core.Tests -c Release \
    --filter 'FullyQualifiedName~H15ControlledCombatSessionV1Tests|FullyQualifiedName~RandomIndexProvidersV111Tests'
or exit 1
dotnet build $emu/src/Sts2Emulator.Cli -c Release
or exit 1
if not test -f $dll
    echo "Missing Release CLI DLL after build: $dll" >&2
    exit 1
end
python -m pytest -q tests/test_h10_native_transformer.py tests/test_h7_combat_ppo.py
or exit 1

# Collect fresh source-run-distinct train/holdout corpora with the
# *current exact emulator SHA*. Never retag older H15 recipes.
if not test -f $train
    python -u tools/collect_h10_native_combat_entries.py \
        --macro-model $macro \
        --combat-checkpoint $model \
        --emulator-root $emu --emulator-revision $rev \
        --runs 256 --start-index 0 \
        --seed-prefix h15-ppo-train-$tag \
        --device cuda --output $train \
        2>&1 | tee $output/collect-train.log
    set -l collection_status $pipestatus
    if test $collection_status[1] -ne 0
        echo "Train corpus collection failed: $output/collect-train.log" >&2
        exit 1
    end
end
if not test -f $heldout
    python -u tools/collect_h10_native_combat_entries.py \
        --macro-model $macro \
        --combat-checkpoint $model \
        --emulator-root $emu --emulator-revision $rev \
        --runs 64 --start-index 0 \
        --seed-prefix h15-ppo-heldout-$tag \
        --device cuda --output $heldout \
        2>&1 | tee $output/collect-heldout.log
    set -l collection_status $pipestatus
    if test $collection_status[1] -ne 0
        echo "Heldout corpus collection failed: $output/collect-heldout.log" >&2
        exit 1
    end
end

# The existing trainer checks corpus disjointness, pinned model SHA,
# emulator DLL/bindings and experiment fingerprint before any update.
set -l resume
if test -f $experiment/optimizer-checkpoint.pt
    set resume --resume
end
python -u tools/train_h10_native_combat.py \
    --train-corpus $train --holdout-corpus $heldout \
    --initialize-from-model $model \
    --emulator-root $emu --emulator-revision $rev \
    --emulator-dll $dll --emulator-bindings $bindings \
    --output-root $experiment \
    --rounds 120 --native-per-round 64 --synthetic-per-round 32 \
    --workers 15 --max-native-decisions 512 \
    --device cuda --actor-device cuda \
    --lr 0.00003 --temperature 0.85 \
    --checkpoint-every 10 --execute $resume \
    2>&1 | tee -a $output/train.log
set -l training_status $pipestatus
if test $training_status[1] -ne 0
    echo "PPO stopped. Inspect $output/train.log and the per-round progress.json." >&2
    echo "Rerun this script to resume ONLY on the unchanged pinned revisions." >&2
    exit 1
end

# Predeclared final endpoint versus frozen H9 R250; do not tune on
# heldout results or select the best intermediate checkpoint.
set -l final_model $experiment/models/player-r0120.pt
if not test -f $final_model
    echo "Training returned success without the final R120 checkpoint" >&2
    exit 1
end
for checkpoint in baseline final
    switch $checkpoint
        case baseline
            set -l candidate $model
        case final
            set -l candidate $final_model
    end
    set -l report $output/heldout-$checkpoint.json
    if not test -f $report
        python -u tools/evaluate_h10_native_combat_entries.py \
            --corpus $heldout \
            --macro-model $macro --combat-checkpoint $candidate \
            --emulator-root $emu --emulator-revision $rev \
            --workers 15 --combat-device cpu \
            --max-combat-decisions 512 --report $report \
            2>&1 | tee -a $output/evaluate.log
        set -l evaluation_status $pipestatus
        if test $evaluation_status[1] -ne 0
            echo "Training completed; heldout evaluation failed. See $output/evaluate.log" >&2
            exit 1
        end
    end
end
echo "[h15-native-ppo] complete=$final_model"
echo "[h15-native-ppo] heldout=$output/heldout-baseline.json,$output/heldout-final.json"
