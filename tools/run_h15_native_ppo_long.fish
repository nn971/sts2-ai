#!/usr/bin/env fish
# H15 research track A: long native A10 on-policy combat PPO.
# Executed on the user's CUDA machine, never by GitHub CI.
#
# Unlike fair H15 PUCT, this trains on actual source-emulator on-policy
# rollouts and does not need a public-history-conditioned chance sampler.
# Results are tagged to the immutable current emulator revision; changing
# that revision creates a NEW corpus and optimizer directory.

set -l script_dir (dirname (status --current-filename))
set -l ai (git -C $script_dir rev-parse --show-toplevel)
or exit 1
set -l emu (realpath $ai/../sts2-emulator)
if not test -d $emu
    echo "Expected emulator checkout alongside AI repo: $emu" >&2
    exit 1
end
set -l branch (git -C $emu branch --show-current)
if test "$branch" != research/h15-native-chance-law-20261030
    echo "Expected research/h15-native-chance-law-20261030; got '$branch'" >&2
    exit 1
end
set -l rev (git -C $emu rev-parse HEAD)
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

for required in $model $macro $bindings/sts2_emulator/__init__.py
    if not test -e $required
        echo "Required model/connector missing: $required" >&2
        exit 1
    end
end

cd $ai
or exit 1
mkdir -p $output

echo "[h15-native-ppo] emulator_revision=$rev"
echo "[h15-native-ppo] train=$train heldout=$heldout"
echo "[h15-native-ppo] experiment=$experiment rounds=120 workers=15"

# Fully rebuild the pinned emulator before collecting any source entries.
dotnet build $emu/src/Sts2Emulator.Cli -c Release
or exit 1
if not test -f $dll
    echo "Missing Release CLI DLL after build: $dll" >&2
    exit 1
end

# Focused existing PPO, corpus and sampling tests; no long H15 diagnostic loop.
python -m pytest -q tests/test_h10_native_transformer.py tests/test_h7_combat_ppo.py
or exit 1

# Exact, current-revision train and holdout corpora. Never relabel older
# h15-native-source-long recipes or mix their hashes into this experiment.
if not test -f $train
    python -u tools/collect_h10_native_combat_entries.py         --macro-model $macro         --combat-checkpoint $model         --emulator-root $emu         --emulator-revision $rev         --runs 256 --start-index 0         --seed-prefix h15-ppo-train-$tag         --device cuda --output $train         2>&1 | tee $output/collect-train.log
    set -l collector_status $pipestatus
    if test $collector_status[1] -ne 0
        echo "Native training corpus collection failed" >&2
        exit 1
    end
end
if not test -f $heldout
    python -u tools/collect_h10_native_combat_entries.py         --macro-model $macro         --combat-checkpoint $model         --emulator-root $emu         --emulator-revision $rev         --runs 64 --start-index 0         --seed-prefix h15-ppo-heldout-$tag         --device cuda --output $heldout         2>&1 | tee $output/collect-heldout.log
    set -l collector_status $pipestatus
    if test $collector_status[1] -ne 0
        echo "Native heldout corpus collection failed" >&2
        exit 1
    end
end

# Existing trainer checks revision, corpus disjointness, DLL/bindings and
# warm-start SHA before updating a parameter; each round persists progress.
set -l resume
if test -f $experiment/optimizer-checkpoint.pt
    set resume --resume
end

python -u tools/train_h10_native_combat.py     --train-corpus $train     --holdout-corpus $heldout     --initialize-from-model $model     --emulator-root $emu     --emulator-revision $rev     --emulator-dll $dll     --emulator-bindings $bindings     --output-root $experiment     --rounds 120     --native-per-round 64     --synthetic-per-round 32     --workers 15     --max-native-decisions 512     --device cuda     --actor-device cuda     --lr 0.00003     --temperature 0.85     --checkpoint-every 10     --execute $resume     2>&1 | tee -a $output/train.log
set -l training_status $pipestatus
if test $training_status[1] -ne 0
    echo "PPO stopped; inspect train.log and progress.json." >&2
    echo "If the pinned emulator revision and inputs are unchanged, rerun this script to resume." >&2
    exit 1
end
echo "[h15-native-ppo] complete=$experiment/models/player-r0120.pt"
