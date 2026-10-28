# H8 M6: isolate metadata stride overhead in the packed actor

## Motivation and exact hypothesis

On the same M3 emulator `33bf8cc`, fixed 1,024-source model and
event-scoped ordinary Nature, frozen packed inference improved 994
steady-state combats from **67.184s (14.795/s)** to **60.326s
(16.477/s)**, but actor forward/output time increased from
20.109s / 4,214 batches (4.77 ms per batch) to
25.924s / 4,742 batches (5.47 ms per batch).
The increased batch count and service-time-per-batch both matter.
We do **not yet know** how much of the latter is GPU scheduling,
batch composition or tensor strides.

The packed collation creates `[batch, object_count, 4]` long metadata
and `[batch, action_count, 4]` long metadata, then supplies
`obj[..., i]` and `act[..., i]` as strided views with strides
`[object_count * 4, 4]`. The legacy version creates
individually contiguous `[batch, count]` metadata tensors.

M6 adds an opt-in `--actor-collation packed-contiguous`. It retains
**identical** Python source packing and seven host-to-device transfers
to `packed`, but immediately calls `.contiguous()` on the eight
metadata fields before inference. This deliberately adds device
copying overhead, which is measured as collation time, to isolate
whether contiguous metadata reduces model forward time. We compare:

- `legacy`: source reference tensor creation and original contiguous
  metadata.
- `packed`: seven packed transfers and strided metadata views.
- `packed-contiguous`: same seven packed transfers followed by
  eight metadata materializations on the execution device.

There is no new random sampling, new model configuration, change in
actor weights or order of actions. The experiment is isolated from
emulator M4/native-turn work and M5 inline Chance Law.

## A: tensor-layout microbenchmark (fish, RTX 5070)

```fish
git -C ~/projects/sts2-ai fetch origin
git -C ~/projects/sts2-ai switch research/h8-contiguous-actor-ablation-20261028
cd ~/projects/sts2-ai
source .venv/bin/activate.fish

python -m pytest -q tests/test_h8_packed_actor_collation.py
python -u tools/benchmark_h8_actor_tensor_layout.py \
    --device cuda --batch 6 --objects 40 --actions 8 \
    --width 384 --heads 8 --layers 6 --feedforward 1536 \
    --iterations 100 --warmup 15 \
    --output results/h8-three-layout-microbench.json
```

The microbenchmark runs `legacy, packed, packed-contiguous,
packed-contiguous, packed, legacy` in one persistent Torch process
and reports synchronized median collation, forward/output and combined
service per batch. `max_abs_model_output_diff` reports numerical
variation; input tensors are tested for bitwise equality and model
outputs are compared with a strict small tolerance on CUDA. Use
`--checkpoint path/to/frozen.pt` to replace the fixed-seeded
14.66M-parameter synthetic checkpoint with a trained checkpoint if
appropriate. Microbenchmark results are *not* combat throughput.

## B: six interleaved, full A10 source-valid cohorts

```fish
git -C ~/projects/sts2-emulator fetch origin
git -C ~/projects/sts2-emulator switch research/h8-tagged-boundary-cursor-20261028
dotnet build ~/projects/sts2-emulator/Sts2Emulator.sln -c Release -m:1

set emulator ~/projects/sts2-emulator
set dll $emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set revision (git -C $emulator rev-parse HEAD)

set layouts legacy packed packed-contiguous packed-contiguous packed legacy
for i in (seq 6)
    set layout $layouts[$i]
    python -u tools/benchmark_h8_h7_policy_pilot.py \
        --emulator-dll $dll \
        --emulator-bindings $emulator/bindings/python \
        --emulator-revision $revision \
        --output results/h8-layout-$i-$layout.json \
        --order native --episodes 1024 --workers 15 \
        --warmup-per-worker 2 \
        --max-boundaries 1200 --replay-every 16 \
        --sampled-mechanics-audit --measure-rpc \
        --actor-device cuda --actor-max-batch 16 \
        --actor-delay-ms 3 --actor-collation $layout
end

# Compare first three cohorts, then the reverse-ordered control block.
python tools/compare_h8_actor_layouts.py \
    results/h8-layout-1-legacy.json \
    results/h8-layout-2-packed.json \
    results/h8-layout-3-packed-contiguous.json \
    --output results/h8-layout-abc.json

python tools/compare_h8_actor_layouts.py \
    results/h8-layout-6-legacy.json \
    results/h8-layout-5-packed.json \
    results/h8-layout-4-packed-contiguous.json \
    --output results/h8-layout-cba.json
```

Every comparison checks the same emulator build, model checkpoint,
all source seeds and cells, terminal states, complete player-choice
sequences, event-scoped Nature selections/laws, sampled replay results,
GPU actor batch configuration and device. A normal GPU throughput
comparison **rejects censored encounters**. CPU CI may use
`--allow-censored` only to check source replay integrity, and reports
`complete_combat_throughput_accepted=false` when applicable.

For each cohort, inspect `forward_ms_per_batch` together with
`mean_batch_size`, per-batch collation, total actor service, and 994
complete-combat wall time. A lower forward cost per batch can still
lose if contiguous copying or smaller batches outweigh the benefit.
Keep whichever mode improves repeated *end-to-end* throughput;
preserve the other modes as controlled baselines.

## Correctness and performance gates

1. Field-by-field exact tensors across three variants (including
   object owners, action source/target pointers and masks).
2. Frozen Transformer logits and values equivalent on CPU and CUDA;
   same player/Nature decisions for the full 1,024-source bank,
   64/64 sampled replays, and 1,024/1,024 terminal outcomes.
3. A-B-C-C-B-A comparison showing stable direction of forward cost
   and combat throughput, independent of order or workload skew.
4. Do not promote `packed-contiguous` to default on an isolated
   synthetic microbenchmark result or a single 1,024-source speedup.
