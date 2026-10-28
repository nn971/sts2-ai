# H8 packed Transformer inference: experiment and acceptance

Background: The 1,024-combat persistent-worker baseline on emulator M3
`33bf8cc0318292357ca00da73407b1f5335a9731` measured
994 completed steady-state fights in 67.184s = 14.795 combats/s. The
shared frozen 14,659,972-parameter Transformer spent 22.992s in
collation + host-to-device transfer, and 20.109s in forward +
output transfer. Across 15 workers, there were 26,855 measured actor
requests in 4,214 batches (6.37 requests/batch).

The **optional** packed actor collator aggregates heterogeneous public
object and legal-action token fields into 7 transfer buffers: object
metadata (kind/identity/zone/owner), action metadata
(kind/semantic/source/target), object numeric values, numeric
observation masks, object presence mask, legal action mask and phase.
It constructs no new Torch tensors per source object. It restores
the original `H7Batch` view fields without changing model parameters,
legal option ordering, object pointers, or stochastic controls.

Reference `collate_h7` and PPO gradient collection remain unchanged,
so this first experiment tests frozen inference serving in isolation.

## Preliminary local matrix (fish, no new worktree)

```fish
git -C ~/projects/sts2-emulator fetch origin
git -C ~/projects/sts2-ai fetch origin

# Deliberately use source-parity-verified M3, NOT experimental M4.
git -C ~/projects/sts2-emulator switch research/h8-tagged-boundary-cursor-20261028
dotnet build ~/projects/sts2-emulator/Sts2Emulator.sln -c Release -m:1

git -C ~/projects/sts2-ai switch research/h8-packed-actor-serving-20261028
cd ~/projects/sts2-ai
source .venv/bin/activate.fish
set emulator ~/projects/sts2-emulator
set dll $emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set revision (git -C $emulator rev-parse HEAD)

python -m pytest -q tests/test_h8_packed_actor_collation.py
python -u tools/benchmark_h8_actor_collation.py \
    --device cuda --batch 6 --objects 40 --actions 8 \
    --iterations 150 --warmup 12 \
    --output results/h8-packed-collation-microbench.json

# Legacy baseline: if the existing 1,024-source report uses this exact
# model, source digest, emulator revision, reference law, and replay
# configuration, reuse it instead of running this a second time.
python -u tools/benchmark_h8_h7_policy_pilot.py \
    --emulator-dll $dll --emulator-bindings $emulator/bindings/python \
    --emulator-revision $revision \
    --output results/h8-steady-state-1024-legacy.json \
    --order native --episodes 1024 --workers 15 \
    --warmup-per-worker 2 --max-boundaries 1200 --replay-every 16 \
    --sampled-mechanics-audit --measure-rpc \
    --actor-device cuda --actor-max-batch 16 --actor-delay-ms 3 \
    --actor-collation legacy

# Candidate, with every other source, actor and Nature parameter fixed.
python -u tools/benchmark_h8_h7_policy_pilot.py \
    --emulator-dll $dll --emulator-bindings $emulator/bindings/python \
    --emulator-revision $revision \
    --output results/h8-steady-state-1024-packed.json \
    --order native --episodes 1024 --workers 15 \
    --warmup-per-worker 2 --max-boundaries 1200 --replay-every 16 \
    --sampled-mechanics-audit --measure-rpc \
    --actor-device cuda --actor-max-batch 16 --actor-delay-ms 3 \
    --actor-collation packed

python tools/compare_h8_actor_collation.py \
    results/h8-steady-state-1024-legacy.json \
    results/h8-steady-state-1024-packed.json \
    --output results/h8-packed-actor-comparison.json
```

You may use the previously uploaded
`h8-steady-state-1024.json` as the legacy argument by copying it
into `results/` **without renaming or changing its contents**.
The comparator requires matching 1,024 source seeds, frozen checkpoint
SHA256, emulator DLL checksum/revision, source-level player and
event-scoped Nature decisions, and sampled replay. If your local
active emulator revision differs, use the exact baseline commit above.
A microbenchmark improvement alone is insufficient.

## Acceptance / next steps

- CPU unit suite must show exact object/action H7Batch tensor parity and
  frozen-policy logits/values for mixed combat and macro inputs; real
  source-valid 16-cell H8/H7 replay should pass with the packed actor.
- CUDA steady-state execution must produce 1,024/1,024 terminal combats,
  all 64 sampled replays, exact reference event and player choices,
  and *lower wall-clock* steady-state rollout duration.
- Review GPU collation, forward and inference batch counts. Small
  differences require alternating A-B-B-A repeats with the same pinned
  source bank/checkpoint. Keep actor `legacy` as default until evidence.
- Once the actor has a verified advantage, make the packed path default
  for frozen shared serving and investigate true dynamic worker sharding,
  larger batch windows, and an optional fused C# H8 transition/RPC.
  Those are independent optimizations and require their own exact
  source/decision parity validation.
