# H8 M5: inline event-scoped Chance Law provider ticket

The source-valid 1,024-combat benchmark measured **33,372**
`h8_reference_law` JSONL round trips, totaling 23.33 aggregated
worker-seconds. The standalone law request is informational:
`H8CombatRuntimeV1.DescribePendingReferenceLaw()` describes the
already-pending deterministic event support and does not advance any
RNG/provider state.

The optional H8 status field `referenceLaw` is exactly the pending
object produced by that method when `include_reference_law=true`
is specified on an H8 begin, player action, intent resolution or draw
resolution. It is `null` at player/terminal boundaries. The standalone
`h8_reference_law` API remains available for external Nature,
verification and replay. Reproducible journals, selected physical
card IDs, event weights/conditioning, action menus and model inputs
are not changed by this optimization.

Python opt-in flag: `--inline-reference-law`. On each replay-audited
source, the first Nature pending event is also queried through the
standalone endpoint and compared for **exact object equality**. All
normal Nature decisions read the law directly from the transition
status, eliminating almost one RPC per stochastic draw/intent.

## Independent CPU acceptance

The dedicated `h8-inline-chance-law` workflow builds the paired
emulator and runs 16 real source-valid Ascension 10 combats, twice:
standalone event-scoped reference law and inline provider law.
It compares source ID, player actions, event IDs, Chance Law IDs,
physical selections, terminal outcome, HP, exact journals and sampled
replay. It also fails if the inline ticket was requested but absent.

## 1,024-combat CUDA experiment (fish)

```fish
git -C ~/projects/sts2-emulator fetch origin
git -C ~/projects/sts2-ai fetch origin
git -C ~/projects/sts2-emulator switch research/h8-inline-chance-law-20261028
dotnet build ~/projects/sts2-emulator/Sts2Emulator.sln -c Release -m:1

git -C ~/projects/sts2-ai switch research/h8-inline-law-pilot-20261028
cd ~/projects/sts2-ai
source .venv/bin/activate.fish
set emulator ~/projects/sts2-emulator
set dll $emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set revision (git -C $emulator rev-parse HEAD)

# Fixed actor collation: only vary the number of Chance Law RPCs.
python -u tools/benchmark_h8_h7_policy_pilot.py \
    --emulator-dll $dll --emulator-bindings $emulator/bindings/python \
    --emulator-revision $revision \
    --output results/h8-1024-standalone-law.json \
    --order native --episodes 1024 --workers 15 \
    --warmup-per-worker 2 --max-boundaries 1200 --replay-every 16 \
    --sampled-mechanics-audit --measure-rpc \
    --actor-device cuda --actor-max-batch 16 --actor-delay-ms 3 \
    --actor-collation legacy

python -u tools/benchmark_h8_h7_policy_pilot.py \
    --emulator-dll $dll --emulator-bindings $emulator/bindings/python \
    --emulator-revision $revision \
    --output results/h8-1024-inline-law.json \
    --order native --episodes 1024 --workers 15 \
    --warmup-per-worker 2 --max-boundaries 1200 --replay-every 16 \
    --sampled-mechanics-audit --measure-rpc \
    --actor-device cuda --actor-max-batch 16 --actor-delay-ms 3 \
    --actor-collation legacy --inline-reference-law

python tools/compare_h8_inline_law.py \
    results/h8-1024-standalone-law.json \
    results/h8-1024-inline-law.json \
    --output results/h8-1024-inline-law-comparison.json
```

The comparator requires identical emitter DLL checksum, source
selection, frozen policy, event-scoped Nature draws/weights,
trajectories and 64 replay audits before reporting an A/B speedup.
`emulator_rpc_calls.h8_reference_law` should fall from ~33k to
~64 (one verification query on each replay-audited source that actually
encounters Nature), with the total number of actual deterministic
Nature decisions unchanged.

**Do not conflate this with packed actor benchmarking:** use the
verified actor-only comparison on the M3 baseline for collation.
Only after both optimizations independently pass should they be
combined and validated against the same 1,024-source checkpoint.

This is a performance experiment, not a replacement of the
interchangeable provider interface. A small A/B speedup needs an
alternating A-B-B-A replication.
