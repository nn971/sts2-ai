# H8: profile native mechanics, observation construction, and state forks

This experiment builds on the paired 128-source A–B–B–A comparison:
optimized H8 produced about 6.31 complete combats/s versus 5.85/s for
the original H8. The next optimization needs an **internal breakdown**,
rather than estimating mechanics time by subtracting CUDA service totals
(which overlap concurrent workers).

## Instrumented emulator

Use `sts2-emulator` branch
`research/h8-profile-native-observe-20261028` (PR #215),
stacked on PR #214.

- Trusted card/potion execution passes the original immutable `RunState`
  to `PrototypeGameEngine.Step`, which already makes an isolated
  `RunState.Fork()`. The v111 externally checked API remains unchanged.
- Native end-turn execution likewise drops its redundant outer fork.
  The real engine continues to fork once before modifying mechanics.
- Public H8 player observation no longer computes the **unused**
  hidden-state canonical SHA, but still produces the exact source-visible
  observation hash, action IDs, action payloads, and observation data.
  Exact legacy snapshots remain available through the audit API.
- New opt-in `h8_perf_reset`, `h8_perf_read`, `h8_perf_stop` and
  `h8_perf_resume` JSONL operations report C# nested scope inclusive
  elapsed time, max latency, call counts and managed allocations.
  Counters are OFF unless explicitly enabled, and are independent per
  persistent .NET worker process.
- Timed categories: status projection, JSONL response and full request,
  H8 player/draw/intent advancement, player public observation,
  Chance Law, provider commit, native draw, native end-turn,
  legacy continuation projection, RunState/PlayerState fork and
  card/potion game engine step.
- A parent stage **contains** nested child stages; their time and
  allocated bytes must NOT be summed as disjoint shares of wall time.
  Sums across independent process roots are also *service time*, not
  the 15-worker end-to-end elapsed wall time.

## Profile the real H7 Transformer

With fish and existing repo layout:

```fish
git -C ~/projects/sts2-emulator fetch origin
git -C ~/projects/sts2-ai fetch origin

git -C ~/projects/sts2-emulator switch research/h8-profile-native-observe-20261028
dotnet build ~/projects/sts2-emulator/Sts2Emulator.sln -c Release -m:1

git -C ~/projects/sts2-ai switch research/h8-native-profiler-rollout-20261028
cd ~/projects/sts2-ai
source .venv/bin/activate.fish
set emulator ~/projects/sts2-emulator
set dll $emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set rev (git -C $emulator rev-parse HEAD)

python -u tools/benchmark_h8_training_iteration.py \
    --emulator-dll $dll \
    --emulator-bindings $emulator/bindings/python \
    --emulator-revision $rev \
    --output results/h8-hotpath-profile-128.json \
    --workers 15 --episodes 128 \
    --max-boundaries 1200 --replay-every 8 \
    --actor-device cuda --learner-device cuda \
    --profile-emulator --sampled-mechanics-audit --compare-legacy
```

The `emulator_hotpath_profile.operations` field in the report merges
all 15 .NET workers by stage name. Sampling **full replay audits** is
still performed for 16 source combats, but the H8 timing counters are
paused during these audits. The `--sampled-mechanics-audit` option
avoids serializing full mechanics fingerprints and materializing complete
decision journals on every *unaudited* source; it does not skip terminal
or source-law validation. The legacy H7 run still enforces **128/128
identical source decisions, chance laws, rewards and on-policy data**.

The profiler adds CPU/GC timing instrumentation and alters scheduling,
so do NOT use the profiled run as an unbiased throughput benchmark.

## Separately compare unprofiled throughput

Run again **without** `--profile-emulator`, and use an output filename
ending `-unprofiled.json` to measure production-style throughput.
For a clean before/after comparison of PR #215 against PR #214, use
the existing A–B–B–A harness after changing its default pinned revisions
explicitly to the two commits (new PR #215 SHA after CI vs
`2e28f983d01a5a1b69bd8ff7808f78af22d4d810`) and running
on the same workstation.

Follow-on engineering is conditional on the dominant measured stage:
- If `h8.fork.run_total` dominates, replace transition-wide deep
  copies with an H8-owned, copy-on-write mechanics cursor.
- If `h8.observe.public_frame` dominates, implement compact typed
  observations and avoid JSON-to-Python-to-PyTorch object conversion.
- If `h8.jsonl.response`/request handling dominates, fuse structured
  requests and response serialization, with separate replay checks.
- If CUDA forward/collation dominates, prioritize shared actor
  batching, packed tensors and persistent GPU execution.
