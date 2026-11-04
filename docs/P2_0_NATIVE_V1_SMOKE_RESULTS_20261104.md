# P2.0 public-v1 native JSONL measurement — first transport smoke

**Date:** 2026-11-04. **Evidence level:** contract/integration smoke, not a matched
production search/learning throughput result. **Source:** [AI CI #41823319790](https://github.com/nn971/sts2-ai/actions/runs/41823319790),
artifact `p2-v1-native-transport-profile` / `profile.json`.

## Provenance and scope

- Emulator SHA `9cc7c4cd9b35881d06898fc304eac238166cd1bc`
  (P1 canonical-power runtime plus P2 opt-in C# profiler, PR #241).
- CLI DLL SHA-256 `bccee5ad603dd47ff6480b3b2cf8844c6d8f70007ddc8dc7438a8606d90bca93`.
- Public schema `prototype-ai-v1`; generated-structure Overgrowth Silent A0;
  seed `p2-native-v1-transport-smoke-0`, initial public action.
- Direct JSONL client, no H16/H21 model, no PUCT and no worker parallelism.
  Exact hashes, full legal-action payloads and public frames match across
  same-state forks, and profiling off/on runs match exactly. Native profile
  reports no remaining handles after `release_many` and a peak at least four.

## Observed opt-in C# operations

Numbers are from **one warmed-up control followed by one profiled case**,
so they are descriptive observations of this small exact workload. Nested
stages are **inclusive** and cannot be added to parent-operation totals.

| C# operation | Calls | Total time | Thread allocations |
| --- | ---: | ---: | ---: |
| `p2.jsonl.op.legal_actions` | 2 | 30.712 ms | 642,064 B |
| `p2.jsonl.op.reset_native_overgrowth` | 1 | 7.862 ms | 1,306,888 B |
| `p2.jsonl.op.batch_rollout_step_frame` | 2 | 6.651 ms | 713,344 B |
| `p2.jsonl.response` (nested response serialization) | 10 | 5.740 ms | 180,424 B |
| `p2.jsonl.op.exact_hash` | 3 | 4.592 ms | 490,360 B |
| `p2.fork.exact_hash` (nested fork stage) | 1 | 1.579 ms | 153,640 B |
| `p2.fork.run_state` (nested fork stage) | 1 | 0.055 ms | 11,648 B |
| `p2.rollout.serialize_observation` (nested) | 2 | 2.306 ms | 290,672 B |

In this single fixture, computing the exact hash inside `fork` took about
**29 times** the measured RunState fork copy itself (1.579 ms vs 0.055 ms).
This is a candidate for further profiling because the H16 Python `fork`
client returns only the handle, but a protocol change requires exact replay,
ownership and broader workload evidence under the *same schema*.

The uninstrumented control wall was **524.156 ms**, and the subsequent
instrumented repetition was **52.053 ms**. This difference is dominated by
cold-start / ordering effects. It is **not a profiling speedup**; randomized
or A/B/B/A repetitions are required to estimate instrumentation overhead.

## What this evidence cannot establish

- The recorded top three are **only** this public-v1 transport smoke's
  C# operations, not the search-heavy end-to-end critical-path top three.
- The frozen H16 neural workload currently uses `prototype-ai-v0` and a
  different emulator revision. Cross-version raw timings are not matched.
- C# thread-allocated bytes are allocations, not retained heap-byte counts.
- Still needed for P2.0: frozen-H16 8-seed 1/15-worker H18 profiles,
  actor/codec/bridge attribution, exact replay on the matching emulator,
  profile overhead control, and the first defensible ranked whole-workload
  bottleneck report. Do not enable P2.1 cache optimizations before that gate.

[Full plan](P2_SEARCH_EMULATOR_THROUGHPUT_PLAN_20261104.md) ·
[Runbook](P2_0_PROFILING_RUNBOOK.md) ·
[AI tracker #166](https://github.com/nn971/sts2-ai/issues/166) ·
[Emulator tracker #240](https://github.com/nn971/sts2-emulator/issues/240).
