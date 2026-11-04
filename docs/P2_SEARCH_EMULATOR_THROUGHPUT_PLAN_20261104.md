# P2 — Search/emulator throughput optimization: measured, parity-preserving plan

**Recorded:** 2026-11-04. **Status:** proposed and scoped; **no performance changes implemented by this document**.
**Project:** Silent Act 1, H18 A0 generated-map, with H16/H17 search and the emerging H21 public relational representation.
**Scope owners:** `sts2-ai` (PUCT, actor, codec, Python bridge, benchmarks); `sts2-emulator` (state handle lifetime, native/JSONL commands, C# hot paths).

## Objective and decision rule

Minimize end-to-end **wall time per equivalent learning/search experience**, not an isolated transitions/s counter. Prefer optimizations that remove repeated work rather than reducing precision or search depth.

Performance changes must preserve, on matched fixtures:
- the **same emulator player decisions**, legal actions, publicly observable state, terminal outcomes, action replay and source mechanics;
- the **same RNG/Nature continuation law and exact cursor/receipt** where the experiment requires exact replay;
- the **same PUCT root visits, teacher policy, child values, budget, depth, action tie-breaking, and terminal/cutoff semantics** under deterministic paired-seed checks, except for explicitly documented floating-point tolerance; and
- no hidden-state information entering the public policy, value network or shared search tree.

Do **not** score reducing PUCT simulations, changing loss/reward, lowering fidelity, suppressing action options, or dropping features as a throughput improvement. Preserve current H18/H16 and H21 as named controls.

**P0/P1 interaction:** P2 *profiling and isolated parity-tested optimizations* can begin now. The H21/H16 trainer integration and remaining P0/P1 acceptance gates are independent blockers for substantial new training. Keep P1 emulator power refactor in the pinned benchmark matrix to detect hidden performance regressions. Do not compare H16 model throughput directly against H21 encoder-only throughput.

## Source-level baseline (observed, not a performance measurement)

- `sts2-ai/src/sts2_ai/search/combat_puct.py`: PUCT retains visit/Q/statistical nodes in a history-keyed `table`, but its sampler begins/ends a simulation at each iteration and re-executes the selected action prefix. `depth=None` searches to combat boundary with a watchdog; selection actions may consume virtual, not physical, depth.
- `sts2-ai/src/sts2_ai/training/h16_joint_act_pilot.py`: `NativeRootConditionedSampler.begin_simulation` calls `backend.fork(self.root)`. `step` uses `batch_rollout_step_frame(((active, action),), ...)` on the fused path; the response is batched but ordinarily contains **one** state/action. `end_simulation` releases all accumulated handles. `_combat_frames` informs defeat-damage credit, so prefix reuse must retain that publicly derived history and score context.
- `sts2-ai/src/sts2_ai/emulator/jsonl_backend.py`: `_request` serializes one JSON message, flushes stdin, waits synchronously for one JSON response, deserializes and measures per-operation elapsed time. Existing `operation_profile`, `batch_step_frame`, `batch_rollout_step_frame`, `fork`, `release_many` provide instrumentation and transport foundations.
- `sts2-ai/src/sts2_ai/models/h16_batch_actor.py`: existing shared GPU actor records host collation, transfer and forward time, number of requests/batches and batch size; worker wait time is measured separately. P2 should add distributions and synchronization-aware timing, not assume GPU utilization implies CPU/actor bottlenecks.
- `sts2-emulator/src/Sts2Emulator.Cli/PrototypeAiJsonlServer.cs`: persistent per-process handles already exist; an in-process tree/prefix API is an *option to benchmark*, not an existing capability to assume.
- H21 v4's strict CPU codec+audit live sweep reported **232.54 decisions/s, p95 6.466 ms**, compared with **279.42/s, p95 5.033 ms** on an earlier v3 sweep; model forward and PUCT were excluded. These are contextual, non-equivalent-version measurements; no speedup/degradation conclusion until matched conditions.

## P2.0 — Reproducible profiling and measurement (prerequisite)

**AI owner:** add `tools/profile_p2_joint_throughput.py` (or similarly named CLI) and machine-readable `results/p2-*/profile.json`; add a focused CI smoke. Reuse existing bridge profiler, actor timing, existing H8 A/B/B/A benchmark conventions, and H21 live encoder profiling. Avoid modifying the original H18 train entrypoint for the first measurement if a standalone dry-run harness suffices.

Record at least:
1. **Whole run:** elapsed wall time; completed/censored runs, floors, combat roots, search sims, real transitions, decisions, updates; completed runs/s and transitions/s, reported separately.
2. **Search:** per root `simulations`, `depth`, path length, expanded nodes, unique path-prefix count, repeated edge traversals, forks/simulation, terminal/cutoff counts, value-leaf evaluations, cached-prefix hit potential and search-root wall share; mark root-conditioned versus independent continuation.
3. **Bridge/C#:** calls/bytes by op, fork/step/observe/release and fused/legacy timings, serialize/deserialize time, C# self-time (and allocated bytes if instrumented), live state-handle count and peak retained bytes; instrument counters conditionally, with negligible overhead when off.
4. **Actor/codec:** Python public JSON parsing, H16/H21 encoding and alias/identity audit, relation collation, queue delay, batch-size histogram, host→GPU transfer, forward+sync time, GPU memory peak; separate worker sums (overlapping) from critical-path wall time.
5. **Environment/provenance:** source SHAs of both repos, built DLL and model hashes, codec/object schema, public observation schema, RNG and Nature provider, map profile, ascension, search configuration, seed set, worker count, CPU/GPU and software versions, host contention.

**Matched workload matrix (staged to avoid over-testing):**
- warm-up and 2–4-seed smoke, then **8 fixed seeds** for diagnostic profiling (generated-map Silent A0; A10 fixed-route as an optional legacy control);
- PUCT 0, 16 and 64 simulations, full-combat mode and depth-8 mode recorded separately; same legal roots and frozen model per matched comparison; prioritize search-heavy modes once baseline quantifies their cost;
- **1 and 15 workers** for contention effects; shared CUDA actor where available, CPU-only deterministic CI smoke;
- freeze seeds, legal action selection, model/critic/H6C configuration and Chance/Nature provider. Keep completed vs censored events distinct.

First report must rank **top three measured critical-path contributors** with CPU, wall and memory evidence and estimate a *defensible upper bound* for each candidate optimization (Amdahl-type share). No production code change before this report except measurement plumbing.

**Gate P2.0:** repeatable baseline with identical replay/teacher checks; complete provenance; zero new failures on focused P0/P1 suites; explicit profile overhead documented.

## P2.1 — Tree-resident search states and prefix reuse (first high-impact candidate)

**AI:** retain an exact emulator state handle, public frame and legal-action menu for visited search-tree nodes/edges, keyed by **complete provenance**, not merely a public observation hash. On repeat traversal, fork from the verified retained prefix/child rather than replaying from the root; restore the publicly observable path needed for `enemy_hp_progress` and any leaf/terminal credit. Keep search-depth and virtual-card-selection accounting unchanged. Initial implementation should be **opt-in**, preserving old PUCT behavior as a gold control.

**Emulator:** investigate lightweight persistent state handles, fork-from-node, and explicit bounded `release_many`/teardown. Existing Python backend handles may suffice for a first experiment; avoid inventing a new RPC until measured necessary.

**Crucial stochastic split:**
- First support only the existing, clearly labeled **exact-root-conditioned Native RNG** sampler, with identical latent continuation/cursors for the retained prefix.
- For independently sampled `Nature` or fair-history-conditioned searches, share only public/statistical node metadata; do not reuse a hidden emulator continuation across independent samples. A later cache must key exact stochastic continuation identity, law/provider revision, receipts and cursor, **or** prove equivalence of the required conditional distribution. Absent such proof, bypass.
- Node equality must validate canonical exact state/RNG as well as legal player action and public observation. Do not key native state reuse on public history alone.

**Ownership/lifecycle:** explicit root lifetime; memory budget and LRU/epoch eviction; fork reused handles before action if mutable; always release on exceptions/cancellation; peak-handle and leaked-handle audits; no cross-root state leakage.

**Parity tests:** run old and new PUCT on identical real combat roots with **8, 16 and 64** sims, fixed root/actor/search seeds, full-combat and depth-8, including enemy summons, card selections, death/defeat and potion changes. At every sampled visited node compare exact-state hash, public frame hash, legal menu, RNG provenance, next action and outcome. For same seed, require exact PUCT root visit counts and teacher distribution; strong floating-point closeness for Q/value if numerical evaluation order changes. Validate on fresh seeds as well. Include tests that automatically bypass caches for incompatible Nature/provider variants.

**Gate P2.1:** no new gameplay/search drift; retained-handle count bounded; no leaks; **meaningful matched wall reduction**, provisionally `>= 20%` in the search-heavy cohort (a decision threshold, not an expected result). If improvement is smaller, report it and move to the next measured bottleneck.

## P2.2 — Reduce Python↔C# transport and observation cost

Start by profiling the already-fused `batch_rollout_step_frame` with one element against the old split calls, measuring Python JSON encode/decode, C# mechanics, DTO creation, JSON serialization, hashing, pipe wait and scheduling separately.

Experiments, one at a time:
- within one native worker, batch **independent ready states** across PUCT roots/rollout workers; preserve dependency ordering and action menus (don't batch sequentially dependent steps as if independent);
- introduce an optional versioned native **search tree/session** command if per-step round trips remain dominant, with strict ownership of legal-action choice and player model inference boundary; the C# side must never substitute an approximate policy;
- cache immutable public metadata / field-schema and reduce repeated construction; use compact projections/deltas only when public semantics and legal actions remain complete, and carry base version/hash for deterministic reconstruction;
- benchmark faster JSON paths or lower-copy frames only if byte/parse profiling identifies the cost.

**Gate P2.2:** identical public observation/action frames, exact state hashes and stochastic receipts for matched seeds, deterministic replay and source fidelity, no change to public/private information boundary. New protocol/schema feature flags isolate compatibility. Target `>= 15%` end-to-end matched benefit for acceptance of a new protocol path, or a measured removal of a documented dominant subcost.

## P2.3 — Actor/representation scheduling and reuse

Run a separate, pinned **H16 train path** profile and an **H21 opt-in codec/model** profile. The latter must wait for a valid H21 model with compatible checkpoint/schema and a verified joint-training integration before claiming whole-training speedups.

- Benchmark H21's identity caching, sparse single-field numerical encoding, attribute/graph-edge assembly, alias audit and collation on exactly the same public frames and action menus. CPU-only codec baseline and model-forward results must be reported separately.
- Measure actor queue wait p50/p95, arrived vs served microbatch sizes, GPU transfer synchronization, forward p50/p95 and CPU saturation at 1/8/15 workers. Sweep bounded delay `0, 1, 2, 4 ms` and batch limit `8/16/32`; include a no-actor-cost emulator control if available.
- Consider worker-side immutable-entity representation caching only if canonically invalidated by any visible card, relic, potion, map, event or persistent attribute change; never cache a full action menu keyed only by phase/floor. Prove exact tokens/edges/action logits under cache on/off.
- Use asynchronous pipelining only across **independent roots/workers**; preserve causal ordering per search path and actor request-response IDs.
- Reuse persistent worker pools across rounds **only if profiling** reveals meaningful initialization overhead; existing persistent C# workers mean per-step process startup is not the baseline bottleneck.

**Gate P2.3:** same inputs, action order, logits within GPU-nondeterminism tolerance, and replay; per-resource profiling plus matched end-to-end gain with peak-memory/p95-latency limits.

## P2.4 — C# mechanics and allocation optimizations (profile-driven)

Using the pinned `refactor/p1-enemy-effects-as-powers-20261104` branch and compatible fair-observation projection, profile the actual H18/H21 combat paths under allocation profiler and existing `H8HotPathProfilerV1`/semantic profiler, not only abstract microbenchmarks.

Candidate hotspots: repeated `RunState.Fork` deep copies; deck/card/enemy/power arrays and dictionaries; repeated public DTO projection; repeated JSON serialization and exact-hash computation; RNG cursor copies; temporary choice payloads. Prior H8 direct/native hot paths and the existing `benchmarks/Sts2Emulator.Microbench` provide useful controls.

Consider copy-on-write/structural sharing for **provably immutable** objects, safe pooled buffers and state-handle reference counting only with precise ownership. Prefer small patches and clear trace-level profiling before refactoring mechanics architecture. Keep original power definitions, trigger order and battle effect semantics unchanged.

**Gate P2.4:** exact/fork replay and failure-parity fixtures (including Poison/Weak/Vulnerable, summons, revivals and exceptional continuations); same output, zero new failures versus the *exact named* inherited core/fidelity failure sets; measured allocations/decision, bytes/step and end-to-end reduction.

## P2.5 — Controlled A/B/B/A acceptance and integration

- For each candidate optimization compare **A1 (baseline), B1 (change), B2 (change), A2 (baseline)** from immutable source SHAs and separate matched binaries; warm up each, avoid active competing jobs, preserve identical worker count and seed cohorts. Use a pilot first; confirm promising changes on an independent seed cohort and at search budgets 16 and 64.
- Enforce: same complete episodes/terminal outcomes, replay receipts, exact hashes and public frames where deterministic; same search-root teacher distributions/visits under frozen network; no censored/capped runs counted as successes; identical evaluator/model, chance law, PUCT budget and reward.
- Report both search-only and complete-run wall (including encoder/actor/transport); per-fork, RPC, C# allocation, GPU and retained-state memory; effect sizes and run-to-run variance. Distinguish true critical-path wall time from overlapping worker-summed durations. Report negative/neutral results.
- Advance only measured isolated improvements into the eventual P0/P1-integrated trainer; repeat a small integrated rollout smoke and actor/critic checkpoint compatibility gate after each merge. Do not start a new large H18/H21 learning study until P0/P1 combined model+emulator trainer gate passes.

**Acceptance:** defensible whole-workload improvement at equivalent experience with no search/fidelity drift. A practical optimization should normally show at least `~15–20%` repeated end-to-end wall-time reduction or remove a dominant cost supported by profiler evidence; don't hard-code this as a universal exclusion for small, valuable, low-risk fixes. If 50+ completed source combats/s or 50+ transitions/s is discussed, define the **unit and workload** explicitly; search transitions/s, combat clears/s and runs/s are different.

## Recorded implementation trackers

- [AI issue #166](https://github.com/nn971/sts2-ai/issues/166) — profile PUCT/bridge/actor, safe search-state reuse, A/B/B/A acceptance.
- [Emulator issue #240](https://github.com/nn971/sts2-emulator/issues/240) — C# fork/handle/JSONL profiling, native state cache support and mechanics/allocation parity.
- [Documentation PR #165](https://github.com/nn971/sts2-ai/pull/165) — this plan, stacked on the existing H21 P0 branch.

## PR/issue breakdown and stop/go rules

Suggested implementation slices:
- **AI P2.0:** profiler, deterministic rooted-search/actor/bridge metrics and report + tests.
- **AI+emulator P2.1:** opt-in retained search node-state handle cache and exact replay parity (separate C# change only when needed).
- **AI+emulator P2.2:** opt-in batch/tree transport protocol or public DTO optimizations.
- **AI P2.3:** H21 codec and shared actor performance with compatible inference model.
- **Emulator P2.4:** C# measured hotpath patches, isolated from the power semantics refactor.
- **AI P2.5:** unified A/B/B/A runner and final acceptance report.

At each stage, stop/rollback if exact replay, search teacher, legal options, RNG/Nature, public-information isolation or memory-leak tests fail. Propose a revised hypothesis from the profile instead of stacking unmeasured fixes. Keep check durations bounded and CI workflows scoped to changed paths.

## Sources and related work

Original user handoff: `sts2-h18-prioritized-research-handoff-2026-11-04.md`, **section C** and **Current priority queue**.
- [H18 runbook](https://github.com/nn971/sts2-ai/blob/research/h18-a0-generated-map-20261103/docs/H18_RUNBOOK.md)
- [H16 search sampler](https://github.com/nn971/sts2-ai/blob/fix/h21-grounded-macro-offers-20261104/src/sts2_ai/training/h16_joint_act_pilot.py)
- [PUCT core](https://github.com/nn971/sts2-ai/blob/fix/h21-grounded-macro-offers-20261104/src/sts2_ai/search/combat_puct.py)
- [JSONL backend](https://github.com/nn971/sts2-ai/blob/fix/h21-grounded-macro-offers-20261104/src/sts2_ai/emulator/jsonl_backend.py)
- [Shared actor](https://github.com/nn971/sts2-ai/blob/fix/h21-grounded-macro-offers-20261104/src/sts2_ai/models/h16_batch_actor.py)
- [H21 acceptance and codec costs](https://github.com/nn971/sts2-ai/blob/fix/h21-grounded-macro-offers-20261104/docs/H21_GROUNDED_MACRO_ACCEPTANCE.md)
- [Earlier A/B/B/A benchmark methodology](https://github.com/nn971/sts2-ai/blob/fix/h21-grounded-macro-offers-20261104/docs/H8_ABBA_EMULATOR_BENCHMARK.md)
- [Emulator parity-before-performance ADR](https://github.com/nn971/sts2-emulator/blob/refactor/p1-enemy-effects-as-powers-20261104/docs/adr/0003-parity-before-performance.md)
