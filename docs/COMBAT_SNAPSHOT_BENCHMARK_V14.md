# V14: phase-matched combat snapshots (pilot)

This benchmark diagnoses the combat head separately from the macro head while
preserving Act-1 progression. It is an **authentic-state pilot** using the
unchanged pinned emulator. It does not inject arbitrary decks, enemies, relics,
potions or chance streams.

## Core rule: build progress constrains the encounter

A snapshot is captured *after* the real run enters combat:

- **Early:** Act 1 floors 1–5.
- **Middle:** Act 1 floors 6–10.
- **Late:** Act 1 floors 11–16.
- **Weak:** the first three **ordinary combat entries**, counted in the
  collector's actual run history. These may be on later floors if the path
  avoided ordinary combat early.
- **Normal:** later ordinary combat entries.
- **Elite:** an authentic elite room (floors 6–14 under the native map rules).
- **Boss:** an authentic boss room on floor 16.

This is stricter than independently sampling a deck and an enemy. In
particular, boss fights use late-run decks that actually reached the boss.
An unusual but *reachable* weak fight late in the run remains recorded as
such. The floor bands describe build progress; the encounter tier describes
the game's encounter pool. The distinction is intentional.

Encounter type is resolved from the **public current map node**. The
pinned emulator encodes public `PrototypeRoomType` values as integers:
0=Combat, 1=Elite, 5=Boss, 6=Unknown (when resolved to combat).
The weak/normal split uses actual ordinary-combat order. These mappings
are specific to the pinned emulator and must be revalidated on upgrades.

## Persisted data and reproducibility

Each JSONL row stores:

- emulator gitlink revision and native environment;
- source policy, run seed, ascension, and ordered **legal action IDs** from
  run start to combat entry;
- public observation hash and exact-state hash for replay verification;
- act, floor, early/middle/late band, weak/normal/elite/boss tier, enemy IDs;
- public entry HP, deck size, relic/potion identities.

Private RNG and the full exact state are **never sent to the neural agent**.
Hashes and action prefixes are benchmark-side provenance only. JSONL recipes
are portable, unlike process-local fork handles.

During evaluation, the collector's **run prefix is replayed once per seed**.
For each selected combat entry, the evaluator verifies public and exact
fingerprints, then forks the exact state once for each combat agent.
Policies receive fair observations and legal actions only. The forks share
the same starting hidden state and RNG, although their random draw
consumption may diverge when they make different decisions.

The bridge must advertise the native reset; there is no legacy fallback.
Emulator hash mismatch, illegal replay action, or observation mismatch is an
explicit error. Censored combat loops are counted separately. No emulator
mechanics are changed. Neither fidelity against a clean game installation
nor a native RNG-law match is claimed.

## Pilot commands (Fish / WSL)

Start on the development branch and initialize the pinned submodule:

```fish
git fetch origin
git switch --track origin/agent/combat-snapshots-v14
git submodule update --init --recursive
source .venv/bin/activate.fish
python -m pytest -q tests/test_combat_snapshots_v14.py
```

Collect authentic builds using the round-120 v8 model as frozen
strategy+combat source. Collection executes complete runs only once:

```fish
python tools/combat_snapshot_benchmark.py collect \
  --collector-model results/ppo-v8-batched-40/models/stage-0120.json \
  --runs 100 --seed-prefix combat-snapshot-v14-source-A \
  --output results/combat-snapshots-v14/source-A.jsonl \
  --build
```

Compare two full phase-split checkpoints **only during combat**:

```fish
python tools/combat_snapshot_benchmark.py evaluate \
  --corpus results/combat-snapshots-v14/source-A.jsonl \
  --baseline-model results/ppo-v8-batched-40/models/stage-0120.json \
  --candidate-model results/ppo-enemy-attention-v13/r1/model.json \
  --report results/combat-snapshots-v14/v8-v13-r1.json
```

The candidate path above is illustrative; locate the actual exported v13
checkpoint in your results before running. `--build` is optional on the
second invocation if the pinned .NET bridge is already built.

The JSON report includes paired win/loss counts, uncensored tactical win-rate
differences, exit HP and potion deltas, metrics by **both encounter tier and
build-progress band**, and a 95% run-seed cluster bootstrap interval. Exit-HP
deltas are shown both overall and for both-win pairs to make selection effects
visible. The interval quantifies empirical sampling variation across
collector runs; it cannot remove bias from the chosen source strategy policy.

## Important limitations and v15 extension

1. **Selection effects:** naturally reached snapshots depend on the frozen
   source policy, and stronger policies change their future state
   distribution. Use multiple frozen collection policies and independent
   held-out seed cohorts before claiming general tactical improvement.
2. **Encounter coverage:** boss/elite snapshots can be rare. Inspect per-tier
   counts before interpreting the pooled result. Later add balanced weights
   and an explicitly separated stress suite.
3. **Chance replication:** v14 provides exact shared initial conditions.
   Repeated stochastic continuations from the same build + encounter require
   a documented RNG-branching capability, currently unavailable through
   the safe native bridge. Do not silently edit encounter RNG.
4. **Synthetic crossings:** combining a build with a different encounter
   needs legality checks for room, ordinal encounter bag, act region, relic
   hooks and map history. This is deferred until the emulator can expose a
   versioned, reproducible **benchmark-only** construction API.
5. **Capacity:** prefix replay is amortized across snapshots of each run;
   it is still more expensive than direct binary snapshot deserialization.
   Add the latter as a new versioned emulator interface when profiling shows
   the need, preserving the existing pinned benchmark.
6. **Weighting:** raw pooled scores reflect the snapshot corpus, including
   repeated combats per run. The report intentionally publishes tier and
   progress results separately. Natural versus balanced/stratified target
   distributions require explicit weights recorded with the corpus.

The first scientific question is whether the v13 enemy-attention head yields
a repeatable improvement *on the same authentic combat entries*, especially
multi-enemy combats, even though it did not improve overall Act-1 clears.
