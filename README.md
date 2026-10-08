# sts2-ai

Research infrastructure for building an AI that can make strong **whole-run strategic decisions** in *Slay the Spire 2*.

The long-term objective is an agent that reasons jointly about:

- map/path choices;
- combat tactics;
- card rewards and skips;
- potion use and retention;
- shops and gold allocation;
- relic choices and boss relics;
- events and rest-site decisions;
- long-horizon deck construction and resource planning.

The project deliberately does **not** treat the game as a live UI to be controlled by a generic language model. Its foundation is a trustworthy, practically fast game model supplied by the sibling [`sts2-emulator`](../sts2-emulator) repository.

## Repository relationship

```text
sts2-ai/                       parent research repository
├── emulator/                  git submodule -> sts2-emulator
├── src/sts2_ai/               search, agents, models, data interfaces
├── configs/                   versioned experiment configurations
├── experiments/               experiment notes / manifests
├── data/                      manifests only; large datasets live elsewhere
└── docs/                      research and engineering design

sts2-emulator/                 independent mechanics repository
├── exact/forkable run state
├── legal actions
├── deterministic transitions
├── RNG semantics
├── native parity tooling
└── high-performance bindings
```

The dependency direction is one-way:

```text
sts2-emulator  <---  sts2-ai
```

`sts2-ai` may reason about the value of a decision. `sts2-emulator` only determines what the decision mechanically does.

## Current status

**Native-structure Overgrowth training gate (blocked on emulator interface):**
the six-floor Silent environment is no longer the default requested
experiment. `tools/train_selfplay.py` now selects `native-overgrowth`
and **fails closed** until the emulator JSONL server explicitly exposes
the existing C# native-overgrowth run factory. No emulator code has been
changed. Six-floor testing remains available via
`--environment legacy-prototype`; old checkpoints cannot resume
across modes. Act 1 progress for native-shaped runs is normalized on
16 floors rather than 6. See
[missing interface contract](docs/NATIVE_OVERGROWTH_TRAINING_GATE.md).


**Parallel, resumable neural self-play:** the teacher-free learner now supports
`--workers N` with independent pinned .NET emulators and actor RNG seeds
stable across worker scheduling. Each cohort is collected under one frozen
public-observation policy, then optimized once. Atomic round-complete
checkpoints retain AdamW and weights; `--resume` validates all training
hyperparameters and emulator revision. A Ryzen 9700X / RTX 5070 / 24 GB
pilot starts with four CPU rollout workers before using CUDA. See
[parallel neural training](docs/PARALLEL_NEURAL_TRAINING_9700X.md).


**Adaptive teacher-free neural learning (v2):** completed-run auxiliary
floor/HP rewards now persist until the agent actually starts winning,
then anneal by observed victories rather than by arbitrary elapsed
rounds. A full freshly collected on-policy cohort produces one
correctly accumulated gradient update, avoiding sequential stale
per-decision updates. Held-out evaluation compares initial neural
weights, trained neural, random and visible-only heuristic on paired
seeds with censored-aware Wilson win intervals. See
[first neural self-play](docs/FIRST_NEURAL_SELFPLAY.md).


**Teacher-free neural self-play pilot:** an observation-only small
policy–value neural agent now collects actual complete Silent runs
under ordinary emulator randomness, samples from its own action
softmax, and updates its policy with episodic REINFORCE and a learned
value baseline. Terminal win targets dominate with annealed
bounded floor/HP auxiliaries; capped unfinished runs are **censored**,
never mislabelled defeats. A dedicated CPU-Torch CI experiment saves
a portable model and held-out seed-paired neural-vs-random report.
No teacher, hidden seed, oracle MCTS or perfect chance posterior
is used. See [first neural self-play](docs/FIRST_NEURAL_SELFPLAY.md).


**Guarded first-reward conditional proposals:** the pinned emulator now
permits independent reward-stream proposals only while that stream
has never been consumed, on independent hypothetical Combat → Reward
transitions. The incremental joint particle filter can generate an
equal number of such proposals per existing particle and retain **all**
publicly compatible successors, automatically preserving likelihood-based
parent weighting. This is a finite empirical research model, **not**
a native RNG posterior or a guarantee against rare-reward collapse.
See [pristine reward conditioning](docs/PRISTINE_REWARD_POSTERIOR.md).


**Longer Overgrowth posterior benchmark:** a reproducible public-only synthetic
Silent trace compares retained full-RNG particle cohorts against bounded,
fresh whole-history joint rejection. Reports include survival by observed
decision, simulator and bridge calls, rejection attempts, collapse, and wall
time; CI archives the machine-readable profile. All probabilities remain
under a declared *experimental independent-stream prior*, not native STS2.
See [coupled belief benchmark](docs/COUPLED_BELIEF_BENCHMARK.md).


**Incremental joint posterior pilot:** a persistent finite cohort of
complete independently sampled hypothetical stream states is now filtered
in-place across observed public decisions. It preserves cross-stream RNG
correlations and avoids replaying the full run on every simulation.
Sampling is exact for the **realized empirical cohort**, not the
underlying full synthetic prior or native Slay the Spire 2. See
[incremental coupled particle belief](docs/INCREMENTAL_COUPLED_PARTICLES.md).


**Joint post-map history conditioning (synthetic independent-stream prior):**
`CoupledFactorizedHistoryRejectionSampler` now draws complete hypothetical
states from the exact factorized RunStart posterior and replays every
subsequent publicly observed action/observation/menu together, preserving
all six RNG streams and correlations. The accepted states are exact
conditional samples under the **declared alternate game prior**; finite
rejection caps fail explicitly. This is a correctness pilot, not a native
STS2 chance model. See [joint history conditioning](docs/COUPLED_FACTORIZED_HISTORY_REJECTION.md).


**Exact factorized RunStart posterior (alternate game prior):** a new
two-frame public-history sampler conditions independent map and combat
stream priors in `O(|M|+|C|)` rather than enumerating every pair,
while resampling the other four streams independently. An accepted
complete emulator state advances every RNG cursor through the ordinary
RunStart mechanics. Exactness is **only** for this explicitly declared
independent-stream model and the `RunStart → MapChoice` boundary.
See [factorized RunStart posterior](docs/FACTORIZED_RUNSTART_POSTERIOR.md).


**Exact enumerated full-history posterior pilot:** a declared finite,
uniform seed universe can now be exhausted and conditioned against every
public observation, player action, and complete legal menu. PUCT samples
independent exact posterior continuations **under that alternate finite
game prior**. This does not reconstruct the native 128-bit seed
distribution or generalize to arbitrary natural-game maps.
See [exact finite public posterior](docs/EXACT_ENUMERATED_PUBLIC_POSTERIOR.md).


**Local full-room combat RNG replay pilot:** the optional pinned
`prototype-local-combat-stream-condition-v1` bridge conditions an independently
sampled hypothetical *pre-entry* combat RNG stream on the player's entire
visible combat-entry frame, preserving that stream's shuffle/enemy-decision/
future-call cursor. It is explicitly **not** the native full-history posterior
because earlier use of the combat stream is not conditioned.
See [local combat-stream bridge](docs/LOCAL_COMBAT_STREAM_BRIDGE.md).


**Certified combat draw law:** a new opt-in, strict public-inventory
model for independently sampling/conditioning ordered draws from an
exchangeable unknown multiset, including a known top-card prefix and
empty-pile reshuffle. It has an opening-combat integration test against
the pinned emulator. This is **not** a full native-game RNG replacement
or a constructed emulator fair-continuation API. See
[certified combat draws](docs/CERTIFIED_COMBAT_DRAWS.md).


**Incremental fair-belief research:** a finite, independently search-seeded
hypothetical cohort can now be conditioned across public actions and reused
across stochastic PUCT simulations without restarting every candidate.
This is exact **for the cohort's empirical seed prior**, not for the
underlying game law, and fails explicitly if all candidates are eliminated.
The default agents remain unchanged. See
[finite-cohort posterior](docs/INCREMENTAL_FINITE_BELIEF.md).


**S1 research pilot:** `emulator/rejection.py` and `search/fair_replay.py`
now connect fair stochastic PUCT to independently seeded, complete
public-history-conditioned prototype emulator continuations. This is an
exact rejection reference **under a declared synthetic 128-bit seed prior**;
long histories have prohibitive acceptance rates and native STS2 chance
prior fidelity remains to be validated. See
[rejection-conditioned replay](docs/FAIR_REPLAY_CONDITIONING.md).
The default agents and oracle-exact UCT remain unchanged.


**New optional research components (PR #13):** a provenance-checked, aggregate
human card-reward prior (with missing-card/Skip fallback), and a mathematical
stochastic PUCT kernel over full public-history keys. These are independently
tested scaffolds, not a live fair-game agent. See
[human statistics prior](docs/HUMAN_CARD_PRIORS.md) and
[stochastic PUCT roadmap](docs/SELF_IMPROVING_STOCHASTIC_PUCT.md).


**Next development direction (early implementation in PR #12):** fair stochastic neural-guided PUCT and iterative self-improvement, with compact normalized progress/HP mean-and-variance predictions. See the [implementation roadmap](docs/SELF_IMPROVING_STOCHASTIC_PUCT.md).

This repository has **working experimental agents, but not a trained expert**.
The `emulator/` submodule is pinned to an implemented whole-run Silent prototype.
The long-lived Python/JSONL bridge, deterministic whole-run evaluation, seeded
baseline agents, transposition-aware **oracle-exact** MCTS, SQLite search-evidence
capture, and preliminary policy/value training are implemented.

A distinct observation-only `route` agent now performs visible-map DAG
lookahead. On an initial eight-seed paired benchmark it **underperformed**
the simpler `heuristic` baseline, so route planning remains opt-in. The
MCTS baseline is explicitly oracle-exact and must not be mistaken for a
fair player; fair stochastic search is a separate research milestone.

Quick reproducible commands (fish):

```fish
git submodule update --init --recursive
python -m pip install -e '.[dev]'
sts2-ai evaluate --agent heuristic --seeds 5
sts2-ai evaluate --agent route --seeds 5
sts2-ai benchmark --budgets --seeds 8 --no-include-random
```

See [current project status](PROJECT_STATUS.md),
[AI prototype roadmap](docs/AI_PROTOTYPE_ROADMAP.md), and
[route-planning baseline](docs/ROUTE_PLANNING_BASELINE.md).


## First-time setup

### 1. Create the two GitHub repositories

Create:

```text
YOUR_ACCOUNT/sts2-emulator
YOUR_ACCOUNT/sts2-ai
```

Publish `sts2-emulator` first.

### 2. Initialize this parent repository

Extract this starter project, then:

```fish
cd /path/to/sts2-ai
git init -b main
git add .
git commit -m "Initial sts2-ai research scaffold"
git remote add origin git@github.com:YOUR_ACCOUNT/sts2-ai.git
git push -u origin main
```

### 3. Add the emulator as a Git submodule

Once the parent has an `origin`, run:

```fish
./scripts/add-emulator-submodule.fish
```

By default this executes the equivalent of:

```fish
git submodule add ../sts2-emulator.git emulator
git submodule update --init --recursive
```

Then record the pinned emulator revision:

```fish
git add .gitmodules emulator
git commit -m "Add sts2-emulator submodule"
git push
```

See [`docs/SETUP.md`](docs/SETUP.md) for the complete workflow, including private-repository CI notes.

### 4. Create a Python environment

The project currently has no runtime dependency on ML frameworks. The base package uses the Python standard library; development tools are optional extras.

```fish
python -m venv .venv
source .venv/bin/activate.fish
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

Run the local checks:

```fish
./scripts/test.fish
./scripts/doctor.fish
```

## Research architecture

The long-term learning loop is intended to look like:

```text
                           ┌─────────────────────────────┐
                           │     sts2-emulator           │
                           │ exact / fast transition T   │
                           └──────────────┬──────────────┘
                                          │
                                          ▼
┌─────────────────┐             ┌──────────────────────┐
│ scenario archive │────────────▶│ targeted tree search │
└─────────────────┘             └──────────┬───────────┘
                                           │ expensive strategic evidence
                         ┌─────────────────┴─────────────────┐
                         ▼                                   ▼
              ┌────────────────────┐              ┌─────────────────────┐
              │ strategic evidence │              │ training corpus      │
              │ graph / cache      │              │ (O, pi_search, V)    │
              └──────────┬─────────┘              └──────────┬──────────┘
                         │                                   │
                         └─────────────────┬─────────────────┘
                                           ▼
                                  ┌──────────────────┐
                                  │ policy/value     │
                                  │ models           │
                                  └────────┬─────────┘
                                           │ guides
                                           ▼
                                    cheaper search
```

The project is designed so that simulation work can accumulate rather than be discarded: exact transpositions may be reused directly, while expensive search results become training targets that generalize to new states.

## Information discipline

The emulator may know the complete engine state, including hidden RNG. The strategic agent should normally receive a player-legitimate observation or belief state.

We therefore distinguish:

```text
EngineState       complete continuation-relevant state
Observation       information legitimately visible/derivable by the agent
BeliefState       distribution/representation of hidden possibilities
```

Oracle experiments that expose hidden RNG are useful as upper bounds, but must be labeled separately from fair-agent results.

See [`docs/INFORMATION_POLICY.md`](docs/INFORMATION_POLICY.md).

## Key documents

- [`docs/PURPOSE.md`](docs/PURPOSE.md) — research objective and non-goals
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — component boundaries
- [`docs/SETUP.md`](docs/SETUP.md) — complete Git/submodule/environment setup
- [`docs/EMULATOR_CONTRACT.md`](docs/EMULATOR_CONTRACT.md) — what this repo may assume from `sts2-emulator`
- [`docs/INFORMATION_POLICY.md`](docs/INFORMATION_POLICY.md) — engine state vs fair observation
- [`docs/SEARCH_AND_LEARNING.md`](docs/SEARCH_AND_LEARNING.md) — proposed expert-iteration loop
- [`docs/SELF_IMPROVING_STOCHASTIC_PUCT.md`](docs/SELF_IMPROVING_STOCHASTIC_PUCT.md) — **next implementation plan:** fair chance sampling, PUCT, bounded uniformized variance, annealed auxiliaries, and teacher-free self-improvement
- [`docs/NEURAL_POLICY_VALUE_PROTOTYPE.md`](docs/NEURAL_POLICY_VALUE_PROTOTYPE.md) — first neural training, MCTS deployment, and held-out experiments
- [`docs/TEACHER_QUALITY_EXPERIMENT.md`](docs/TEACHER_QUALITY_EXPERIMENT.md) — semantic teacher audit, exact-root shadow search, and Q-based policy targets
- [`docs/STRATEGY_DATABASE.md`](docs/STRATEGY_DATABASE.md) — persistent strategic evidence
- [`docs/DATASETS.md`](docs/DATASETS.md) — truth/search/training/scenario corpora
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — experiment identity and manifests
- [`docs/MILESTONES.md`](docs/MILESTONES.md) — staged research milestones
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — longer-term program
- [`docs/REFERENCES.md`](docs/REFERENCES.md) — design influences and primary references

## Development principle

A strategic result is only meaningful relative to the mechanics used to obtain it. Every experiment should therefore record at least:

```text
AI Git commit
emulator Git commit
game build
emulator schema/binding version
information policy
experiment config
random seeds
model/checkpoint identity
strategy-database snapshot (when applicable)
```

This is why the emulator begins as a pinned Git submodule rather than an implicitly moving dependency.
