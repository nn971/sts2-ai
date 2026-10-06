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

This starter repository contains **research scaffolding, not a trained agent**. In particular:

- the emulator submodule is intentionally not embedded in this archive;
- the public Python-side emulator protocol is present so research code can be written against a stable boundary;
- a mock backend is used by tests;
- a small SQLite-backed strategic evidence store is included as a starting point, not as a final schema;
- search/model/training modules define interfaces and baseline plumbing rather than claiming strategic competence.

The immediate project dependency is **Milestone 1 of `sts2-emulator`: a trustworthy, practically fast whole-run emulator**.

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
