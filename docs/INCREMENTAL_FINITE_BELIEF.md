# Incremental finite-cohort public posterior (experimental)

**Status:** opt-in correctness and throughput experiment, not a calibrated
full-game belief state. It is intended for single-player Silent and the pinned
prototype emulator, **not** native-game parity claims.

The previous [rejection reference](FAIR_REPLAY_CONDITIONING.md) independently
replays candidate seeds from RunStart for every PUCT simulation. This is
conditionally exact for its declared synthetic seed prior, but has acceptance
rate approximately equal to the probability of the entire public history.
In particular, a generated map can nearly identify a synthetic seed.

`FiniteSeedPosteriorSampler` reduces *repeated* work by constructing a cohort
of `M` independent hypothetical seeds **before consulting any hidden live
state**. It observes the public frame and rejects incompatible candidates.
For every actual visible action/observation pair, it advances **every surviving
candidate once**, checks the exact public observation and full legal-action
menu, and releases incompatible candidate states immediately. Each tree
simulation samples a surviving candidate uniformly with replacement and forks
its *hypothetical* exact state. The actual game's hidden state is never forked.

## Statistical contract

Let the initial iid hypothetical seeds be `U_1,...,U_M`, drawn from the
declared `belief-pool:<128-bit hex>` search-side ensemble. Conditional on the
generated cohort, define the discrete empirical prior

\[
  \widehat{P}_M = \frac1M\sum_{i=1}^M\delta_{U_i}.
\]

After conditioning on a complete public history `h`, the sampled posterior
is uniform on the surviving indices `i` with `H(U_i)=h`.

This is **exact for the versioned empirical finite-cohort prior, not exact for
the original 128-bit prior**. In particular:

- Finite pool approximation produces *sampling error* and often loses rare
  posterior alternatives entirely. Multiple repeated PUCT draws are iid
  *conditional on the fixed cohort*, but they are **not independent draws from
  the full underlying prior**. Do not label them as such in training.
- If no particle survives a visible transition, the sampler raises
  `ParticlePosteriorExhausted`, releases all owned states, and permanently
  closes. It does **not** replenish from the true hidden state, clone a
  survivor to invent support, or silently return an unrelated prior.
- Every PUCT simulation can reuse the cohort without replaying its root
  history, but conditioning a new *real* game action still requires stepping
  all surviving particles once. Stored particle states use memory proportional
  to the surviving cohort.
- Matching includes the information-policy ID, exact observation JSON, exact
  ordered legal-action menu, and the public chosen action, just like the
  rejection reference. Refactoring this equivalence needs an audited policy
  change to avoid hidden-information leakage.
- Once an independent hypothetical particle has been generated, forking it
  is safe within the **declared empirical prior**. Forking the *actual live
  game's state* would instead be oracle-exact and remains prohibited.

## Example research workflow

```python
import random
from sts2_ai.emulator import (
    FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend,
)
from sts2_ai.emulator.particle_posterior import FiniteSeedPosteriorSampler
from sts2_ai.search.fair_replay import FairReplayPuctAdapter
from sts2_ai.search.puct import StochasticPuct

policy = InformationPolicy(FAIR_POLICY_ID)
with JsonlEmulatorBackend() as backend:
    real = backend.reset("private-actual-run-seed")
    try:
        observation = backend.observe(real, policy)
        actions = tuple(backend.legal_actions(real))
        with FiniteSeedPosteriorSampler(backend) as belief:
            belief.initialize(
                observation, actions,
                search_rng=random.Random(123),
                cohort_size=256,
            )
            adapter = FairReplayPuctAdapter(
                backend, sampler=belief,
                goal="prototype-act1-clear-v1",
            )
            root = adapter.root(belief.public_history or (), actions)
            # Research only. Full-run PUCT may exceed this depth and
            # will explicitly fail rather than mislabel a cutoff.
            report = StochasticPuct(
                adapter, seed=456, max_depth=128
            ).search(root, simulations=8)
            print(belief.stats, report)
    finally:
        backend.release_many([real])
```

After an *actual* move, append only public information by calling

`belief.advance(action, new_public_observation, new_legal_action_menu)`.

The cohort's `public_history` then supplies the complete registered history
for the next PUCT root. Never call `advance` with an exact hidden state.

## Measurement and gates

- Rejection sampler: `C / p(h)` expected hypothetical run starts for
  `C` independent accepted continuations when `p(h)>0`.
- Incremental finite pool: `M` run starts initially, at most one emulator
  step per surviving seed after each *real* action, and `C` cheap independent
  forks per `C` PUCT simulations at a fixed current history.
- `ParticlePosteriorStats` reports initial candidates, survivors, empirical
  survival fraction, real-history conditioning steps, emulator transitions,
  and sampled forks. The survival fraction is not a proof of posterior
  coverage or an estimate of gameplay win probability.

Synthetic tests verify exact empirical conditional frequencies and retention
of correlated future outcomes; a pinned JSONL test covers independently
generated RunStart hypotheses, first map reveal, incremental filtering and
continued branch sampling. The previous rejection sampler remains unchanged.

**The next essential task** is to replace finite-cohort degeneracy with
mechanically justified conditional chance reconstruction (map/reward/deck
generation, correlated stream rules). The finite cohort is a useful low-floor
reference and performance control, not a complete solution to long-history
filtering, nor an appropriate source of full-law aux variance labels.
