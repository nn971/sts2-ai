# Desired Nature Training Implementation

**Implementation note (2026-10-14):** The original document below is the
long-range design, and its introductory status describes its 2026-10-12
proposal state. Work on [AI PR #71](https://github.com/nn971/sts2-ai/pull/71)
now implements **N0, a restricted N1/N2 isolated HP/intent bridge, and
N4 synthetic toy-gradient groundwork**. See
[runbook](NATURE_GROUNDWORK_RUNBOOK.md) for executable acceptance evidence
and [missing emulator interfaces](NATURE_EMULATOR_INTERFACE_GAPS.md)
([emulator issue #143](https://github.com/nn971/sts2-emulator/issues/143)).
Work on [AI PR #72](https://github.com/nn971/sts2-ai/pull/72)
adds **N3 isolated corpus collection**, a 15-process worker runner,
strict run/partition provenance, resumable records and deterministic
replay; see the [N3 runbook](NATURE_N3_CORPUS_RUNBOOK.md).
Full playable controlled combat, 15-worker **native emulator benchmark
acceptance**, original chance-law calibration, PPO Nature integration
and learned Nature training remain future stages.


**Status:** proposed `sts2-ai` implementation roadmap, documentation only; 2026-10-12.  
**Upstream dependency:** [`sts2-emulator` Desired RNG Implementation](https://github.com/nn971/sts2-emulator/blob/docs/desired-rng-implementation-v111/docs/DESIRED_RNG_IMPLEMENTATION.md), [emulator design PR #119](https://github.com/nn971/sts2-emulator/pull/119).  
**Development base:** v15q combat importance-sampling/diagnostics branch; do not reinterpret existing checkpoint fingerprints.  
**Scope:** single-player Silent combat first; later generalize to whole-run macro stochasticity. Multiplayer-only and cross-character mechanics are outside this initial milestone.

## 0. Decision and objective

The *emulator* implements deterministic game mechanics, correct trigger timing/order, legal random-choice boundaries, native streams and controlled-outcome resolution. The *AI repository* implements **Nature**, the external policy that selects outcomes at these boundaries, as well as learning objectives, distributions, diagnostics, search and experiment management.

We want one mechanics engine with interchangeable stochastic controllers:

- **Native:** emulator supplies source-authentic RNG outcomes; legacy v15 training/evaluation remains the control.
- **Scripted Nature:** AI supplies recorded/heuristic legal outcomes, initially without probability claims.
- **Learned Nature:** AI predicts a distribution over the choices offered at a chance boundary.
- **Enumerated Nature:** AI evaluates all outcomes at a small supported chance boundary for exact tests or search.

A Nature-selected outcome need not coincide with any particular native seed's next PRNG value. It must be *mechanically legal* and resolved through precisely the same game hooks. We do **not** require Nature to preserve a shadow native RNG cursor.

**Development order:** (1) interchangeability and mechanically sound rollouts; (2) source-law/conditional-probability audits; (3) importance-corrected original-game optimization; (4) optionally adversarial risk-sensitive optimization. Do not block (1) on (2).

Existing v15l whole-run PPO, v15m/n snapshot collection, v15o weighted snapshot sampling, and v15p/q diagnostics/ablation are separate baselines and must keep their original outputs and resume fingerprints.

## 1. Strict repository boundary

| `sts2-emulator` supplies | `sts2-ai` supplies |
| --- | --- |
| Versioned pending chance events, stable IDs, stream/call-site identity, ordered candidate support and outcomes | Nature selection policy and model inference |
| Normal native-mode `step` and opt-in controlled `advance`/`resolve`, fork/snapshot/restore | Driver/orchestrator, multi-worker scheduling, policy rollout |
| Outcome legality, continuation resumption, mechanical effects and event transcript | Proposal `q`, sampling seed, log `q`, training and evaluation |
| Optional audited reference law `p` with explicit fidelity/provenance status | Likelihood ratios only when the declared target law is justified |
| Same public player observation and legal-action contracts | Player-only public-state policy/value input and hidden-information audit |

**Never** reimplement game effects, move eligibility, shuffle/draw mutation or random-event timing in the Nature network. **Never** train or run a Nature model inside the emulator. Do not mutate a completed emulator transition to fabricate an alternative outcome.

An event's legal support may be determined using hidden internal game state; the controller may receive enough *chance-event metadata* to choose an eligible outcome, but the **player model** must receive only the ordinary public observation/history and legal player actions. Explicitly version Nature's information access; no unlabelled transfer of hidden state to a player policy or value target.

## 2. AI-side interfaces and types

Proposed package locations (adjust to existing module layout after integration):

```text
src/sts2_ai/
  emulator/controlled_chance.py       # capability negotiation, typed bridge, resume/fork
  nature/protocol.py                 # NaturePolicy + legal event/request types
  nature/policies.py                 # native / scripted / uniform / heuristic proposals
  nature/model.py                    # optional trainable conditional proposal network
  nature/trace.py                    # episode event records and log probabilities
  training/chance_importance.py      # independent correction/ESS math
  training/nature_objectives.py      # variance-target + KL-adversarial opt-in losses
  evaluation/nature_ablation.py      # native RNG reference and matched comparison
tools/
  inspect_nature_rollout.py          # trace/legality smoke
  benchmark_nature_workers.py        # CPU workers/transport/cost
  train_nature.py                    # only after learning gates
tests/
  test_nature_protocol.py
  test_chance_importance.py
  test_nature_gradient_reference.py
  test_nature_emulator_integration.py
```

A typed chance event should retain event ID, occurrence order, stream/site/owner, legal outcome IDs and optional probability-law record. The adapter must support operations with non-enumerated/structured support. First training experiments should use finite, audited choices only; reject unsupported law/operation combinations.

```python
# Sketch, not an API already implemented.
class NaturePolicy(Protocol):
    def propose(self, public_history, event: ChanceEvent) -> NatureProposal:
        """Return a legal outcome distribution or a deterministic legal choice."""

@dataclass(frozen=True)
class NatureProposal:
    outcome_id: str
    log_q: float | None                 # absent for pure scripted/no-law mode
    proposal_id: str                    # immutable version / policy hash
```

The *native* controller continues to be implemented by the emulator. The AI's "native" backend is an adapter which delegates resolution, **not** a Python recreation of xoshiro or other native streams.

### Exact provenance and transcripts

For every controlled chance decision record: game/emulator and chance schema versions, source snapshot/run ID, parent boundary fingerprint, event/site/stream identity, outcome ID, candidate-support fingerprint, Nature policy version, proposal sampling seed, optional `log p` / `log q` with target-law ID, and chosen result. Keep private simulator state in the worker, not in public PPO episode tensors.

Persist chance decisions interleaved with player decisions. Their order is part of replay identity. Reject missing/extra events, stale choices, schema drift, probability mismatches, and cross-worker handle reuse. Censored fights must remain censored rather than mislabeled defeats.

## 3. Baselines before any neural Nature training

Implement these interchangeable controllers using the same backend:

1. **Native delegation (`q=p` by definition of the selected engine law):** reference behavior; no control intervention.
2. **Scripted transcript:** chooses fixed allowed outcomes, with deterministic replay and first-divergence diagnosis.
3. **Uniform legal and adversarial heuristic:** controlled stress tests; no unbiased-training claims unless their `q` and a valid reference `p` are known.
4. **Finite exact enumeration:** exhaust a tiny event support and check the expected result of every continuation.

Start with the emulator's initial v111 migration seams: `run.niche` monster HP, `run.monster_ai` intent selection, and encounter-local formation choices; expand to random combat target/card draws **only when the emulator exposes those operations at the actual point of randomness**.

Existing combat-entry snapshots may already contain an established hidden shuffle. Changing a visible later draw by editing the draw pile is **not** equivalent to intervening at the earlier shuffle. For Nature experiments, fork at a legitimate pending chance boundary or generate new trajectories through an intercepted operation. Keep source-run-disjoint snapshot partitions.

**First structural success gate:** controlled single-player combat segments complete deterministically (or fail explicitly for unsupported events), using the same public player policy and mechanics; replay/fork and legal-menu traces match the forced outcomes. No probabilistic inference or Nature learning is necessary for this gate.

## 4. Player/Nature information and event probabilities

Separate three notions of "probability":

- **Snapshot selection target `p_c(i)`:** v15o's finite observed-corpus distribution, often progress/tier balanced. `q_c(i)` is its priority proposal. This is a **sampling design**, not the STS2 random-event law.
- **Event reference law `p_t(ω | context)`:** the target stochastic model for a controlled random-choice boundary. It must have an explicit model/version and audited conditional scope.
- **Nature proposal `q_{\phi,t}(ω | context)`:** the distribution used to choose the outcome in controlled rollouts.

Correct event-level `p` requires more than a matching PRNG algorithm. Event likelihoods depend on eligible pools, prior source consumption, hidden state, and possibly observations accumulated in the run. The player policy must not gain oracle access to the actual seed or hidden future draws.

Initially, allow mechanically valid **unweighted** structural experiments with `p` absent or tagged unverified. Defer claims of matching the true conditional game distribution until a separately documented source-law audit. No default fallback that silently labels uniform-over-candidates as real-game `p`.

For calibrated finite positive supports, use a full-support mixture

```text
q_phi = (1 - alpha) * p + alpha * q_learned,   0 <= alpha < 1
```

and compute ratios in log space. Reject events not covered by the chosen reference law. Reference-law identity (including its conditioning assumptions) is part of the immutable experiment fingerprint.

## 5. Original-game objective: importance-sampled chance outcomes

This phase **must wait for an explicitly justified reference law** (`p` may initially be a named synthetic game law used solely for correctness tests).

For a fixed player policy `pi_theta` and a sequence of controlled chance events, a valid likelihood ratio in the declared stochastic process is

```text
log W_chance = sum_over_events (log p_t(outcome_t) - log q_t(outcome_t)).
```

The estimator `E_q[W_chance * G] = E_p[G]` is valid for properly supported complete-trajectory random variables `G` under the stated factorization. It does *not* assert that a synthetic reference law equals native STS2 conditioned on public history. Use separate actor-policy corrections if actions were collected off-policy.

**First correctness oracle:** enumerate a tiny game and compare exact expected return and on-policy REINFORCE gradient with importance-corrected Monte Carlo estimates. Use *complete-trajectory* ratios initially; optimize estimators only after validating the identity. Preserve correct differentiation boundaries for Nature probabilities versus player gradients.

Existing v15o snapshot correction is a separate outer sampling stage:

```text
W_snapshot = p_c(snapshot) / q_c(snapshot)
W_chance   = product_t p_t(outcome_t) / q_t(outcome_t)
W_total    = W_snapshot * W_chance
```

Combine only when both proposal designs and desired target distribution are defined. In particular, `p_c` is an empirical *corpus target* and cannot be called natural game-state frequency. Do not silently reuse `w_i/T_i` from v15o as a complete chance-corrected per-decision policy gradient.

For PPO, clipped surrogates, self-normalized minibatches, advantage normalization, decision sampling, and truncated trajectories can introduce bias. Keep the unbiased finite-horizon reference path for regression; label PPO integration an **importance-weighted approximation** and compare its policy/critic gradients against the reference. Reject censored training episodes until a supported censoring method is implemented.

Report `log W` histograms, max absolute log ratios, mean/variance of weights, normalized ESS, sample failures, zero/invalid support, PPO clip fraction, advantage and gradient norms.

## 6. Trainable Nature policies and two objective families

### Mode I — Variance-minimizing Nature (preferred for ordinary-game strength)

Freeze the player initially and train a small event-conditional model of `q_phi`. On finite support, the ideal fixed-context proposal for a scalar/vector gradient sample `g(ω)` has

```text
q*(ω) proportional to p(ω) * ||g(ω)||
```

under the applicable single-event estimator assumptions. With stochastic future continuations use a conditional second-moment estimate; these formulas are *guides to proposal design*, not exact global solutions to a long-horizon recurrent game.

Recommended progression:

1. Fit a supervised/evaluable proposal for bounded **combat survival or exit-HP estimation**, with player frozen.
2. Measure held-out **variance or mean-squared error per simulator-second**, not just the frequency of defeats.
3. Estimate gradient-magnitude/second-moment priorities from independent rollouts; train `q_phi` with a separate optimizer and fixed-target intervals.
4. Alternate Nature update / frozen-Nature player rollout / player update. Store the proposal version used by each episode.
5. Re-evaluate and refresh Nature priorities as the player changes.

Use a small conditional object/event encoder before attempting a large second transformer. Prefer shared *public* features, but do not accidentally couple optimizer state or trainable modules with the frozen player in the first pilots.

### Mode II — KL-adversarial Nature (separate risk-sensitive objective)

For a validated or explicitly synthetic reference law, optionally minimize

```text
E_{pi,q}[R] + tau * D_KL(Q_{pi,q} || P_pi).
```

At a single finite chance event with continuation values `V(ω)`, the optimal local adversarial proposal is `q*(ω) ∝ p(ω) exp(-V(ω)/tau)`. Use this as a small-support analytic test/teacher. In a full game the relevant values depend on subsequent player decisions, chance events and state.

Keep risk-sensitive/adversarial checkpoints and metrics distinct from original-game importance-corrected training. Train/evaluate the final player against **native/random reference play** as a separate acceptance gate; improvement under aggressive Nature alone does not establish improvement in ordinary STS2.

## 7. Integration with current training and evaluation

- Reuse the **existing public-only v15 relational combat player**, existing frozen v8 strategy baseline, portable model/export and action binding. Nature is a new **opt-in rollout source**, not an implicit replacement for `train_pooled_objects_v15.py`.
- Reuse **v15m** source-run-disjoint snapshot manifests; **v15n** isolated native emulator worker pool; **v15o** explicit finite-corpus snapshot target/proposal; **v15p/q** stochastic paired evaluation, telemetry, and controlled A/B harness.
- Preserve all existing v15l/v15o/p/q checkpoint, corpus and report semantics. Introduce a separate Nature experiment config, new output directory and fingerprints. Do not migrate or overwrite running checkpoints.
- **Default: 15 rollout workers** for new full machine experiments, per current workflow preference; pilot 1/4/8/15 for deterministic correctness and measured throughput. Emulator handles belong to individual workers; Nature inference must be reproducibly seeded and independent of scheduling.
- For the first end-to-end test, run **fixed scripted Nature against a frozen v15 player** on a small supported Silent combat set. Compare with native resolution on public-legal action sequences; do *not* label the two reward populations statistically comparable until their probability law and coupling are defined.
- When calibrated, evaluate fixed-budget `q=p` control vs scripted proposal vs learned proposal, starting from identical player checkpoint/vocabulary/optimizer and disjoint source runs; distinguish efficiency of gradient estimation from raw game strength.
- Final model strength uses an untouched, native-RNG, player-visible-only, source-run-disjoint held-out evaluation plus fresh full-run seeds. Report survival, HP, potion usage, encounter tier/Act progress, full Act-1 win rate, sampling cost, ESS, clustered uncertainty. Real-game parity remains a separate qualification.

### Example experiment manifest (proposed; not yet implemented)

```yaml
experiment: nature-structural-pilot
mode: controlled-scripted             # native | controlled-scripted | calibrated-is | kl-adversarial
game_build: v0.111.0
emulator_revision: pinned-submodule-sha
chance_protocol_version: pending
reference_law_id: null                 # explicitly absent until calibrated
nature_policy_id: scripted-v1
player_model_sha256: exact-portable-model-hash
snapshot_manifest_sha256: exact-source-run-disjoint-manifest-hash
seed: 29
workers: 15
reward_schema: hp_preservation-v15
```

## 8. Work packages and acceptance gates

| Stage | Implementation in `sts2-ai` | Dependency / gate |
| --- | --- | --- |
| **N0 — Contract/spec** | Define Nature/trajectory types and manifest; pin emulator expected chance capability; keep old paths stable. | Emulator design D0; documentation/protocol tests. |
| **N1 — Bridge** | Opt-in controlled backend, event decoding, `advance/resolve` orchestration, handles, explicit failure modes. | Emulator D3/D4. Replay same scripted outcomes; no hidden player leak. |
| **N2 — Fixed Nature** | Native delegator, scripted, legal-uniform/heuristic, small finite enumerator. | Supported emulator chance call sites. Complete supported combat segments. |
| **N3 — Structured corpus** | Chance transcripts, source-run manifests, fork/replay audit, 15-worker determinism/perf. | N1/N2. Strict hashes and worker-equivalent transcripts. |
| **N4 — Toy probability oracle** | Pure `p/q` / `log W` / ESS maths plus exact finite-horizon expected-reward and policy-gradient tests. | Explicit toy `p`, not native calibration. Exact identities and stochastic convergence. |
| **N5 — Reference law gate** | Accept audited event probabilities with versioned conditioning assumptions; reject unverified or absent `p` in calibrated modes. | Emulator D6, source-law evidence. No silent probability fallback. |
| **N6 — Calibrated IS learning** | Separate exact-reference REINFORCE trainer then opt-in weighted v15 PPO and ablations. | N4/N5. Value/gradient tests and fixed-compute native evaluation. |
| **N7 — Learned Nature** | Event model, variance/gradient-signal predictor, alternating optimization, replay-version tracking. | N6. Held-out estimator-variance and sample-efficiency improvement. |
| **N8 — Adversarial / search** | Separate KL game training; finite chance tree enumeration / stochastic search integration. | N5 onward. Risk-sensitive metrics and native evaluation kept distinct. |

Each stage should be an independent PR where practical. Structure and replay (**N0–N3**) can progress while the emulator is still auditing native chance laws; toy math (**N4**) proceeds in parallel. Only probability-sensitive game-training stages (**N5–N8**) depend on calibration.

## 9. Proposed first PR sequence

1. **AI PR A — N0/N4 pure groundwork:** docs, typed `ChanceEvent`/`NaturePolicy`/`ChanceTrace`, synthetic toy event trees, exact ratio/gradient/ESS tests. No emulator submodule bump and no PPO changes.
2. **AI PR B — N1/N2 controlled bridge:** once emulator D3/D4 lands, repin emulator in a dedicated integration PR, implement controlled JSONL adapter, scripted Nature and deterministic fork/replay tests.
3. **AI PR C — N3 multi-worker corpus:** source-run-grouped parallel rollouts (15 workers supported), chance transcripts, strict manifests/resume, live progress, structural benchmarks.
4. **AI PR D — N5/N6 probability gate:** versioned conditional probability auditor and calibrated `p/q` reference training plus independent PPO integration tests. Do not claim native unbiasedness before the law is verified.
5. **AI PR E — N7/N8 experiments:** learned variance-minimizing Nature and a separate adversarial risk objective with matched natural-RNG holdouts.

## 10. Non-goals and blockers

- Do not change or restart v15l/v15o/v15p/v15q checkpoints merely to introduce the Nature abstraction.
- Do not assume all old snapshots permit changing an RNG event after it has occurred.
- Do not reimplement native RNG streams or mechanics in Python.
- Do not expose exact hidden emulator state/actual future RNG to the **player**.
- Do not treat legal Nature interventions or idealized uniform draws as native-calibrated probabilities.
- Do not combine risk-sensitive rewards with importance-corrected expected rewards under the same experiment identity.
- Do not assume an observed increase in adversarial difficulty is an increase in player strength.

**Success definition:** `sts2-ai` can run a frozen player through reproducible, mechanically coherent chance-intervened combat via the emulator's standardized boundary, while maintaining a separate native baseline. Once calibrated chance laws exist, it can test whether importance-driven Nature reduces estimator variance and improves native-RNG held-out performance per unit compute.
