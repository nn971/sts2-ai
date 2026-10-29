# Hierarchical MuZero-inspired architecture for STS2

**Status: accepted intended architecture (2026-10-29); design decision, not a completed implementation.**
**Owner:** `sts2-ai`. **Game-mechanics authority:** `sts2-emulator`.
**Initial objective:** single-player Silent, Ascension 10, Act 1 Overgrowth clear; extend to whole-run win as reliable coverage permits.

This is the canonical architecture target. It supersedes the H12 standalone
joint-reward predictor and H13 predictor-first/map-override *plans*, and the
earlier variance-only stochastic PUCT architecture as the primary development
order. Keep historical code, checkpoint fingerprints, journals and negative
results for reproducibility; **do not train, deploy, or merge the old
H12 reward-contaminated predictor as the new planner**. H13's public-state
boundary instrumentation and observed run journals may be reused after
explicit semantic review. No trained model is claimed to implement this design.

## Design in one diagram

```text
                         FAIR PUBLIC RUN STATE
                                  |
                  +---------------v----------------+
                  | Macro Transformer / encoder    |
                  |                                |
                  | policy head: macro legal moves |
                  | value head: V_m(run state)     |
                  | dynamics head: OPTIONAL        |
                  +----+-----------+---------------+
                       |           |
              map/shop/reward    combat room
               macro action         |
                       |       +---v-----------------+
                       |       | Combat Transformer  |
                       |       | policy head pi_c    |
                       |       | tactical value V_c  |
                       |       +---+-----------------+
                       |           |
                       |      Exact emulator mechanics
                       |      and valid Chance Laws
                       |           |
                       |      Combat ends / reward offer generated
                       |           |
                       +-----------v----------------+
                                   |
                         MACRO DECISION STATE
                       (including reward offers)
```

The macro **policy and continuation-value heads share a representation and
long-term objective**. The combat **policy and tactical-value heads form a
separate tactical model**; their terminal boundary value is supplied by the
macro continuation critic. There is no mandatory third neural model.

This borrows AlphaZero's shared policy/value heads and search-policy
distillation, and Stochastic MuZero's separation of **agent action**, **chance
outcome**, and **resulting decision state**. We retain an explicit emulator
and legal public states rather than learn an unconstrained latent game engine.

## Macro-combat hierarchy and value semantics

- A complete combat is a temporally extended macro transition (semi-Markov
  option): the selected combat policy executes individual combat decisions.
  The post-combat macro boundary is the first **reward decision state** after
  the emulator resolves combat and generates visible offers, or a terminal
  defeat. The macro policy then chooses the actual reward actions.
- For a fixed combat policy version `pi_c`, a macro combat option has
  a transition kernel `K_pi_c(s_next, reward, duration | s, option)`.
  The macro actor selects actions, never stochastic consequences. The
  random outcome is sampled/integrated according to admissible game laws.
- The macro critic `V_m(s)` evaluates *complete persistent public run state*:
  HP, deck, potions, relic identities and persistent counters, gold, map,
  rewards currently offered, and other relevant legal mechanics.
- The combat critic `V_c(x)` evaluates intermediate **observable** combat
  states. At combat termination, its bootstrap target is the value of the
  resulting macro boundary state `V_m(s_exit)`, with immediate rewards or
  discounting accounted for exactly once. Use `gamma^duration` if gamma
  is not one. Both critics ultimately optimize the same actual run objective.
- The critic, rather than a hand-tuned scalar mixture of HP/potion/reward
  bonuses, determines whether to conserve a potion, delay for The Hunt,
  or enter the next fight with a particular relic counter residue.
- Persistent relic counters and all source-visible resource semantics must be
  represented faithfully. The current object codec and H12 resource snapshots
  omit or collapse some of these; fix the observation/representation before
  using the new critic.

## Optional policy-conditioned combat dynamics

An **optional head** sharing the macro encoder may predict a *joint
distribution* of combat-caused consequences conditioned on a fully realized
(or **legally sampled**) public combat-entry state and a pinned combat policy
identity. Example output targets:

- survival and **survival-conditional discrete final HP**;
- explicit potion uses/consumption, preserving item identity;
- persistent counter residues and combat-caused resource changes;
- sparse, typed effect receipts such as The Hunt's **extra card reward count**.

Use a finite mixture or similarly tractable distribution to preserve
correlations, including bad tails. An extensible effect ledger records
trigger causes, allowing the emulator to deterministically apply consequences
instead of independently predicting correlated effects.

**Exclude** ordinary randomly generated card identities, gold reward amounts,
potion-generation rolls, and ordinary relic offers from the learned combat
outcome targets. Use the emulator's mechanics and stochastic laws for them.
Separate combat-earned gold and exceptional acquisition/transform effects
from normal reward generation. Reward **selection** belongs to the macro actor.

The former `H12JointOutcomeMixture` and `H12MacroContinuation` are
**retired as architecture and inference interfaces**. Their source files,
H12 notebooks/scripts, original checkpoints and reports are historical,
not deployment dependencies. H12's surviving-HP metrics remain historical
evidence, but its reward-change heads confuse combat effects with generated
offers and the old macro policy's reward selections; its standalone
continuation critic failed heldout. Do not refit the same targets.

A map decision precedes encounter identity and opening-hand revelation.
Any model-based lookahead must sample **complete prospective entry states**
using player-information-consistent emulator/Chance-Law machinery. Never
feed the actual future draw, enemy intent, native RNG cursor, seed, or another
run's hand to the macro actor. An empirical observed encounter histogram
alone is not a counterfactual entry generator.

Dynamics training uses proper likelihood/calibration on **real observed
transitions**, not a desirability-weighted loss; actor gradients must not
rewrite chance probabilities. A new combat policy requires refreshed or
explicitly policy-conditioned dynamics predictions.

## Learning and search

1. **Macro actor/value:** train from native full-run trajectories and measured
   run outcome. Where reliable search exists, its action visit distribution
   can supervise the policy head. The value head learns real final returns
   and appropriate temporal-difference targets at macro boundaries.
2. **Combat actor/value:** use combat trajectories, a tactical critic
   bootstrapping from the *frozen, validated* macro value at combat exit,
   and selective legal emulator search to supply improved combat action
   distributions. Train the combat policy to imitate search where useful.
3. **Optional dynamics:** independently train/calibrate the distributional
   head on correctly labeled combat consequences. Use it to accelerate macro
   planning only after validating both the probability law and the resulting
   action rankings against actual emulator rollouts.

Search priors are over **legal player actions**. Chance nodes follow their
conditional laws, and backed-up values average over outcomes rather than
selecting the best sampled result. The model never chooses its combat outcome.
No fixed `aggressive/defensive/reward-seeking` combat modes or external
resource-preference weights are part of the target design. Search and the
critic handle tradeoffs from actual states and newly revealed draws.

Stability: freeze identifiable policy snapshots when collecting data; fit
value/dynamics against those snapshots; improve one policy level at a time;
recalibrate after accepted changes. Preserve incumbent policies, source seed
separation, emulator/DLL and checkpoint hashes, and real native paired
evaluation. No online actor update through a simultaneously drifting
dynamics head.

## Migration from existing research

| Existing work | Intended treatment |
| --- | --- |
| H6-C macro + H9 combat | **Incumbent regression baseline**; retain checkpoints |
| H7 `SharedObjectTransformer`, action codec and PPO | Reuse architecture/training infrastructure; it already has phase-specific policy/value heads |
| H11 frozen-encoder macro PPO | **Negative experiment** (0/128 versus H6-C 10/128 heldout Act 1 clears); reuse useful code, not weights as an assumed strong policy |
| H12 eight-component joint outcome + standalone continuation critic | **Retired model/targets**; archive reproducibly; no deployment |
| H13 public combat/reward boundary and selected-room journal | Reusable instrumentation candidate; H13 empirical catalog and H6-C map override are **not** the canonical prospective dynamics solution |
| `sts2-emulator` native stochastic provider / Chance Laws | Authoritative mechanics and probability boundary; preserve interchangeability |

## Implementation order and promotion gates

1. Formalize public run/combat boundary records, **source-fair relic counters**
   and combat-effect receipts (including explicit potion use and Hunt trigger
   counts). Verify counter persistence in the emulator before collecting
   labels. Preserve independent training/heldout source runs.
2. Establish the **shared macro actor/value** with a stronger macro action
   and persistent-state representation. Treat native full combat as a
   single macro option. Fit and validate critic reliability and actual
   action ranking, not merely training MSE. Keep H6-C/H9 as baseline.
3. Add the **combat critic boundary bootstrap** and use the macro critic
   only after out-of-sample calibration; improve combat policy with
   carefully selected emulator search and search-distilled action targets.
4. Add the **optional dynamics head** only when an observed-transition
   contract, legitimate prospective combat-entry sampler, effect-receipt
   model and calibration benchmarks are ready. Reuse valid H13 records;
   do not revive old H12 resource targets.
5. Alternate policy improvement and predictor refresh, validating Act 1
   clears, survival risk, HP/potion tradeoffs, counter effects, calibration
   and search speed on independent paired native seeds.

**Stop conditions:** do not use an inaccurate macro critic to guide combat
search; do not accept a learned dynamics model merely because its marginal
Brier score is good; do not deploy an unverified map planner. All architecture
pieces are proposals until native acceptance establishes value.

## References and historical sources

- `src/sts2_ai/models/shared_transformer.py` (H7 shared policy/value heads)
- H11 shared macro PPO: PR #139; H12 standalone outcome experiments: PR #140;
  H13 public-boundary instrumentation: PR #141 (draft branches, not merged).
- Silver et al., *Mastering the game of Go without human knowledge* (2017):
  policy/value heads and search distillation.
- Schrittwieser et al., *Mastering Atari, Go, chess and shogi by planning
  with a learned model* (2020): representation/dynamics/prediction split.
- Antonoglou et al., *Planning in Stochastic Environments with a Learned
  Model* (2022): decision and chance-node separation.
