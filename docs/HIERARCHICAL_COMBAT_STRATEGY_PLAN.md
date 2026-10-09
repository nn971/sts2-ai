# Combat–strategy hierarchy: boundary milestone (2026-10-10)

## Goal

Train combat tactics from completed combats without waiting for an entire run
to end, while preserving the strategic value of HP, potions and future fights.
The first milestone is **data and phase segmentation only**. It does **not**
change REINFORCE returns, neural weights, checkpoint fingerprints or the
emulator.

## Implemented in this milestone

- `PublicDecision.phase` is `combat` when the fair observation has a
  `combat` object, otherwise `strategy`. There is still **one** neural
  policy for both phases. Phase routing is now explicit and testable.
- `Episode.combat_outcomes` holds individual `CombatOutcome` samples.
  Every resolved combat records its act/floor, visible enemy identities,
  zero-based half-open decision span, last combat turn, result, resource entry
  and exit snapshots, cumulative observed HP decreases/increases, and actual
  `use_potion` events resolved by publicly visible potion slot/type.
- `RunResources` preserves **potion identities with multiplicity**, relic
  identities, gold, HP and max HP. Comparing only potion counts would
  incorrectly equate potions with different effects.
- Unresolved fights at a decision cap produce **no fabricated outcome**.
  Defeat frames are recognized whether the bridge keeps or removes the
  `combat` field. All observations are public-only; no exact-state handles,
  RNG seeds, or private game information enter the recorder.
- Each training round now prints combat victory/loss counts, average observed
  HP decreases per won combat and potion use. These metrics are **descriptive**
  and do not yet influence updates. A round's sample collection still
  represents one frozen-policy cohort.

Optional **individual outcome-sample export** during ordinary training:
`--combat-samples-dir results/combat-samples` produces one JSONL file per
completed round (e.g. `round-0001.jsonl`), preserving seed, round, sampling
temperature, emulator revision, and the full public resource vector. The
trainer logs these exports as they happen. Missing sidecar files after a
crash are **not** automatically reconstructed from an already-checkpointed
round. The export callback is observational, not a training signal.

Run focused tests:

```fish
cd ~/projects/sts2-ai
source .venv/bin/activate.fish
python -m pytest -q tests/test_combat_outcomes.py tests/test_neural_selfplay.py
```

## Next milestone: separate learner and improve credit assignment

1. Expose two legal-action-compatible policy heads: tactical for combat,
   strategic for other phases. Build target-aware representations before
   simply adding more layers. Combat features must preserve each enemy's
   identity, HP, block, intent, and each card's available targets and cost.
2. Train a combat outcome model on observed **individual** resource-transition
   samples, rather than collapsing them into an average HP loss. Validate
   predictions on held-out seeds/encounters and keep the current learner
   as the control.
3. At a combat boundary, use a **strategic continuation critic** to evaluate
   the post-combat state, including specific remaining potions. Tactical
   learning can use short-horizon advantages with that boundary value.
   Avoid a globally fixed HP-per-potion exchange rate. Use a temporarily
   frozen target critic when updating tactical policy to stabilize coupling.
4. Only then add PPO/GAE or other temporal credit assignment. Maintain
   terminal-win priority and full-run independent evaluation. Always
   distinguish emulator chance outcomes from future information exposed
   to the agent.

## Why keep outcome samples rather than only expectations?

Two combat policies can have equal expected remaining HP but very different
failure probabilities; HP and potion outcomes are also correlated. Later
critics may estimate calibrated distributions or quantiles over
`(win, HP_exit, inventory_exit)`, not only componentwise means. For a cheaper
first step, use a **normalized variance** (e.g. variance divided by a
squared resource scale) alongside expectations, keeping the formulation
consistent across HP ranges. More advanced risk measures (quantiles, CVaR)
remain possible. The controller should optimize an explicitly defined
run-level objective, not automatically prefer variance in either direction.

Dominance supervision is valid only when the future-relevant resources and
other persistent state are equal: higher HP is preferable *all else equal*.
Trading HP for a particular potion requires the strategic critic, not a
fixed combat reward. Any deterministic shaping must avoid prolonging
fights or spending potions excessively.

## Invariants and cautions

- Use `native-overgrowth` with the pinned emulator revision; unsupported
  mechanics must remain explicit errors, not be silently implemented in
  `sts2-ai`.
- Do not use game RNG seeds or exact handles as policy input.
- Distinguish `hp_decreases` from *enemy damage*: it also counts self-inflicted
  HP costs; `hp_increases` counts healing. A later reward model must account
  for these categories carefully.
- Defeated combats have zero exit HP; analyze won combats and encounter
  survival rates separately when diagnosing tactical quality.
- Compare tactical/strategic ablations on the same unseen seed cohort,
  not just raw training progression. Primary run metrics remain Act-1
  clears, boss-entry rate and eventual full victories, with HP/potion and
  encounter-level diagnostics.
