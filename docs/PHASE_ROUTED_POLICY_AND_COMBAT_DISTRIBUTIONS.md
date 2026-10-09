# Tactical versus strategic inference: second milestone

This follows `docs/HIERARCHICAL_COMBAT_STRATEGY_PLAN.md`.

## What was added

- `PhaseRoutedNeuralAgent` selects between two **independent** existing
  portable neural models. A public `combat` observation dispatches to
  the tactical model; all other phases use the strategic model. Legal
  actions remain authoritative. Greedy and temperature-sampled inference
  both work. Models can be loaded from two different JSON paths.
- `EmpiricalCombatDistribution` summarizes **individual and correlated**
  post-combat HP and potion-inventory outcomes from the JSONL sidecars
  produced by `--combat-samples-dir`. It retains probabilities of
  victory, normalized HP variance, and discrete joint exit outcomes,
  not just expected HP. Its conditional grouping is a deliberately
  conservative, low-dimensional baseline and now **abstains** when a
  subgroup is too small. An unconditional mixed-context summary is
  available separately and is explicitly not a conditional forecast.
- Focused deterministic tests exercise phase dispatch, sample loading,
  potion/HP correlations and sparse subgroup abstention.

These components are independent of `sts2-emulator`; there is no new
game-mechanics implementation.

## Current limitations — important

The phase router is **inference only** at this point. It does not
pretend that the two policies have been jointly trained. Existing
self-play still updates one shared model using full-episode REINFORCE.
The distribution estimator is descriptive **on-policy** statistics,
not a learned conditional dynamics model and not a counterfactual
model of what would happen if a different card were played. A handful
of combats is not sufficient to estimate useful conditional tails.
A per-context threshold does not establish calibration: the grouping omits
decks, relic effects, some combat-specific inputs, and policy variation.
Only use the unconditional summary for descriptive dataset analysis.
The v2 estimator intentionally returns null predictions when a context is
undersampled; future policy code must handle abstention rather than treating
null as a probability or a low-value estimate.

## Commands

After running several ordinary self-play training rounds with
`--combat-samples-dir results/combat-samples`, inspect their samples:

```fish
cd ~/projects/sts2-ai
source .venv/bin/activate.fish
python tools/inspect_combat_distribution.py results/combat-samples/round-*.jsonl \
  --output results/combat-distribution-report.json
python -m pytest -q tests/test_phase_routed_neural.py tests/test_combat_distribution.py
```

## Next integration

Introduce separate optimizer parameter sets for tactical and strategic
models. Train the tactical model using short combat segments and a
**temporarily frozen strategic continuation value** at combat exits.
Resource tradeoffs must be conditional on the run state: the value of
a potion depends on its identity and forthcoming encounters. Ensure
truncated combats do not receive fictional loss labels. Compare held-out
boss-entry, Act-1 clear and HP/resource metrics against the shared-policy
control. Future critic upgrades may learn normalized variance, quantiles,
or full calibrated resource-outcome distributions.

A risk functional has deliberately not been chosen in this milestone.
