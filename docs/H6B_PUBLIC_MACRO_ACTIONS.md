# H6-B: candidate-conditioned public macro action features

**Scope:** An opt-in macro-policy representation for H6-A's frozen-combat
Silent A10 whole-run PPO. It does not change H4/H5 combat policies, the
emulator, run RNG, or existing H6 checkpoints.

## Why this matters

H6-A's semantic strategy action representation encodes
`choose_map_node:room=<type>`. Two nodes of the same room type are
indistinguishable even when one leads to rests and shops while another leads
to elites. The map for the entire Act is player-visible, so this limitation
is artificial. Likewise, choosing an already-owned reward card differs from
choosing a novel one because the deck multiset is public.

## Opt-in strategy model schema

`sts2-neural-policy-value-v16-public-macro-action` adds:
- Visible selected map room, floor, immediate outgoing branch count and
  categorical room-type proportions reachable in each of the next four
  map layers (unique, reachable nodes only; these **are not path
  probabilities**).
- Selected reward card ID, deck size/diversity, and number of owned copies.
  Removal/upgrade choices also resolve the *public* selected card instance.
- Candidate-conditioned current HP fraction and gold. The rest of the
  public state remains available to the existing strategy state encoder.

The tokenization avoids unstable node and card-instance IDs: IDs are used
only to match an action to a player-visible object. Unknown/unsupported
public schema keys are not inferred from hidden engine state.

**Training/inference parity:** `NeuralPolicyValueModel.evaluate`,
`phase_split_ppo_batch.encode_decision` and `train_phase_split` use the
same `macro_action_features` when selected. The new format ID is
explicit and serialized; existing strategy models keep the old
`NEURAL_FORMAT` and produce the same inference.

The embeddings change the network inputs. Warm-starting from H4/H5 strategy
weights does **not** preserve its exact action distribution when
`macro_v1` is enabled. Treat this as a separately versioned ablation,
not an invisible update to H6-A.

## Local use (fish)

On top of the proven H6-A CLI command (see
[H6-A](H6_FROZEN_COMBAT_MACRO.md)), add:

```fish
--strategy-action-encoding macro_v1
```

The new encoding is only supported when `--freeze-combat` and
`--optimizer-method ppo --ppo-backend batched` are selected.
Use **new** training seed prefixes and output/checkpoint paths for H6-B.

First run paired baseline H6-A training and H6-B training with the
same frozen tactical policy and equal *run* budgets, then compare on the
independent H6 full-run evaluator. Both exports must have exactly
identical combat weights. A zero-training warm-start H6-B baseline should
also be evaluated to isolate the immediate representation shift.

## Limits / next work

- This does not yet encode the probability or utility of route outcomes:
  it only provides visible reachability features for a learned network.
- If source schema changes, version the encoder again. Do not silently
  fall back to a heuristic path estimator.
- Card synergy currently uses owned-card counts, not card text effects,
  relic interaction knowledge, or future deck generation odds.
- Shop offer prices and event choices need their own audited, typed
  public-source encoders in a later version.
- No native fidelity inspector or RNG parity test is a training gate.
- A 1-round smoke is not evidence of policy improvement, and a
  20-round run is insufficient to judge H6 generalization.
