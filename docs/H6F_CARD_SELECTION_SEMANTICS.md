# H6-F: visible card-selection operation semantics (V20)

## Motivation

A `choose_event_deck_card` action in Luminous Choir can remove two cards,
not upgrade them. In pinned emulator revision
`0469424fc20679fd81c0d5e4ee8d9a484ec19108`, the event's `reach`
choice is **remove two cards, then gain Spore Mind**. The player-facing
`event.pending_deck_choice.kind` distinguishes `Remove`, `Upgrade`,
`Transform`, and `Enchant`. The matching shop and reward prompts expose
`shop.pending_deck_choice.kind` and `reward.pending_deck_choice.kind`.

V18/V19 identified the selected card, but not the currently requested card
operation in each action's own representation; V19 also omitted the action's
card target pointer for these three prompt kinds. The global public state
could contain the operation, but that is weaker than binding it to the
selection action and chosen deck object.

## Fix

Introduce opt-in `--strategy-encoding macro_card_ops` with a **new portable
format V20** `sts2-neural-policy-value-v20-deep-macro-card-operations`.
Historical V17/V18/V19 inference is **not changed**.

- Read **only public** `pending_deck_choice.kind` from the action's event,
  shop, or reward. Decode both JSON string enum and the pinned C# enum values
  (Remove=0, Upgrade=1, Transform=2, Enchant=3).
- Condition each card-selection action on operation, source, remaining count,
  and already-public source relic / upgrade-transformed flag when present.
- Bind each candidate action to the corresponding deck card object embedding,
  retaining the chosen card even if the deck exceeds the 48-object cap.
- Reuse V19's attention and parameter shapes. V19 checkpoints can warm-start
  V20 with no new trainable tensors; the old checkpoint retains its format
  and previous behavior. The frozen combat head is not modified.
- Route portable inference, per-decision Torch and batched PPO through exactly
  the same public features and pointers; test all three prompt sources.

This is a representation fix, **not proof of sound card selection**. The
policy must still learn which cards are valuable. An unknown or absent kind is
explicitly `unknown`, not inferred from the event name or hidden state.

## Validation

```sh
python -m pytest -q tests/test_h6f_card_selection_semantics.py \
  tests/test_h6e_deep_macro_attention.py tests/test_h6d_semantic_macro_actions.py
```

This is a correctness gate only. Do not interpret a few short episodes as
macro-performance evidence. Keep the H6-D-80 and H6-E-80 checkpoints unchanged.

## Research priority following this fix

Next train and certify a **substantially stronger Ascension-10 combat policy**.
Use varied decks, encounters, boss fights and relic states. Compare against the
existing frozen H4 policy using held-out combat situations, greedy and MCTS
evaluations, win probability, and HP/boss-HP outcomes. The outcome predictor
should be trained only after that combat backbone is competent enough to make
macro route/reward comparisons meaningful. Combat fidelity-inspector work
remains independent and must not block agent training.
