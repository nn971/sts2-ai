# H6-C: public macro-object attention (experimental)

This adds an opt-in strategic network inspired by combat v13's four-head
enemy attention. It does **not** replace legacy H6-A2 nor alter the emulator,
Chance Law, game-reference mod, or frozen tactical weights.

## Architecture

At each macro decision, only player-visible values are read.

- A **global state vector** uses the historical hashed public features,
  excluding the raw map node list to avoid tokenizing ephemeral node IDs.
- A **typed object set** contains grouped deck cards (upgrade and multiplicity),
  relic and potion identities, each legal map candidate with its publicly
  visible four-layer reachable room distribution, and visible reward cards.
- A shared 128→32 **object projection**, four-head masked
  **self-attention** (the v13 kernel) and pooled context condition a
  128→32 global state trunk.
- Every legal action gets its semantic 128-dimensional feature vector and a
  **pointer to its associated visible object**, if any. The policy scorer
  combines global, action, and contextual object embeddings. Critic predicts
  the same bounded run-return target as legacy H6.
- Variable-sized objects and action menus use masks in batched PPO. Portable
  JSON model inference matches Torch single and batch algebra. Hidden game
  state, RNG streams, future event/card realizations and arbitrary instance IDs
  are excluded.

Model format: `sts2-neural-policy-value-v17-macro-object-attention`.
CLI: `--strategy-encoding macro_attention`.

The standalone network is **not** a full map GNN: visible map connectivity is
summarized locally at selected nodes, and the learned attention relates those
object embeddings. Items in unsupported shop/event schemas do not yet receive
a typed pointer. Model weights reuse v13-compatible parameter keys internally,
but the new format ID and strategy-encoding config keep checkpoint semantics
unambiguous.

## First experiment

Use the existing H6-A2-long checkpoint as an initial strategic prior; **never
resume an old H6-A2 optimizer checkpoint with the new architecture**.
Warm-starting copies compatible state/action/policy/value layers and randomly
initializes the new object/attention layers. The attention residual starts at
zero but initial context and target projections are not zero-gated: the initial
policy distribution is not function-identical to the old model. Record a
zero-training warm-start evaluation if a fair pre/post training ablation is
needed.

```fish
cd ~/projects/sts2-ai
source .venv/bin/activate.fish
python -u tools/train_phase_split.py \
  --warm-start results/nature/h6-a2-long-a10.json \
  --freeze-combat \
  --strategy-encoding macro_attention \
  --optimizer-method ppo --ppo-backend batched \
  --tactical-state-encoding enemy_instances \
  --combat-objective hp_preservation \
  --combat-advantage-baseline critic --hp-monotonic-weight 0 \
  --dimension 128 --hidden 32 --ascension 10 \
  --temperature-start 0.85 --temperature-end 0.65 \
  --rounds 80 --episodes 128 --workers 15 \
  --win-anneal-threshold 1024 --max-decisions 4096 \
  --eval-seeds 32 \
  --train-seed-prefix h6c-macro-attention-a10-train-v1 \
  --eval-seed-prefix h6c-macro-attention-a10-select-v1 \
  --checkpoint results/nature/h6c-macro-attention-a10.pt \
  --output results/nature/h6c-macro-attention-a10.json \
  --report results/nature/h6c-macro-attention-a10-report.json
```

Training results alone cannot certify improvement. Once stable and with
training/eval throughput measured, compare to H4, H6-80 and H6-160 on
**new paired held-out Act 1 A10 seeds**. Do not change the combat model or
require native-game fidelity for this experimental gate.

## Acceptance before large training

1. Focused Python tests including numerical logits/gradient parity and
   JSON inference equivalence, masking and action pointer resolution.
2. Frozen combat weights byte-equivalent to the warm start across checkpoint
   save/reload and resumed PPO.
3. Verify two same-type map choices with different future visible routes have
   different object features. Renaming raw map node IDs must not change logits.
4. Short runtime smoke is **only** a correctness check. Run a sufficiently
   long confirmation to make learning claims.
5. Profile CPU costs: object attention is quadratic in object count. Selected
   map nodes only, grouped cards and capped inventory keep this tractable;
   add further sampling/caching only after measuring actual rollouts.
