# Stronger MCTS teacher and neural holdout

The initial neural-v2 pilot successfully trained a model but the 12-simulation
teacher's normalized action-visit entropy was 0.995, nearly uniform.
Increasing network size without improving supervision is unlikely to help.

## Action-semantics-aware teacher audit

The command **sts2-ai diagnose-training** now reports original visit entropy
and additional semantic-action metrics: literal/semantic action counts,
normalized semantic entropy, top semantic probability above uniform, minimum
visits per semantic action, value gap between top two estimated choices, and
agreement between the best-value and best-visited semantic choices.

Multiple copies of truly resolved identical decisions can be aggregated;
unknown semantics are deliberately **not** treated as equivalent, since
different shop purchases or event options can produce different outcomes.
No hidden state is used to resolve semantics; all classification uses the
observable frame plus legal action payloads.

A conservative signal-quality filter is available:

~~~fish
sts2-ai curate-teacher results/high-roots.jsonl \
    --min-budget 64 --min-visits-per-semantic-action 2 \
    --min-semantic-top-margin 0.08 \
    --min-semantic-value-gap 0.01 \
    --output results/curated-roots.jsonl \
    --json-output results/teacher-audit.json
~~~

The selected records retain their *original, unchanged* probabilities.
Quality filters are heuristics, not statistical confidence intervals or
guarantees that their chosen actions are optimal. Zero retained records is
a meaningful negative finding, not a reason to fabricate target sharpness.

## Automated experiment

The GitHub Actions workflow at
.github/workflows/teacher-quality-experiment.yml runs:

1. MCTS-12 vs MCTS-96 teacher generation with the same run seed prefixes.
   The root states may diverge after the first different decision.
2. Strict per-action and semantic-policy diagnostics and a named curation
   policy with fallback to all high-budget records if too few are accepted.
3. A small capped independently continued cutoff dataset spread across
   original source seeds; nonterminal continuations stay censored.
4. A neural-v2 checkpoint trained on the explicitly chosen teacher records.
5. Three *unseen* seed-matched MCTS-32 evaluations with equal depth: plain
   heuristic, learned value (25% blend), and neural rollout policy.
6. Seed-paired summaries of continuous frontier progress, victories,
   truncations, wall-clock and emulator transitions.

The workflow artifact is **stronger-teacher-neural-heldout**. It is a small
methodology pilot, not proof of improved gameplay. Follow-up experiments
need substantially more full-run heldout seeds and equal runtime budgets.

The emulator pin, default heuristic MCTS, and oracle-exact information-policy
boundary are unchanged.
