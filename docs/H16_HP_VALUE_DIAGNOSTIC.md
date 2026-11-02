# H16 R79: Current HP versus learned continuation value

This is a **read-only** critic diagnostic. The previous five-arm full-combat
PUCT results contain per-run outcomes but **no per-decision critic predictions**.
This tool runs **sampled-policy only**, with no PUCT, on the exact 64 policy
seeds from the original \`paired-diagnostic.json\`, using the frozen R79 model,
unchanged H6C macro prior, and the common \`lambda=0.4113950303216448\`.

The collector now records the continuation critic value it **already computes**
while choosing an action; thus there is no extra neural inference per decision.
The runner checks frozen model/progress/macro hashes and verifies the replayed
run-level outcome against the supplied original policy-only reference.

## Fish shell commands

\`\`\`fish
cd ~/projects/sts2-ai
git fetch origin
git switch research/h16-paired-checkpoint-diagnosis-20261102
git pull --ff-only

set emu ~/projects/sts2-emulator
set macro results/nature/h6c-macro-attention-r80.json
set progress results/h16-v5-combat-credit-r150/progress.json
set reference results/h16-v5-combat-credit-r150/paired-diagnostic.json
set output results/h16-v5-combat-credit-r150/hp-vs-value-r79.json

set args --reference $reference --progress $progress \
    --emulator-root $emu --macro-model $macro --output $output

python -m pytest -q tests/test_h16_hp_value.py
python tools/diagnose_h16_hp_value.py $args                 # preflight
python tools/diagnose_h16_hp_value.py $args --limit 2 --execute
python tools/diagnose_h16_hp_value.py $args --limit 16 --execute
python tools/diagnose_h16_hp_value.py $args --limit 64 --execute
\`\`\`

The \`--limit\` arguments extend one resumable
\`hp-vs-value-r79.json.partial.jsonl\` journal. Use the original 64-seed
\`paired-diagnostic.json\`, not the newer \`puct-full-combat-r79.json\`.

The final command writes:

- **\`hp-vs-value-r79.json\`**: full per-decision dataset, summary statistics
  and correlations;
- **\`hp-vs-value-r79.svg\`**: two-panel current HP fraction versus predicted
  continuation value scatter plot for macro and combat, with ten-bin means
  and the current-HP diagonal.

Use \`analysis.by_phase\` to separate macro/combat; \`analysis.by_floor_phase\`
to see floor-specific correlations; and
\`analysis.first_decision_per_seed_floor_phase\` as a robustness check that
avoids giving the many consecutive combat decisions excessive weight.
\`within_floor_phase_pearson\` removes floor/phase group means before computing
HP/value correlation. Finally,
\`analysis.same_hp_floor_phase_value_spread\` identifies states with **identical
current HP / maximum HP, floor, and phase** whose value predictions differ;
variation within such groups is direct evidence that the critic uses additional
state information.

For each complete run the tool also computes the **actual H16 geometric
training target** using post-room HP and, for labeled combat decisions, the
same combat-quality mixture as the training code. Reported target MAE/MSE
measure value calibration against *single realized outcomes*, which remain
stochastic.

## Interpretation and caveats

- The learned \`V(s;lambda)\` is a geometric **future-room HP/Act success**
  target for macro and a combat-quality-shaped mixture for combat. It is
  not the probability of winning Act 1.
- Current HP is an important input, but differences in floor, combat state,
  deck strength and resources confound the unadjusted correlation. Examine
  phase/floor-stratified plots and same-HP spread.
- These are **sampled-policy states**, so the continuation critic is evaluated
  on-policy for the no-search baseline. Full-combat PUCT can visit other states,
  which would warrant a separate evaluation if necessary.
- Critic targets include many correlated decision states within each combat,
  and per-trajectory realized targets are noisy; the tool reports both
  all-decision and one-decision-per-seed/floor/phase summaries.
