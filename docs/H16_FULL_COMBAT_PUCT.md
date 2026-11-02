# H16 R79 full-combat PUCT: 16 versus 64 simulations

The previously collected 64-seed evaluations already establish three read-only
reference arms: policy-only, PUCT16 depth 8, and PUCT64 depth 8.

This experiment adds two arms using the same frozen R79 checkpoint and 64 run
seeds: **PUCT16 full combat** and **PUCT64 full combat**.

"Full combat" means **full-depth PUCT tree selection until the combat boundary
is reached on every simulation**. No depth-8 neural value cutoff is used and
there is no policy rollout after depth 8. The fight's actual win/loss is
observed, and the same learned continuation critic evaluates the successful
combat boundary. The PUCT search still uses exact-state forks, so this is
a privileged evaluation.

A 512-action per-simulation watchdog prevents infinite fights. Exceeding it
raises \`IncompleteCombatRollout\`: no synthetic loss, win, or value is recorded.
Raise \`--max-simulation-steps\` for another *new* output if this occurs.

## Fish shell

Pull the research branch:

\`\`\`fish
cd ~/projects/sts2-ai
git fetch origin
git switch research/h16-paired-checkpoint-diagnosis-20261102
git pull --ff-only

set emu ~/projects/sts2-emulator
set macro results/nature/h6c-macro-attention-r80.json
set progress results/h16-v5-combat-credit-r150/progress.json
set reference results/h16-v5-combat-credit-r150/paired-diagnostic.json
set depth8 results/h16-v5-combat-credit-r150/puct16-vs64-r79.json
set output results/h16-v5-combat-credit-r150/puct-full-combat-r79.json
\`\`\`

If the previous JSON outputs have different local paths, update only
\`$reference\` and \`$depth8\`. The preflight validates that the first report
matches the second, plus the original progress, checkpoint, macro-model hash,
and emulator revision.

Preflight (does not run any games):

\`\`\`fish
python tools/diagnose_h16_full_combat.py \
    --reference $reference --depth8-budget $depth8 \
    --progress $progress --emulator-root $emu --macro-model $macro \
    --output $output
\`\`\`

First run two full-combat evaluations for each budget, saving resumable journals:

\`\`\`fish
python tools/diagnose_h16_full_combat.py \
    --reference $reference --depth8-budget $depth8 \
    --progress $progress --emulator-root $emu --macro-model $macro \
    --output $output --limit 2 --execute
\`\`\`

The same command with \`--limit 16\` extends to a paired 16-seed pilot, and
with \`--limit 64\` completes all 64 seeds per budget. Both arms resume from
\`$output.s16.partial.jsonl\` and \`$output.s64.partial.jsonl\`; completed runs
are never repeated on resume.

Run statistics are printed after each successful batch. The completed output
provides five summaries and five paired comparisons, including:

- full16 versus depth8-16 and full64 versus depth8-64;
- full64 versus full16;
- both full-search arms versus policy-only;
- floor-7 outcomes, Act clears, max floor, paired floor changes with bootstrap
  intervals, and actual search transitions.

The 16/64 simulation budget is **per eligible combat decision** (one search
root per decision), not 16/64 simulations per run. All arms keep depth-8
evaluation's \`temperature=0.85\`, macro prior, and HP horizon constant.

## Constraints

The terminal-combat search can be expensive and may hit the watchdog on
unusually long combats. Such an exception is a surfaced incomplete experiment,
never a numerical terminal reward. Paired seeds permit run-level comparisons
without guaranteeing identical later combat states. R79 was selected on these
same seeds, so a confirmation on new seeds is needed for a population claim.
