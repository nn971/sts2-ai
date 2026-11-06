# H24 implementation status addendum — 2026-11-06

This addendum takes precedence over earlier Stage-0-only notes in
`docs/H24_HARD_RESOURCE_MONOTONICITY_20261106.md`.

Completed: standalone monotone-logit primitive, fail-closed public
pair applicability checker, human-verdict and mechanics-review gate
separation, explicit G1–G6 / H1–H3 / D1–D3 / P1–P5 / R1–R5 registry,
and associated synthetic tests. See the canonical
`docs/H24_PREDICATE_VERDICT_20261106.md`.

Pending: independent emulator-backed proof for all mechanical predicates,
resource-specific approved item lists, native executable counterfactual
snapshots, separate invariant resource-neutral value context, H24 head/
checkpoint wiring, inference/search/replay training integration.

**Important refinement**: initially constrain the binary *optimal
Act-clear continuation value* only. The current H17 geometric future-HP
head is trained on behavior-dependent, shaped labels and remains
unconstrained in the first controlled pilot. A mathematically valid
dominance relation for optimal Act-clear probability does not, by
itself, establish monotonicity of shaped on-policy value labels.

Neither human-verdict assertions nor pure public JSON comparisons are
a substitute for native-game mechanical justification. All hard
resource constraints remain **off** until stage-1b and stage-2 gates.
