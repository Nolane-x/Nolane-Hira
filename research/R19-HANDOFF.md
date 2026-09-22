# R19 handoff — train-only relation-delta coordinate surgery

Status: **protocol frozen before empirical evaluation**.

## Starting evidence

R15 remains the competence anchor:
- HIRA head SHA-256: `007e24fff0e0e48a096de59276f7ab8d0e25bdb826a3f81a7838c5bd6151723f`;
- HIRA parameter count: exactly 422,159.

R16 remains a negative full-training result, but its failure-analysis head defines a frozen structural update direction:
- SHA-256: `dbe0ddd8bf3811c98f5062bbf482f5d5b4f1f5991fa6f734f7ca88c24e88cc75`.

R17 showed that a small **global** interpolation can retain competence but extracts too little structural gain.

R18 then isolated the failure:
- exact support-valid 500-example ranked-structural slice;
- R15 structural baseline: 0.626;
- best eligible scalar block candidate: relation+representation, alpha 0.025;
- matched: 0.5620;
- structural: 0.630;
- gain: +0.004, below the frozen +0.03 gate;
- relation alpha 1.0 reached 0.702 structural but only 0.498 matched;
- relation+coarse alpha 1.0 reached 0.710 structural but only 0.4967 matched.

Therefore the useful structural direction exists, but scalar block movement does not separate it from competence damage.

## R19 hypothesis

The R16 relation delta contains both:
- coordinates aligned with structural improvement;
- coordinates expensive for retained MultiNLI competence.

R19 does not train a new direction. It uses **MultiNLI train only** to rank coordinates of the already-frozen R15→R16 relation delta, then constructs sparse parameter-neutral candidates.

## Frozen sources

- A13 revision: `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- A13 safetensors SHA-256: `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- MultiNLI revision: `da70db2af9d09693783c3320c4249840212ee221`;
- R15 head SHA above;
- R16 failure-analysis head SHA above.

Forbidden for R19 selection:
- HANS;
- Breaking NLI;
- XNLI;
- MASSIVE;
- Banking77;
- Laya final benchmark cells;
- Jev/JEV final benchmark cells.

## Mutable scope

Only the R18 `relation` atomic group can move:

- `late_scale`;
- `cross_attn.*`;
- `cross_ln.*`;
- `cross_ff.*`;
- `cross_score.*`;
- `delta_scale.*`.

Every representation/coarse/budget-control coordinate stays bit-identical to R15.

Relation coordinates are flattened deterministically by:
1. sorted state-dict key;
2. row-major flattened tensor offset.

For coordinate `i`:

`delta_i = R16_i - R15_i`.

If selected:

`theta_i = R15_i + alpha * delta_i`.

If masked off, it remains bit-identical to R15.

## Train-only structural utility

From MultiNLI `train`:
- valid labels;
- neutral + contradiction;
- hypothesis length >= 3 lowercase alphanumeric tokens;
- structural score = max(multiset recall, ordered-LCS recall);
- secondary score = min(multiset recall, ordered-LCS recall);
- per-label ranking: descending primary, descending secondary, ascending source index;
- exactly top 2,000 neutral + 2,000 contradiction.

At exact R15, compute the mean gradient of:

`CE + 0.5 * anti_entailment_margin(margin=0.5)`.

For coordinate `i`:

`raw_benefit_i = -g_struct_i * delta_i`

`benefit_i = max(0, raw_benefit_i)`.

Only coordinates with:
- benefit > 0;
- abs(delta) > 1e-12

enter the positive-benefit pool.

## Train-only retention sensitivity

From MultiNLI `train`:
- add filtered-split source index;
- shuffle seed 13;
- exclude every structural-utility source index;
- first exactly 6,000 remaining rows.

At exact R15:
- batch size 96;
- CE only;
- for each batch compute the mean CE gradient;
- accumulate sample-count-weighted squared batch-mean gradients.

This is called **batch-gradient energy proxy**, not Fisher information.

`cost_i = retention_energy_i * delta_i^2`.

## Frozen coordinate scorers

1. `benefit`
   - score = benefit.

2. `benefit_over_cost`
   - score = benefit / (cost + 1e-12).

All scoring arithmetic is stored/evaluated in float64 after gradients and deltas are materialized.

Positive-benefit support gate:
- pool size >= 10% of all relation coordinates.

## Frozen mask grid

Within each scorer:
- stable descending score;
- exact ties keep ascending global coordinate index.

Fractions of the positive-benefit pool:
- 5%;
- 10%;
- 20%;
- 40%.

Mask size:
- `ceil(fraction * positive_pool_size)`, minimum 1.

Increasing masks must be nested.

For every mask:
- alpha 0.25;
- alpha 0.50;
- alpha 1.00.

Total:
- 24 sparse candidates;
- plus exact R15 baseline as comparison authority only.

## R19 validation — disjoint scored examples

### Ranked structural

MultiNLI `validation_mismatched`:
- same deterministic ranking as R18;
- zero-based per-label slice `[250:500]`;
- human ranks 251–500;
- 250 neutral + 250 contradiction;
- exactly 500 total;
- must have empty source-index intersection with R18 ranks 1–250.

This is adapted/disjoint development evidence, not a new held-out benchmark family.

### Matched retention

MultiNLI `validation_matched`:
- valid labels;
- add source index;
- shuffle seed 14;
- positions `1500:3000`;
- exactly 1,500;
- must be disjoint from the prior positions `0:1500`.

## Eligibility

First measure exact R15 on the R19 slices.

A sparse candidate is eligible only when:

- matched accuracy >= 0.55;
- matched accuracy >= R19 R15 matched baseline - 0.002.

## Selection

Among eligible sparse candidates:

1. maximize ranked-structural non-entailment accuracy;
2. tie-break by matched accuracy;
3. prefer fewer changed relation coordinates;
4. prefer smaller alpha;
5. lexical scorer name.

R15 itself cannot win selection.

## Primary gates

All must pass:

- structural train n = 4,000;
- retention train n = 6,000;
- train sets disjoint;
- positive-benefit pool >= 10% of relation coordinates;
- structural validation n = 500;
- structural validation disjoint from R18 scored top-250-per-label examples;
- matched validation n = 1,500;
- matched validation disjoint from prior positions 0:1500;
- selected candidate eligible;
- matched >= 0.55;
- matched >= same-slice R15 baseline - 0.002;
- ranked-structural non-entailment accuracy >= 0.60;
- ranked-structural gain >= +0.02 over same-slice R15;
- all non-relation parameters bit-identical to R15;
- every masked-off relation coordinate bit-identical to R15;
- parameter count exactly 422,159;
- masks nested.

No threshold, scorer, fraction, alpha, ranking rule, or selection tie-break may change after the empirical run.

## Implementation

R19 adds:
- `src/nmd/delta_surgery.py`;
- `tests/test_delta_surgery.py`;
- `scripts/r19_relation_delta_surgery.py`;
- `.github/workflows/r19-relation-delta-surgery.yml`;
- this handoff.

The empirical workflow must retrieve exact R15/R16 artifacts and verify both hashes before any coordinate scoring.

## Interpretation discipline

A pass supports only the narrow claim that train-only coordinate surgery extracts more of the already-observed R16 structural direction while retaining competence on disjoint MultiNLI development slices.

It does not establish:
- Laya parity;
- Jev parity;
- broad OOD robustness;
- multilingual generalization;
- general reasoning competence.

Only after every R19 primary gate passes may already-adapted HANS/Breaking diagnostics run.

A failed R19 remains a negative result and must not be rescued by post-hoc changes under the R19 label.
