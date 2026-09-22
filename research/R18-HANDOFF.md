# R18 handoff — parameter-neutral module-local interpolation

Status: **protocol frozen before empirical evaluation**.

## Starting evidence

R15 remains the competence anchor:
- head SHA-256: `007e24fff0e0e48a096de59276f7ab8d0e25bdb826a3f81a7838c5bd6151723f`;
- matched MultiNLI accuracy: about 0.5613;
- HIRA parameters: exactly 422,159.

R16 is a preserved negative result:
- the strict structural curriculum moved too far from the R15 competence basin;
- failure-analysis head SHA-256: `dbe0ddd8bf3811c98f5062bbf482f5d5b4f1f5991fa6f734f7ca88c24e88cc75`.

R17 is also a preserved negative result:
- global interpolation selected alpha 0.025;
- matched accuracy 0.5620 passed retention;
- frozen near-structural accuracy improved only 0.5833 -> 0.5909;
- gain was only +0.0076, below the frozen +0.05 requirement;
- frozen 0.80 support gate failed because only 66 neutral examples were available;
- no R17 thresholds or alpha values are changed after seeing that result.

A13 remains frozen:
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- safetensors SHA-256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`.

## R18 hypothesis

The R16 update may mix:
- useful structural anti-entailment changes in one HIRA functional subsystem;
- harmful competence changes in another subsystem.

R18 tests whether the R15->R16 direction can be localized by parameter group.

For selected keys only:

`theta(alpha) = theta_R15 + alpha * (theta_R16 - theta_R15)`

Every unselected key stays bit-identical to R15.

The result remains a single 422,159-parameter HIRA head. R18 adds no runtime ensemble, second head, routing expert, or hidden parameters.

## Frozen atomic parameter groups

### relation

- `late_scale`
- `cross_attn.*`
- `cross_ln.*`
- `cross_ff.*`
- `cross_score.*`
- `delta_scale.*`

### representation

- `q_proj.*`
- `seg_proj.*`
- `opt_proj.*`
- `token_proj.*`
- `type_emb.*`
- `state_ln.*`
- `option_ln.*`
- `option_token_weight.*`

### coarse

- `coarse_scale`
- `coarse_bias.*`

### budget_control

- `budget_gate.*`

`budget_control` is a negative control. The frozen NLI path uses non-adaptive all-K selection, so changing only this group must not alter final logits/probabilities.

Every HIRA state key must map to exactly one atomic group. Tests reject overlap or uncovered keys.

## Frozen candidate group specs

Single atomic groups:
- relation
- representation
- coarse
- budget_control

Frozen unions:
- relation+coarse
- relation+representation
- representation+coarse
- all_non_budget = relation+representation+coarse

## Frozen alpha grid

For every non-baseline group spec:

- 0.025
- 0.05
- 0.10
- 0.20
- 0.30
- 0.50
- 1.00

R15 alpha=0 is evaluated once as the baseline.

No group or alpha may be added after looking at R18 results.

## Model-selection source

Only:

`nyu-mll/multi_nli@da70db2af9d09693783c3320c4249840212ee221`

Forbidden for R18 selection:
- HANS
- Breaking NLI
- XNLI
- MASSIVE
- Banking77
- Laya final benchmark cells
- Jev/JEV final benchmark cells

## Frozen ranked-structural authority

R18 does not lower or edit the failed R17 threshold.

Instead, on `validation_mismatched`:

1. lowercase alphanumeric tokenization;
2. gold label must be neutral or contradiction;
3. hypothesis length >= 3 tokens;
4. multiset recall = fraction of hypothesis-token multiplicity found in premise;
5. ordered-LCS recall = LCS(premise, hypothesis) / hypothesis token count;
6. structural score = max(multiset recall, ordered-LCS recall);
7. secondary score = min(multiset recall, ordered-LCS recall);
8. rank separately within neutral and contradiction by:
   - descending structural score;
   - descending secondary score;
   - ascending source index;
9. choose exactly top 250 neutral and top 250 contradiction.

There is no structural-score inclusion threshold.

Support validity:
- at least 500 hypothesis-length-eligible neutral examples;
- at least 500 hypothesis-length-eligible contradiction examples;
- selected slice exactly 500 examples.

The receipt must report eligible counts and minimum/mean selected scores.

## Matched retention authority

Existing frozen convention:
- MultiNLI `validation_matched`;
- valid labels only;
- shuffle seed 14;
- first 1,500 examples.

Eligibility:
- matched accuracy >= 0.5606666612625122.

## Selection rule

Among eligible candidates:

1. maximize ranked-structural non-entailment accuracy;
2. tie-break by matched accuracy;
3. prefer fewer atomic groups changed;
4. prefer smaller alpha;
5. stable lexical group name.

## Primary gates

All must pass:

- structural support validity;
- matched accuracy >= 0.5606666612625122;
- ranked-structural non-entailment accuracy >= 0.62;
- ranked-structural improvement >= +0.03 absolute over exact R15 on the same 500-example slice;
- HIRA parameter count exactly 422,159;
- selected candidate eligible;
- every unselected state key bit-identical to R15.

No gate, group, alpha, ranking rule, or tie-break may change after the empirical run.

## Falsification controls

R18 is invalid if:
- any state key maps to zero or multiple atomic groups;
- an unselected state key changes;
- the budget-control-only candidate changes NLI logits/probabilities in non-adaptive mode;
- endpoint hashes differ;
- model-selection data includes a forbidden source.

The `budget_control` lane is retained even if it cannot win; its purpose is implementation falsification.

## Implementation

R18 adds:
- `src/nmd/block_interpolation.py`;
- `tests/test_block_interpolation.py`;
- `scripts/r18_module_local_interpolation.py`;
- `.github/workflows/r18-module-local-interpolation.yml`.

## Post-selection

Only if every R18 primary gate passes may adapted HANS/Breaking diagnostics run.

Those suites remain adapted diagnostics because their prior results are already known. XNLI/MASSIVE/Banking77 and Laya/Jev final cells remain unopened for R18 model selection.

A failed R18 is preserved as a negative result and must not be rescued by changing the R18 protocol.
