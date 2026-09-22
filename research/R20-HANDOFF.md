# R20 handoff — retention-subspace projection of the frozen relation delta

Status: **protocol frozen before empirical evaluation**.

## Starting evidence

R15 remains the competence anchor:
- selected head SHA-256: `007e24fff0e0e48a096de59276f7ab8d0e25bdb826a3f81a7838c5bd6151723f`;
- HIRA parameter count: exactly 422,159.

R16 remains a negative full-training result, but its failure-analysis head defines the frozen structural direction:
- SHA-256: `dbe0ddd8bf3811c98f5062bbf482f5d5b4f1f5991fa6f734f7ca88c24e88cc75`.

R18 showed relation-block structural utility but a strong retention trade-off.
R19 then showed that independent coordinate sparsification still cannot separate that trade-off:
- exact R15 baseline on the disjoint R19 matched slice: 0.54933;
- R15 structural baseline: 0.640;
- positive-benefit relation coordinates: 112,729 / 198,021;
- best retention candidate: 0.54733 matched / 0.654 structural;
- best structural candidate: 0.508 matched / 0.690 structural;
- no candidate eligible.

R20 therefore changes the geometry rather than extending R19's mask grid.

## Hypothesis

The frozen R15→R16 relation delta contains a correlated component aligned with retention-sensitive gradient directions.

A low-rank rotation that removes that correlated component may preserve more structural utility per unit of competence damage than scalar interpolation or coordinate-wise masking.

R20 performs **no gradient training update**. Gradients define the retention subspace and structural diagnostic only.

## Frozen sources

- A13 revision: `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- A13 safetensors SHA-256: `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- MultiNLI revision: `da70db2af9d09693783c3320c4249840212ee221`;
- exact R15/R16 head hashes above.

Forbidden for R20 model selection:
- HANS;
- Breaking NLI;
- XNLI;
- MASSIVE;
- Banking77;
- Laya final cells;
- Jev/JEV final cells.

## Mutable parameter scope

Only relation state may move:
- `late_scale`;
- `cross_attn.*`;
- `cross_ln.*`;
- `cross_ff.*`;
- `cross_score.*`;
- `delta_scale.*`.

All non-relation state stays bit-identical to R15.

## Train-only structural authority

Pinned MultiNLI `train`, neutral/contradiction only.

Using the same exact deterministic structural ranking as R18/R19:
- hypothesis length >=3;
- primary score = multiset hypothesis-token recall;
- secondary tie-break = ordered-LCS recall;
- ascending filtered source index after score ties.

Before ranking, exclude the exact union of:
- R19 structural-train source indices;
- R19 retention-train source indices.

Then rank the remaining eligible examples and take the strongest:
- 2,000 neutral;
- 2,000 contradiction;
- exactly 4,000 total.

This pre-run exclusion rule makes the R20 structural authority disjoint from every R19 scored-train example by construction.

At exact R15 compute mean:
`CE + 0.5 * anti_entailment_margin(margin=0.5)`

gradient over relation parameters.

This gradient is diagnostic only.

## Train-only retention-subspace authority

First reconstruct exact R19 structural and retention source-index sets.

From MultiNLI train:
- valid labels;
- shuffle seed 17;
- exclude R19 structural train;
- exclude R19 retention train;
- exclude R20 structural train;
- take first exactly 6,000 remaining examples.

Batch size 96.

For each batch compute relation-coordinate mean CE gradient `g_b`.

Weighted row:
`r_b = sqrt(n_b/N) * g_b`.

Stack rows into `G`.

Compute eigendecomposition of the small Gram matrix:
`G G^T`.

Numerical eigenvalue floor:
- relative to largest eigenvalue: `1e-10`.

Map nonzero Gram directions to the equivalent orthonormal retention subspace.

Required max orthonormality error:
- `1e-7`.

This is a retention gradient-energy subspace proxy, not exact Fisher information.

## Frozen coverage grid

Retention-energy targets:
- 0.50;
- 0.75;
- 0.90.

For each target use the minimum numerical rank whose cumulative retained eigenvalue energy reaches the target.

## Frozen rotation grid

Let:
`d = theta_R16_relation - theta_R15_relation`.

For retention basis `Q`:

`P(d) = Q Q^T d`

`d_rot = d - lambda * P(d)`.

Removal strengths:
- 0.50;
- 1.00.

Application scales:
- 0.25;
- 0.50;
- 1.00.

Candidate:
`theta_relation = theta_R15_relation + alpha * d_rot`.

Exactly 18 candidates.

For each candidate record:
- retention rank;
- realized coverage;
- projected-component energy fraction;
- retained delta L2;
- removed delta energy fraction;
- cosine with original delta;
- train-only first-order structural predicted benefit;
- applied update L2;
- exact changed relation coordinate count.

## R20 validation authority

### Ranked structural

MultiNLI `validation_mismatched`:
- exact same continuous ranking;
- zero-based per-label slice `[500:750]`;
- human ranks 501–750;
- 250 neutral + 250 contradiction;
- exactly 500;
- disjoint from R18 [0:250] and R19 [250:500].

### Matched retention

MultiNLI `validation_matched`:
- exact filtered source indexing;
- shuffle seed 14;
- positions `3000:4500`;
- exactly 1,500;
- disjoint from all prior positions `0:3000`.

## Baseline-validity gate

Before candidate selection evaluate exact R15 on the R20 matched slice.

Authority is valid only if:
- exactly 1,500 examples;
- each label count >=400;
- R15 matched accuracy >=0.53.

If invalid, R20 must be preserved without changing the slice or floor.

## Candidate eligibility

Candidate must satisfy:
- baseline validity;
- matched >=0.53;
- matched >= same-slice R15 baseline -0.002;
- first-order structural predicted benefit >0;
- removed relation-delta energy fraction >0.

## Selection

Among eligible candidates:
1. maximize ranked-structural non-entailment accuracy;
2. tie-break by matched accuracy;
3. prefer smaller applied update L2;
4. prefer smaller alpha;
5. prefer larger removal strength;
6. prefer lower coverage target.

R15 is comparison authority only.

## Primary gates

All must pass:
- structural train exactly 4,000;
- retention train exactly 6,000;
- R20 train sets disjoint from each other and R19 scored train sets;
- retention basis orthonormal error <=1e-7;
- every requested coverage achieved;
- structural validation exactly 500 and disjoint from R18/R19;
- matched validation exactly 1,500 and disjoint from prior windows;
- matched label support valid;
- baseline-validity gate passes;
- selected candidate eligible;
- matched >=0.53;
- matched >= same-slice R15 -0.002;
- structural accuracy >=0.60;
- structural gain >=+0.02 absolute over same-slice R15;
- removed delta energy fraction >=0.05;
- all non-relation state bit-identical to R15;
- HIRA parameter count exactly 422,159.

No threshold, coverage, removal strength, alpha, validation slice, or tie-break may change after empirical observation.

## Implementation

R20 adds:
- `src/nmd/subspace_projection.py`;
- `tests/test_subspace_projection.py`;
- `scripts/r20_retention_subspace_projection.py`;
- `.github/workflows/r20-retention-subspace-projection.yml`;
- this handoff.

## Interpretation discipline

A pass would support only the narrow claim that a train-only retention-gradient subspace can rotate the already-observed R16 relation direction into a better structural/retention trade-off on new disjoint MultiNLI development slices.

It does not establish:
- Laya parity;
- Jev/JEV parity;
- broad OOD robustness;
- multilingual generalization;
- general reasoning competence.

Only after every R20 primary gate passes may adapted HANS/Breaking diagnostics run.

A failed R20 must be preserved as a negative result.


## Pre-run protocol consistency amendment

Before any R20 empirical result was observed, the originally drafted fixed-rank structural window was found to conflict with the simultaneous requirement that R20 structural scoring data be disjoint from R19's random retention-scoring set.

The frozen R20 authority is therefore the deterministic exclusion-first rule above:
1. reconstruct exact R19 structural and retention scored-train source indices;
2. exclude their union;
3. rank the remaining MultiNLI train neutral/contradiction examples with the already-frozen structural ranking;
4. take the strongest 2,000 per label.

This amendment is a protocol-consistency repair made before empirical evaluation. It does not change the R20 candidate grid, validation slices, thresholds, or selection rule.


## Empirical result — PRIMARY GATE FAIL

Authoritative workflow run: `35709849173`.

Execution correctness:
- unit tests: pass;
- exact R15/R16 endpoint SHA verification: pass;
- empirical tournament: completed;
- evidence artifact uploaded;
- full repository bundle uploaded.

Artifact evidence:
- `r20-subspace-projection-evidence` artifact ID: `10686262121`;
- digest: `sha256:9d84f03e1b14ad049646fb6880e678ec4d79e8b3970ea591ce74c230eac07629`;
- `r20-full-bundle` artifact ID: `10686996378`;
- digest: `sha256:8aa86a868d4acec99cffde40d8c549bec0074efe3c929cd808b271b77888bede`.

### R20 validation authority

Exact R15 baseline:
- matched accuracy: **0.5513333082**;
- ranked-structural non-entailment accuracy: **0.708**.

Matched label counts:
- entailment: 543;
- neutral: 466;
- contradiction: 491.

Baseline-validity gate: **PASS**.

All data authorities pass:
- structural train: exactly 4,000;
- retention train: exactly 6,000;
- R20 train sets disjoint from each other and every R19 scored-train set;
- structural validation: exactly 500 and disjoint from R18/R19;
- matched validation: exactly 1,500 and disjoint from prior windows.

### Retention-subspace evidence

Numerical retention-subspace rank: **63**.

Basis orthonormal max error:
- **4.61e-12**, well inside the frozen 1e-7 bound.

However the spectrum is extremely concentrated:
- 50% coverage target -> rank 1, realized coverage **0.95048**;
- 75% coverage target -> rank 1, realized coverage **0.95048**;
- 90% coverage target -> rank 1, realized coverage **0.95048**.

Therefore all three preregistered coverage targets collapse to the same rank-1 direction.

Frozen R15→R16 relation-delta L2:
- **3.96121**.

The rank-1 retention direction captures only:
- **4.00926e-05** of the relation-delta energy, about **0.0040%**.

This is the key negative result: the dominant generic CE-gradient-energy direction is almost orthogonal to the actual R16 relation update.

### Candidate anatomy

No candidate is eligible.

Best matched-retention candidate:
- removal strength: 1.0;
- alpha: 0.25;
- matched accuracy: **0.5453333259**;
- structural accuracy: **0.716**;
- structural gain: **+0.008**;
- removed delta energy fraction: **4.00926e-05**.

It passes the absolute 0.53 matched threshold numerically, but fails the same-slice relative floor:
- frozen relative floor: **0.5493333082**.

Best structural candidate:
- removal strength: 0.5;
- alpha: 1.0;
- matched accuracy: **0.5126666427**;
- structural accuracy: **0.754**;
- structural gain: **+0.046**;
- removed delta energy fraction: **1.002315e-05**.

A full removal candidate at alpha 1.0 reaches:
- matched: **0.5206666589**;
- structural: **0.728**;
- structural gain: **+0.020**.

It still fails retention.

Because all three coverage targets resolve to rank 1, the nominal 18-candidate tournament contains only six distinct projected update geometries.

### Frozen gate result

Passed:
- data sizes/disjointness;
- baseline validity;
- matched label support;
- retention basis orthonormality;
- coverage achievement;
- non-relation bit identity;
- parameter count exactly 422,159.

Failed:
- selected candidate eligibility;
- same-slice relative matched retention;
- selected structural gate because no candidate is eligible;
- selected +0.02 structural-gain gate because no candidate is eligible;
- removed-delta-energy >=0.05.

Primary scientific gate: **FAIL**.

The selected output remains the exact R15 head:
`007e24fff0e0e48a096de59276f7ab8d0e25bdb826a3f81a7838c5bd6151723f`.

No HANS/Breaking diagnostic is authorized for R20.

## Failure anatomy and successor constraint

R20 falsifies the specific hypothesis that the competence-damaging component of the R16 relation delta lies in the dominant **generic CE-gradient-energy subspace at exact R15**.

The failure is not caused by insufficient support or an invalid validation authority:
- baseline validity passes;
- all train/validation disjointness gates pass;
- the structural slice is strong and balanced.

The dominant retention direction simply has almost no geometric overlap with the R16 relation delta.

A successor must therefore estimate retention damage **conditioned on the actual R16-induced functional drift**, not merely generic task gradients at R15.

A principled R21 direction is:
- preserve exact R15 as teacher;
- move to a fixed train-only probe point along the frozen R16 relation delta;
- compute batch-wise gradients of `KL(R15 teacher || probe student)`;
- construct a functional damage subspace from those KL gradients;
- project the frozen relation delta away from that damage subspace;
- evaluate on new disjoint MultiNLI development slices;
- do not reuse R20 validation cells for selection.

R20 is a completed negative result, not an unfinished tuning stage.
