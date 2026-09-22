# R21 handoff — functional teacher-KL damage-subspace projection

Status: **completed empirical negative result; mechanism-valid, primary gate failed; merged to `main` in `55f504d992d473494b3ccf4a17aeda4acad30a0a`.**

## Evidence entering R21

R15 remains the exact competence anchor:
- head SHA-256: `007e24fff0e0e48a096de59276f7ab8d0e25bdb826a3f81a7838c5bd6151723f`;
- HIRA parameter count: exactly 422,159.

R16 remains a failed structural-repair training result, but its failure-analysis head defines the fixed relation direction:
- head SHA-256: `dbe0ddd8bf3811c98f5062bbf482f5d5b4f1f5991fa6f734f7ca88c24e88cc75`.

R20 is preserved as a negative result:
- R15 baseline on R20 authority: matched 0.55133, structural 0.708;
- generic CE-gradient retention subspace numerical rank: 63;
- top direction explains 95.05% of generic retention-gradient energy;
- that direction captures only about 0.0040% of frozen R16 relation-delta energy;
- no R20 candidate is eligible.

R21 therefore tests a different mechanism: functional output drift induced by the actual R16 delta.

## Frozen protocol authority

Issue #36 is authoritative.

Key frozen choices:
- exact R15 teacher;
- fixed probe beta = 0.25 along the R15→R16 relation delta;
- teacher-KL gradients only on R21 train retention data;
- KL batch size 96;
- exact projection ranks 1, 4, 16, 32;
- removal strengths 0.50 and 1.00;
- application scales 0.25, 0.50, 1.00;
- exactly 24 projected candidates when numerical rank supports all four ranks;
- unprojected controls alpha 0.25, 0.50, 1.00 are diagnosis-only;
- new validation structural ranks 751–1000 per label;
- new matched seed-14 positions 4500:6000;
- HANS/Breaking/XNLI/MASSIVE/Banking77/Laya/Jev final cells forbidden for selection.

## Prior-train reconstruction

R21 implementation deterministically reconstructs:
1. R19 structural train;
2. R19 retention train seed 13;
3. R20 structural train using the exact exclusion-first rule;
4. R20 retention train seed 17.

Their exact union must contain 20,000 distinct source indices.

R21 then:
- excludes that union before structural ranking;
- selects strongest 2,000 neutral + 2,000 contradiction;
- samples 6,000 retention examples with seed 23 after excluding prior train + R21 structural train.

R21 train authorities must be pairwise disjoint and disjoint from every R19/R20 scored-train example.

## Functional teacher-KL construction

On the R21 retention cache:
1. exact R15 computes frozen teacher probabilities;
2. probe state is `R15 + 0.25 * relation_delta`;
3. only relation parameters require gradients;
4. per batch compute `KL(R15_teacher || probe_student)`;
5. flatten the batch-mean relation gradient;
6. weight row by `sqrt(n_b/N)`;
7. stack rows into `G_KL`;
8. construct the Gram eigenspace using the existing numerically checked subspace implementation.

Probe-signal gates:
- mean teacher KL >1e-4;
- every weighted gradient row finite;
- numerical subspace rank >=32.

Functional basis orthonormal error must be <=1e-7.

## Mechanism gate

Before candidate selection:
- project the frozen relation delta onto rank-32 functional damage subspace;
- projected component energy fraction must be >=0.005.

If this fails, R21 is a scientific negative result even if a candidate happens to score well.

## Candidate selection

Eligibility requires:
- valid R21 matched baseline;
- valid functional mechanism;
- matched >=0.53;
- matched >= same-slice R15 -0.002;
- positive train-only first-order structural predicted benefit;
- nonzero removed delta energy.

Selection:
1. maximize ranked-structural non-entailment accuracy;
2. tie-break matched accuracy;
3. smaller applied update L2;
4. smaller alpha;
5. larger removal strength;
6. lower projection rank.

## Primary structural gate

Selected candidate must:
- structural accuracy >=0.60;
- improve same-slice exact R15 structural accuracy by >=+0.02 absolute;
- preserve all frozen retention and integrity gates.

## Implementation files

R21 adds:
- `src/nmd/functional_damage.py`;
- `tests/test_functional_damage.py`;
- `scripts/r21_functional_kl_damage_subspace.py`;
- `.github/workflows/r21-functional-kl-damage-subspace.yml`;
- this handoff.

Existing R20 projection geometry is reused rather than rewritten.

## Interpretation discipline

A pass supports only the narrow R21 mechanism claim on new disjoint MultiNLI development authorities.

It does not establish Laya parity, Jev/JEV parity, broad OOD robustness, multilingual generalization, or general reasoning competence.

No adapted HANS/Breaking diagnostic is authorized unless every R21 primary gate passes.

A failed R21 must be preserved exactly as a negative result.


## Authoritative empirical result — PRIMARY GATE FAIL

Authoritative workflow run: `35715962261`.

Execution authority:
- empirical head SHA: `c6db3262dd6f43da9167717b429ce7bdbc39bb85`;
- merge commit: `55f504d992d473494b3ccf4a17aeda4acad30a0a`;
- evidence artifact ID: `10689827337`;
- evidence digest: `sha256:6a9c8dbd0a94f7ab0fc7a80817d3f30a8ed3b24c992a37cdc261ca14834cd785`;
- full bundle artifact ID: `10689792271`;
- full bundle digest: `sha256:3c9e55ba0dc3108f2abf64f00ea5d371f312a1ae186d81167693c7fdcde4961a`.

### Same-slice R15 authority

Exact R15 baseline:
- matched accuracy: **0.5326666832**;
- ranked-structural non-entailment accuracy: **0.696**.

Matched label support:
- entailment: 547;
- neutral: 482;
- contradiction: 471.

Baseline validity: **PASS**.

### Functional-damage mechanism

The fixed beta=0.25 probe produces:
- mean teacher KL: **0.0010153648**;
- functional subspace numerical rank: **63**;
- basis orthonormal max error: **2.09e-10**;
- rank-32 projected relation-delta energy fraction: **0.0057529096 = 0.57529%**.

Every frozen probe/mechanism gate passes.

This is materially different from R20: the functional teacher-KL subspace overlaps the actual R16 relation delta enough to satisfy the preregistered mechanism gate.

### Selected candidate

Only one projected candidate is eligible:
- projection rank: 32;
- removal strength: 0.5;
- alpha: 0.25;
- matched accuracy: **0.5306666493**;
- ranked-structural non-entailment accuracy: **0.702**;
- structural gain over same-slice R15: **+0.006**;
- first-order structural predicted benefit: **0.0797620**;
- applied update L2: **0.988165**;
- relative-retention gate: **PASS** under the frozen exact finite-sample authority (3 fewer correct predictions out of 1,500, exactly the 0.002 allowance);
- non-relation parameters: bit-identical;
- parameter count: unchanged at exactly 422,159.

Selected head SHA-256:
`24651caf478f44cf0ebd08cadc2b5ac7def9f79184a7f2ad5bc1a0c7f6e558c3`.

### Projection benefit is real but insufficient

Same alpha=0.25 unprojected control:
- matched: **0.5200**;
- structural: **0.706**.

Selected KL-projected candidate:
- matched: **0.53067**;
- structural: **0.702**.

Therefore the functional KL projection recovers about **+1.07 percentage points matched accuracy** relative to the same-alpha unprojected R16 update, while giving up only 0.004 structural on this slice.

However stronger projected candidates expose the remaining tradeoff:
- highest structural projected candidate: **0.732** structural, but only **0.504** matched;
- rank-32 removal 0.5 alpha 1.0: **0.706** structural, **0.506** matched;
- no stronger candidate preserves retention.

### Frozen gate result

Passed:
- train and validation sizes/disjointness;
- baseline validity and label support;
- probe signal;
- functional subspace rank;
- basis orthonormality;
- mechanism overlap;
- selected candidate eligibility;
- absolute matched floor;
- exact finite-sample relative matched floor;
- structural absolute floor;
- non-relation bit identity;
- parameter-count integrity.

Failed:
- preregistered structural gain >= +0.020.

Observed selected structural gain: **+0.006**.

Primary scientific gate: **FAIL**.

No HANS/Breaking diagnostic is authorized for R21.

## Failure anatomy and successor constraint

R21 does **not** falsify functional teacher-KL damage modelling. It establishes that the mechanism can materially protect retention.

It does falsify the narrower hypothesis that projecting the frozen R16 delta away from that damage subspace, by itself, is enough to reach the preregistered structural gain while retaining competence.

A successor must therefore:
1. keep a functional-damage constraint derived only from train data;
2. stop searching the R21 rank/removal/alpha grid post hoc;
3. generate **new structural utility inside the measured functional-damage nullspace**;
4. use new disjoint train and validation authorities;
5. preserve R15 as the competence anchor;
6. preserve negative results and forbid Laya/Jev benchmark cells from selection.

R21 is closed as a completed negative result with a useful retained mechanism.
