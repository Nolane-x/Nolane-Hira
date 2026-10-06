# HIRA V1 S69 → S70 handoff

Parent verdict: **S69 Case B**

## What S69 established

The treatment representation materially improved fresh pairwise evidence:
- gold-pair accuracy **51.74% → 55.16%**
- gold-pair margin **0.01605 → 0.04071**
- pairwise paraphrase aggregate **26.82% → 31.77%**
- pairwise canonical aggregate **27.86% → 28.91%**
- pairwise question-swap **0% → 10.94%**.

But the final fused decision gained only **+0.26 pp** canonical and paraphrase accuracy, while agreement/JS did not improve.

## New bottleneck

Current pairwise aggregate is the uniform row mean:
`s_j = mean_{k!=j} P[j,k]`.

That treats every opponent equally.

S69 shows that the pair matrix contains useful local comparisons, but uniform averaging can dilute them before the downstream bounded scalar gate sees them.

## Required S70 question

> Can a zero-parameter **fused-anchored weighted pairwise aggregation** convert the stronger frozen S69 pair matrix into more useful decision evidence than uniform row-mean aggregation, while preserving permutation equivariance and the exact same downstream gate?

## Proposed matched composition

Freeze:
- S69 treatment representation in **both arms**
- exact S59 pairwise head architecture/training
- same pairwise head params/trajectory between matched arms
- exact S64 contextual gate
- exact S66 per-view responsibility objective
- no target reopening.

Reference pairwise evidence:
`e_ref(j) = mean_{k!=j} P[j,k]`
— exact current uniform aggregation.

Treatment pairwise evidence:
1. obtain detached fused baseline logits `f`;
2. compute `w = softmax(f)` with fixed temperature **1.0**;
3. remove self weight and renormalize opponents:
   `w_{j,k} = w_k / sum_{l!=j} w_l`, k!=j;
4. compute
   `e_treat(j)=sum_{k!=j} w_{j,k} P[j,k]`.

No learned aggregation parameters.

Interpretation:
- comparisons against plausible fused competitors matter more;
- weak irrelevant distractors are downweighted;
- if fused weights are uniform, treatment collapses exactly to reference.

## Controlled variable

Both arms:
- same frozen S69 representation family
- same 32,832-param pairwise head
- same pairwise training trajectory
- same fused baseline/correction trajectory
- same 60-param S64 gate
- same S66 responsibility objective
- same optimizer
- same selector
- same inference parameter count.

Treatment added params: **0**.

## Required A0

Must prove:
- treatment aggregation is permutation equivariant
- uniform fused distribution collapses treatment to reference exactly/tolerance
- diagonal pair entries never contribute
- finite for K=3/7/255
- probability mass finite
- no new trainable tensors
- pairwise/fused inputs detached at aggregation boundary
- gate gradients remain isolated
- alpha0 exact fused identity
- checkpoint replay exact
- no fresh S70 DEV.

## Fresh S70 court

Use:
- seed **91001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh domains
- exact S69 overlap **0**
- K=4
- 24 epochs
- one DEV only.

Primary readout:
1. final canonical/paraphrase accuracy
2. paired both-correct
3. question-swap
4. agreement / JS
5. pairwise gold-pair evidence (must remain matched)
6. weighted-vs-uniform aggregate diagnostics.

## Frozen interpretation direction

**A** — weighted composition materially improves final correctness/stability while pairwise evidence stays matched:
composition bottleneck resolved; open confirmation.

**B** — weighted aggregate diagnostics improve but final decision remains weak:
scalar gate is now the bottleneck; next stage studies a richer correctness-preserving composition operator.

**C** — no material difference:
uniform aggregation was not the bottleneck; move to richer composition rather than another weighting sweep.

**D** — final metrics regress materially:
reject fused-anchored weighted aggregation.

**E** — full DEV_READY:
freeze and confirm before external Laya/Jev evaluation.

No temperature/weighting-function sweep after DEV.
