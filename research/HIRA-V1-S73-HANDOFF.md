# HIRA V1 S73 → S74 handoff

Parent verdict: **S73 Case C**

## What the S59–S73 line established

The line has isolated several facts:

1. **S69** showed that query-gated identity interaction materially improves fresh pairwise evidence.
2. **S70** showed scalar opponent weighting is not the main bottleneck.
3. **S71** showed richer row statistics do not rescue a scalar residual gate.
4. **S72** showed changing residual geometry has real effect, but unconditional application is unsafe.
5. **S73** showed detached safety features cannot reliably decide when that residual is safe; the policy collapses to reject-all.

Therefore the architectural problem is deeper than residual composition.

## S74 must replace the decision interface

S74 should test a **Direct Set Arbitration Core (DSAC)** that consumes fused/native evidence and pairwise relational evidence jointly and emits the final full-K decision directly.

It must not be another additive bounded residual.

### Core input per option

For each option j, use detached, permutation-equivariant features:

1. standardized fused logit
2. fused probability
3. S69 pairwise row mean
4. row max
5. row min
6. row RMS
7. positive-win fraction
8. strongest-loss magnitude
9. S69 pairwise identity-query confidence norm
10. exact S59 reference-context norm
11. fused-vs-pairwise rank disagreement
12. option-wise pairwise entropy proxy.

All features are computed state-once / full-K.

### Direct set core

Use one shared permutation-equivariant option encoder:

- input dim: **12**
- hidden dim: **16**
- per-option encoder: Linear(12,16) + tanh
- global set context: concatenate mean-pool and max-pool of hidden => 32
- per-option scorer sees [hidden_j, global_mean, global_max] => 48
- scoring head: Linear(48,1).

Parameter count:
- option encoder: 12×16 + 16 = 208
- score head: 48 + 1 = 49
- total **257 trainable params**.

No K-specific parameters.

### Matched reference/treatment

Both arms use exactly the same 257-param DSAC and bit-identical initialization.

Reference input:
- fused-native channels live;
- pairwise relational channels mechanically zeroed.

Treatment input:
- all 12 channels live.

Thus S74 tests whether direct relational evidence inside the decision core improves decisions, while capacity is exactly matched.

### Output semantics

The DSAC score is the **final arbitration logit**.

Do not add it as a residual to fused logits.

The final distribution is:
`softmax(dsac_scores)`.

The fused system participates only through input features, not through an additive fallback path.

### TRAIN objective

Both arms:
- canonical CE
- paraphrase CE
- paired cross-view JS consistency.

Fixed preregistered loss:
`0.5*(CE_c + CE_p) + 0.10*JS(c,p)`.

No target sweep.

No teacher/pseudo-target.

No DEV target.

### Required A0

Must prove:
- 257 vs 257 params
- bit-identical initialization
- reference pairwise channels exactly zero
- treatment pairwise channels non-degenerate
- permutation equivariance
- finite K=3/7/255
- one encoder/state-once
- full-K
- no K-specific learned tensors
- direct logits are not fused-plus-residual
- gradients reach all DSAC layers
- no gradient into native/correction/pairwise evidence sources
- exact checkpoint replay
- fresh S74 DEV exposed=false.

### Fresh S74 court

- seed **95001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh domains
- exact S73 state/question/option overlap **0**
- K=4
- two semantic option views
- 24 epochs
- one DEV only.

### Primary readouts

1. canonical accuracy
2. paraphrase accuracy
3. paired both-correct
4. question-swap
5. cross-view agreement / JS
6. mean gold margin
7. pairwise gold-pair evidence, which must remain shared/matched
8. DSAC confidence/calibration diagnostics
9. permutation/full-K/state-once gates.

### Frozen interpretation direction

**A** — treatment direct core materially improves correctness/stability over capacity-matched fused-only DSAC:
decision-core bottleneck resolved; open independent confirmation toward v1.0.

**B** — treatment improves some semantic metrics materially but not enough for DEV_READY:
freeze DSAC architecture and run one deeper semantic-evidence improvement stage inside the core, not another residual family.

**C** — relational channels are used but no material benefit:
S69 pairwise evidence is not sufficient for final direct arbitration; return upstream to semantic representation construction.

**D** — treatment materially regresses:
reject DSAC relational integration.

**E** — full DEV_READY:
freeze immediately and open independent confirmation before v1.0/external Laya-JEV evaluation.

## Stop rule

After one S74 DEV:
- no hidden-width sweep
- no channel subset/addition sweep
- no loss-weight sweep
- no activation sweep
- no fused residual fallback
- no target change
- no retry
- no second DEV.

Scientific weakness is valid.
