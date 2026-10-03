# HIRA V1 S42 handoff — to S43 Cross-View Relative-Gap Geometry Anchoring

S42 is frozen as:

`HIRA_V1_S42_MATCHED_CROSS_VIEW_RELATIONAL_GEOMETRY_DEV_COMPLETE`

Interpretation:
**Case B — correctness retained but relative geometry still collapses.**

## Residual scientific picture

S41:
- strong correctness gain
- agreement largely rescued
- signature discrimination margin collapsed **-0.1201**

S42:
- correctness remains positive
- full KxK MSE anchor is live and optimizer-faithful
- signature discrimination still collapses **-0.09165**
- fused/relation agreement also regress

This implies the missing target is not the entire similarity matrix itself.

The failed metric is inherently a **relative gap**:
same-option similarity versus strongest wrong-option similarity.

## S43 scientific question

> Can S38/S41-style correctness co-adaptation be retained when the actual AdamW runtime movement is constrained against the full set of cross-view same-option-vs-wrong-option similarity gaps?

## Frozen candidate mechanism

Keep exactly:
- S35 native 256D signature representation
- S17 primary/fusion shell
- S38 full bilinear W 256x256 / 65,536
- reference trainable 49,152
- treatment trainable 114,688
- exact S41 stateful AdamW candidate/state engine
- exact actual-step projection
- exact float-target application / rounding guards
- W candidate delta never projected
- no learned anchor parameters
- no anchor coefficient
- no gold labels
- no option-ID feature input

For normalized cross-view signatures:

`G[b,i,j] = cosine(S_c[b,i], S_p[b,j])`

Define row-wise relative-gap tensor:

`R[b,i,j] = G[b,i,i] - G[b,i,j]`

for all `j != i`.

The diagonal entries are excluded from the mean because they are identically zero.

Reference target:
`R_r = stopgrad(relative_gap(G_reference))`

Treatment:
`R_t = relative_gap(G_treatment)`

Anchor:

`A_gap = mean_{b,i,j != i}((R_t - R_r)^2)`

Properties:
- directly preserves each logical option's same-option advantage over every wrong option;
- uses no gold answer;
- uses only matched logical option identity already required by the transport task;
- no strongest-wrong max operator;
- no numeric margin target;
- no diagonal/off-diagonal weighting hyperparameter;
- invariant to a shared logical-option permutation.

## Actual-step semantics

Let `a = grad(A_gap)` on treatment runtime parameters.

Use the exact S41/S42 stateful AdamW candidate.

For candidate runtime movement `delta_r`:
- if `a dot delta_r <= 0`: exact identity
- otherwise remove exactly the anchor-increasing component with epsilon 1e-12

W candidate movement remains unchanged.

Persist AdamW state from the original clipped correctness gradient.

## Required S43-A0

Fresh diagnostic authority must prove:
- zero-init reference/treatment identity;
- exact surfaces 49,152 / 114,688 / W 65,536;
- gap tensor shape and off-diagonal count correct;
- zero loss for identical reference/treatment geometry;
- anchor responds to same-option diagonal perturbation;
- anchor responds to one wrong-option off-diagonal perturbation;
- anchor distinguishes two matrices with equal-ish global MSE but different relative gaps;
- shared logical-option permutation scalar invariance;
- anchor -> reference runtime gradient exactly zero;
- anchor -> W gradient exactly zero;
- anchor -> treatment LoRA nonzero under synthetic gap drift;
- correctness -> W/offdiag-W/LoRA nonzero;
- exact AdamW candidate/state parity;
- actual conflict projection;
- safe-step identity;
- W candidate movement exact identity;
- float-applied movement within rounding bounds;
- applied anchor dot within rounding bound;
- arbitrary K=3/K=7;
- question/padding invariance;
- projection independence;
- checkpoint/probability/full-K/state-once mechanics.

A0 semantic outputs diagnostic only.

## Fresh matched authority

Only after qualified A0 and exact staged-head CI.

Intended:
- seed **64001**
- TRAIN 768
- DEV 192
- 12 wholly fresh S43 domains
- K=4
- 2 state views
- 2 question wording views
- 2 option views
- 24 epochs
- batch 16
- same frozen optimizer semantics
- identical rows/order
- independent runtime/optimizer state

Freshness:
- no exact S0-S42 exposed rows
- no S43-A0 rows
- no M5 final/confirmatory rows
- no W29-W34 sealed rows

## Interpretation

A — correctness retained + relative-gap geometry protected:
correctness remains materially positive while signature discrimination avoids S41/S42-like degradation and agreement/JS do not re-collapse.

B — correctness retained but gap geometry still collapses:
relative-gap anchor is insufficient; close.

C — gap geometry protected but correctness collapses:
the constrained relative-gap movement was required for correctness; close.

D — treatment DEV_READY:
freeze immediately and move to a separate fresh confirmation stage before any external Laya/Jev benchmark.

## Stop rule

After one S43 DEV:
- no strongest-wrong variant
- no hinge/margin target
- no gap weighting
- no mixing with S41/S42 anchors
- no coefficient/slack
- no optimizer reinterpretation
- no partial detach / gradient mixing
- no W-only LR/scheduler
- no rank/factorization
- no scale/bias/nonlinearity
- no seed/LR/epoch/batch retry
- no gate weakening
- no second DEV

Scientific FAIL is valid.
