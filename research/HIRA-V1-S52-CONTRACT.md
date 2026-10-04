# HIRA V1 S52 contract — Paired-View Query Relation Canonicalization

Status: **PREREGISTERED / NO S52-A0 EXPOSURE**

Issue: #287

Parent:
- S51 fresh run `37197889785`
- artifact `11301721375`
- Case **C**
- query-free option identity improves correctness modestly but does not improve stability.

## Scientific question

> With native evidence and query-free option identity held fixed, can a compact paired-view query relation canonicalizer improve cross-view stability while preserving useful correctness?

## Frozen backbone

S52 inherits:
- sealed S51 persisted native authority artifact;
- immutable shared native evidence;
- query-free state↔option identity;
- one encoder/state-once;
- correction A/B/W surface.

Native retraining is not part of the S52 variable.

## Query relation canonicalizer

Both reference and treatment contain the exact same architecture:

`q_raw = normalize(mean(question tokens))`

`q* = normalize(q_raw + B(gelu(A q_raw)))`

Frozen shapes:
- A: `64 x 256`
- B: `256 x 64`
- no bias
- total query-canonicalizer params: **32,768**
- adapter seed: **73052**
- A initialized Gaussian std **0.02**
- B initialized exactly zero
- zero-init output therefore preserves `q_raw` exactly at initialization.

Raw query MUST NOT bypass `q*` in private residual or bilinear correction.

## Matched private surface

Per arm:
- correction params: **114,688**
- query canonicalizer params: **32,768**
- query-free identity params: **0**
- total private trainable params: **147,456**

Both arms:
- bit-identical initialization
- same native/cache bytes
- same TRAIN rows/order
- same optimizer/LR/weight decay/grad clip
- same private correctness CE + fused JS
- same checkpoint selector
- same epochs/batch.

## Controlled variable

Reference:
- relation-code auxiliary coefficient **0.0**

Treatment:
- relation-code auxiliary coefficient **0.10**

No architecture/capacity difference.

## Treatment relation-code auxiliary

For each semantic case, let:
- A1/A2 = canonical/paraphrase questions for relation A
- B1/B2 = canonical/paraphrase questions for relation B.

Same-relation term:
`L_same = 0.5 * [(1-cos(A1,A2)) + (1-cos(B1,B2))]`

Relation centroid:
`A = normalize(A1 + A2)`
`B = normalize(B1 + B2)`

Different-relation hinge:
`L_sep = relu(cos(A,B) - 0.25)`

Treatment auxiliary:
`L_rel = L_same + L_sep`

Treatment total adds:
`0.10 * L_rel`

Frozen separation ceiling: **0.25**.

No coefficient/margin sweep.

## Ownership

Relation-code auxiliary may update **only canonicalizer A/B**.

It must have zero gradient path to:
- persisted native runtime
- immutable cache
- correction adapter A/B
- correction bilinear W
- query-free option identity (zero params).

Private correctness objective may update correction + canonicalizer in both arms.

## Required S52-A0

Architecture:
- correction count 114,688 each;
- canonicalizer count 32,768 each;
- total private trainable 147,456 each;
- identity params 0;
- initialization bit-identical;
- zero-init canonicalizer output equals raw normalized query;
- no raw-query bypass.

Mechanics:
- K=3/7/255
- option permutation
- full-K
- probability mass <=1e-6
- one encoder/state-once
- native optimizer absent
- shared cache bytes exact.

Auxiliary:
- reference auxiliary exactly 0;
- treatment same-relation term finite/nonzero on controlled probes;
- treatment separation hinge finite/nonzero on controlled probes;
- treatment auxiliary gradient reaches canonicalizer;
- auxiliary gradient into correction params exactly 0;
- distinct relation probe does not collapse.

## Fresh S52 authority

Use wholly fresh S52 TRAIN/DEV rows.

Intended:
- seed **73001**
- TRAIN 768
- DEV 192
- 12 fresh domains
- K=4
- private epochs 24
- batch 16
- one DEV only.

Use the already sealed S51 native artifact as a frozen pre-S52 backbone unless A0 exposes an incompatibility.

## Prohibited

No:
- auxiliary coefficient sweep
- separation ceiling sweep
- hidden dimension sweep
- raw-query bypass
- native retraining
- identity variant
- correction capacity change
- alternate selector
- retry
- gate weakening
- second S52 DEV.

Scientific failure is valid.
