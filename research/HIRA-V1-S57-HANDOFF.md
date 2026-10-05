# HIRA V1 S57 handoff — to S58 Consensus-Teacher Pairwise Ranking

S57 closes as **Case B**.

## Evidence chain

S56:
- full-distribution consistency gave strong stability gains;
- correctness collapsed.

S57:
- selective pairwise ordinal consistency improved fused agreement **+8.07 pp**;
- correctness still fell;
- relation agreement worsened **-7.55 pp**.

This means:
**decision-level ranking consistency is useful, but self-generated anchors are unreliable.**

## S58 direction

**S58 — Consensus-Teacher Pairwise Ranking**

Retain:
- exact persisted S51 native authority;
- immutable shared cache;
- one encoder/state-once;
- correction params 114,688;
- added trainable params 0;
- matched architecture/init/optimizer/selector.

### Frozen teacher

Use a frozen reference/private checkpoint or frozen native+reference decision shell as a non-trainable teacher.

Teacher must be fixed before S58 TRAIN/DEV and must not be selected using S58 DEV.

### Consensus target

For every option pair i,j and paired views c,p:

Teacher margins:
`t^c_ij`, `t^p_ij`.

A pair is eligible only if:
1. both teacher views have |margin| >= frozen threshold;
2. both teacher views agree on ordering sign;
3. if gold is involved, the agreed sign must rank gold above the alternative.

Disputed teacher pairs are ignored.

Treatment loss:
- preserve only the teacher consensus sign with a hinge/logistic ordinal margin;
- no full-distribution JS;
- no student-self anchor.

Reference:
- coefficient 0.

Treatment:
- frozen pairwise teacher coefficient **0.05**.

Suggested frozen mechanics:
- teacher active threshold **0.25**
- student target margin **0.05**
- no sweep.

### Why S58 should be safer

Uniform/student flattening does not create more eligible targets.
Incorrect self-generated ordering cannot become its own target.
Only cross-view teacher consensus supplies supervision.

## Required A0

- teacher frozen/no gradients
- teacher checkpoint/hash pinned
- consensus pair active only when both teacher views agree and exceed threshold
- disagreement -> inactive
- low-confidence -> inactive
- wrong gold-order -> inactive
- non-gold consensus pairs remain eligible
- student loss zero when target ordering/margin satisfied
- nonzero on controlled violation
- option permutation equivariant
- offset invariant
- K=3/7/255
- auxiliary gradients to student correction only
- native/cache/teacher gradients 0
- flat student logits do not masquerade as success
- one encoder/state-once.

## S58 interpretation

A — stability improves while correctness/discrimination is retained:
teacher-consensus ordinal targets solve the unsafe-anchor failure.

B — stability improves but correctness still collapses:
teacher itself is not safe enough; move to explicit learned pairwise decision head.

C — correctness remains but stability does not improve:
consensus mask is too conservative; move to explicit pairwise model.

D — both regress:
reject teacher-consensus ranking.

E — full DEV_READY:
freeze and open separate confirmation before external Laya/Jev.

One S58 DEV only.
