# HIRA V1 S56 handoff — to S57 Discrete Pairwise Ranking Consistency

S56 closes as **Case B**.

## Critical evidence

S56 demonstrates:
- direct decision-level consistency **can** increase selected-choice stability;
- forcing full standardized distributions together causes severe correctness collapse.

Therefore the next operator should constrain only the **relative ordering of options**, not entire probability vectors.

## S57 direction

**S57 — Discrete Pairwise Ranking Consistency**

Retain:
- exact persisted S51 native authority;
- immutable shared cache;
- one encoder/state-once;
- full-K;
- correction params fixed at 114,688;
- architecture/capacity identical between arms.

Reference:
- existing private correctness objective only.

Treatment:
- same objective plus a pairwise ordinal consistency loss.

### Pairwise ordinal target

For paired canonical/paraphrase logits `z^c, z^p`, for every option pair `i,j`:

`m^c_ij = z^c_i - z^c_j`
`m^p_ij = z^p_i - z^p_j`

Activate a pair only when one or both views have a preregistered absolute margin above threshold.

Do **not** force margins to have the same magnitude.

Instead penalize only sign disagreement and too-small preserved ordering:

`L_rank = mean softplus(margin_floor - sign(stopgrad(m_anchor)) * m_other)`

with a symmetric canonical↔paraphrase form.

Anchor sign must be detached so the objective cannot improve itself by moving the target sign.

### Correctness protection

Exclude pairwise constraints whose anchor ordering contradicts the gold-vs-option relation unless the gold is not involved.

Additionally retain exact CE/relation correctness losses.

This prevents consistency from locking in obviously wrong gold orderings.

### Frozen first court

Suggested preregistration:
- rank coefficient **0.05**
- activation threshold **0.25**
- preserved margin floor **0.05**
- no JS decision-consistency term
- no coefficient/threshold sweep.

Reference coefficient 0.
Treatment coefficient 0.05.
Added trainable params 0.

## Required S57-A0

- identical architecture/init/capacity
- pairwise loss exactly zero on sufficiently matching ranking
- nonzero on controlled sign flip
- invariant to shared logit offset
- positive shared scale preserves signs
- detached anchor verified
- wrong gold-involving anchor pair filtering verified
- all non-gold pairs remain eligible
- option permutation equivariance
- K=3/7/255
- finite full-K behavior
- auxiliary gradients reach private correction
- native/cache gradients 0
- uniform/flat logits do not masquerade as good ranking stability
- one encoder/state-once.

## S57 interpretation

A — selected-choice stability improves while correctness is retained:
ordinal consistency is the right decision-level constraint.

B — stability improves but correctness still collapses:
ranking pressure still propagates bad anchors; next test teacher/consensus anchoring.

C — correctness remains but stability does not improve:
pairwise ordinal consistency is too weak; move to explicit pairwise decision model.

D — both regress:
reject this objective.

E — full DEV_READY:
freeze + confirmation before external Laya/Jev.

One S57 DEV only.
