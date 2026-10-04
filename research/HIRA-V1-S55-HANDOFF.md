# HIRA V1 S55 handoff — to S56 Explicit Cross-View Decision Consistency

S55 closes as **Case C**.

## Evidence chain S51–S55

S51 — query-free identity:
- correctness modestly improved;
- stability did not.

S52 — learned global query canonicalization:
- regressed correctness/stability.

S53 — token-level query↔option late interaction:
- context highly stable;
- selected choices still unstable.

S54 — parameter-free state-query-option interaction:
- correctness improved;
- stability still did not.

S55 — learned joint relation transform:
- canonical correctness improved;
- paraphrase correctness mixed;
- fused/relation selected-choice agreement worsened.

This is enough evidence to stop modifying only the **representation/factorization before the final decision**.

## S56 direction

**S56 — Explicit Cross-View Decision Consistency**

Retain the best mechanically stable backbone:
- persisted S51 native authority;
- query-free option identity;
- immutable shared cache;
- one encoder/state-once;
- full-K;
- correction surface fixed.

Use a matched-capacity pair whose architecture is identical.

Reference:
- existing private correctness objective only.

Treatment:
- same objective plus a direct **decision-distribution consistency** term on paired canonical/paraphrase views.

### Treatment consistency objective

For each semantic case and each relation:
- obtain fused full-K logits for canonical and paraphrase views;
- standardize logits per view to remove pure scale/offset differences;
- compute symmetric JS between the paired full-K distributions.

Add an explicit **ordering consistency** term:
- pairwise option-order signs/margins for canonical vs paraphrase;
- penalize sign disagreement only where either view has a preregistered confidence margin.

This operates on final decisions, not latent query/context geometry.

### Anti-collapse / correctness protection

Decision consistency alone can collapse to uniform predictions.

Therefore treatment must retain the exact correctness CE and relation-binding losses, plus a fixed entropy-floor or margin-preservation condition preregistered before A0.

No post-hoc coefficient sweep.

### Preferred controlled design

Keep architecture/capacity identical and change only auxiliary objective:
- reference decision-consistency coefficient **0**
- treatment decision-consistency coefficient **0.10**
- ordering-consistency coefficient **0.05**
- no extra trainable params.

Use frozen coefficients with no sweep.

### Required A0

- reference/treatment identical architecture + initialization
- extra params 0
- consistency loss exactly zero when paired logits identical
- nonzero on controlled disagreement
- invariant to shared logit offset
- invariant to positive shared logit scale after standardization
- ordering penalty zero when pairwise ordering matches
- ordering penalty nonzero on controlled sign flip
- auxiliary gradients reach private correction params
- auxiliary gradients to native/cache = 0
- full-K K=3/7/255
- option permutation
- probability mass
- one encoder/state-once
- no collapse on synthetic uniform/logit probes.

## S56 interpretation

A — selected-choice stability improves while correctness remains:
decision-level consistency is the missing mechanism.

B — stability improves but correctness collapses:
consistency pressure is too strong / collapses useful discrimination.

C — correctness remains but stability still does not improve:
move to a discrete pairwise-ranking decision family rather than probability consistency.

D — both regress:
reject direct decision-consistency objective.

E — full DEV_READY:
freeze and confirm before external Laya/Jev.

One S56 DEV only.
