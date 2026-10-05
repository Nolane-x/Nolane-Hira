# HIRA V1 S59 handoff — to S60 Calibrated Hybrid Pairwise-Evidence Composition

S59 closes as **Case C**.

## Evidence chain

S56:
- full-distribution consistency improved stability;
- correctness collapsed.

S57:
- self-anchored ordinal pressure improved fused stability;
- unsafe orderings damaged correctness.

S58:
- frozen teacher consensus removed self-anchor;
- stability still improved but correctness collapsed.

S59:
- teacher removed completely;
- explicit gold-supervised anti-symmetric pairwise head;
- selected-choice agreement **+32.55 pp**
- cross-view JS **-0.128684**
- canonical accuracy **-15.36 pp**
- paired both-correct **-21.87 pp**
- question-swap **-66.15 pp**.

Therefore:

**the pairwise signal is useful and stable, but it must not replace absolute fused evidence.**

## S60 direction

**S60 — Calibrated Hybrid Pairwise-Evidence Composition**

Retain:
- exact S51 persisted native authority;
- immutable cache;
- one encoder/state-once;
- existing correction shell;
- explicit S59 pairwise head mechanics;
- gold-supervised pairwise TRAIN objective;
- pairwise gradients isolated from correction/native;
- no teacher/pseudo-target/self-anchor.

New scientific variable:
- a preregistered, capacity-controlled hybrid composer that combines standardized existing fused logits with standardized pairwise aggregate.

The hybrid must preserve absolute evidence while allowing pairwise residuals to repair unstable ordering.

Required properties:
- full-K
- option-permutation equivariant
- shift/positive-scale controlled
- no K-specific parameters
- explicit bounded pairwise influence
- identity path available exactly
- pairwise-only path not used as the reference
- no DEV-based blend tuning
- K=3/7/255 mechanical court
- strict parameter/runtime accounting.

The exact composition rule and coefficient/calibration mechanism must be frozen **before S60-A0 and fresh S60 TRAIN/DEV exposure**.

One S60 DEV only.
