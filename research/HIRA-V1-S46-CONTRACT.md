# HIRA V1 S46 contract — Robust Three-Expert Evidence Consensus

Status: **PREREGISTERED / NO S46-A0 EXPOSURE**

Issue: #275

Parent:
- S45 fresh run `37128945285`
- artifact `11276882786`
- interpretation **Case C**
- S45 treatment DEV_READY false

## Scientific question

> Can a zero-parameter robust decision shell preserve the useful S45 private correction signal while preventing one unstable expert from dominating the final full-K decision?

## Frozen training mechanics

S46 does **not** change how the native runtime or private correction expert are trained.

Reuse exact S45:
- native trainable surface **49,152**
- private A/B **49,152**
- full W **65,536**
- correction-only **114,688**
- treatment total **163,840**
- one encoded batch / state-once
- correction inputs detached from native runtime
- correction objective:
  `0.10 * CE_corr + 0.25 * JS_corr`
- exact S45 native optimizer
- exact S45 correction optimizer
- exact S45 native/correction ownership split.

This isolates the new scientific variable to the decision shell.

## Three experts

For each already-encoded query view, expose:

1. `p` — primary triadic logits;
2. `n` — native relation logits before private correction;
3. `c` — corrected private relation logits.

No extra encoder call is permitted.

## Frozen robust consensus operator

Each expert is independently standardized with the existing S14 primitive:

`std(x) = (x - mean_K(x)) / RMS_K(x)`

using the existing fusion epsilon **1e-6** and exact flat-vector handling.

Let:
- `P = std(p)`
- `N = std(n)`
- `C = std(c)`

Stack along expert axis:

`E = stack(P, N, C) [B,3,K]`

S46 fused evidence:

`F46[b,k] = median(E[b,:,k])`

The median is coordinate-wise across the three experts.

No:
- learned parameter
- learned or fitted scalar
- confidence threshold
- entropy weight
- temperature
- gate
- routing network
- trimmed-mean alternative
- second encoder pass.

## Frozen baseline counterfactual

At the same runtime and correction checkpoint, compute the exact legacy S45 shell:

`F45 = 0.5 * (std(p) + std(c))`

S46 scientific interpretation is based primarily on **same-checkpoint F46 minus F45**, not on cross-run absolute scores.

This isolates the decision-family effect from representation/training changes.

## Epoch/checkpoint rule

Training trajectory and checkpoint selection remain exact S45 mechanics.

The selected checkpoint is frozen by the existing S45 selection rule.

After checkpoint selection:
- replay exact selected checkpoint;
- evaluate legacy S45 shell and S46 median shell on the same single fresh DEV exposure;
- no shell-based re-selection of epoch;
- no second DEV.

## Required S46-A0

Mechanical:
- fusion parameter count exactly 0;
- K=3, K=7, K=255;
- exact logical-option permutation equivariance;
- independent additive-shift invariance for each expert;
- independent positive-scale invariance for each expert;
- all-three-identical standardized identity;
- coordinate-wise median lies between min and max expert values;
- if one expert is an arbitrarily extreme outlier and the other two agree, output equals the agreeing value;
- finite output with one flat expert;
- deterministic output;
- softmax probability mass <= 1e-6 error.

Runtime:
- primary/native/corrected logits all come from one encoded batch;
- corrected private expert parameter surface unchanged;
- no extra trainable parameters;
- no second encoder;
- checkpoint compatibility.

Ownership:
- S46 shell itself introduces no trainable gradient surface;
- S45 correction CE+JS still reaches A/B/W only;
- correction -> native gradient remains zero;
- native objective -> A/B/W remains zero.

A0 semantic values are diagnostic only.

## Fresh matched authority

Only after:
1. S45 merge;
2. contract frozen;
3. interpretation plan frozen;
4. qualified S46-A0;
5. A0 receipt frozen;
6. wholly fresh S46 TRAIN/DEV authority frozen;
7. trainer/workflow frozen;
8. exact staged-head generic CI PASS;
9. separate one-shot TRAIN/DEV marker.

Intended:
- seed **67001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S46 domains
- K=4
- 24 epochs
- batch 16
- identical native rows/order
- one DEV only.

## Stop rule

After one S46 DEV:
- no mean/median/trimmed-mean sweep
- no confidence threshold
- no entropy weighting
- no temperature
- no learned gate
- no capacity change
- no optimizer change
- no native-gradient leakage
- no second encoder
- no seed/LR/epoch/batch retry
- no gate weakening
- no second DEV.

Scientific failure is valid.
