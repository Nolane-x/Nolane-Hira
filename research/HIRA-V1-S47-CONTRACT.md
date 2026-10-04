# HIRA V1 S47 contract — Ordinal Pairwise Consensus with Private Tie-Break

Status: **PREREGISTERED / NO S47-A0 EXPOSURE**

Issue: #277

Parent:
- S46 fresh run `37163095749`
- artifact `11289416980`
- interpretation **Case C**
- S46 robust DEV_READY false

## Scientific question

> Can Hira remove cross-expert score-magnitude interference entirely and recover stable final decisions using only ordinal pairwise consensus, while preserving the useful corrected-private ordering signal and adding zero trainable parameters?

## Frozen training mechanics

S47 changes no training behavior.

Reuse exact S45/S46 training mechanics:
- native trainable surface **49,152**
- private adapter A/B **49,152**
- full W **65,536**
- correction-only **114,688**
- treatment total **163,840**
- one encoded batch / state-once
- detached correction ownership
- correction objective `0.10 * CE_corr + 0.25 * JS_corr`
- exact native and correction optimizers
- checkpoint selected by the existing S45 rule before S47 shell evaluation.

## Three experts

At the selected checkpoint, expose the same one-pass full-K experts:

1. `p` — primary triadic logits;
2. `n` — native relation logits;
3. `c` — corrected private relation logits.

No extra encoder call is permitted.

## Frozen ordinal operator

For expert logits `x`, define pairwise ordinal comparison:

`R_x[i,j] = sign(x[i] - x[j])`

with exact ties represented as 0.

Three-expert pairwise majority:

`M[i,j] = sign(R_p[i,j] + R_n[i,j] + R_c[i,j])`

Copeland score:

`C[i] = sum_j M[i,j]`

Corrected-private ordinal tie-break:

`T[i] = sum_j R_c[i,j]`

Let `B = 2K - 1`.

Final S47 evidence:

`F47[i] = B * C[i] + T[i]`

The bound is derived from K:
- `T[i] in [-(K-1), K-1]`
- maximum pairwise difference between two private tie-break scores is `2(K-1)=2K-2`
- therefore any one-unit Copeland advantage dominates every possible private tie-break difference because `B=2K-1`.

No score magnitude enters after the ordinal comparisons.

## Frozen baseline counterfactual

At the exact same selected checkpoint, compute:
- legacy S45 shell:
  `0.5 * (std(primary) + std(corrected_relation))`
- S47 ordinal shell:
  pairwise majority/Copeland/private ordinal tie-break above.

Primary scientific comparison:
**same-checkpoint S47 minus legacy S45**.

No shell may influence checkpoint selection.

## No tuning surface

No:
- Borda alternative
- alternative Copeland weighting
- majority threshold
- expert weight
- rank temperature
- score normalization
- confidence threshold
- learned gate
- learned/fitted scalar
- alternate tie-break expert
- second encoder pass.

## Required S47-A0

Mechanical:
- decision trainable parameter count exactly 0
- K=3, K=7, K=255
- full-K output
- exact logical-option permutation equivariance
- independent positive-affine invariance for each expert
- strictly increasing nonlinear rank-transform invariance
- arbitrary positive magnitude blow-up invariance
- two-identical-expert pairwise dominance over one adversarial expert
- private tie-break cannot overturn a strict Copeland advantage
- deterministic private ordinal tie-break when Copeland ties
- no raw-magnitude dependency after ordinal extraction
- finite softmax and probability-mass error <= 1e-6.

Runtime:
- all three experts from one encoded batch
- no second encoder
- state-once
- no extra correction parameters
- checkpoint compatibility
- exact native/private ownership inherited.

A0 semantic values are diagnostic only.

## Fresh matched authority

Only after:
1. S46 merge;
2. contract frozen;
3. interpretation plan frozen;
4. qualified S47-A0;
5. frozen A0 receipt;
6. wholly fresh S47 TRAIN/DEV authority;
7. trainer/workflow frozen;
8. exact staged-head generic CI PASS;
9. separate one-shot TRAIN/DEV marker.

Intended:
- seed **68001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S47 domains
- K=4
- 24 epochs
- batch 16
- one DEV only.

## Stop rule

After one S47 DEV:
- no Borda-vs-Copeland sweep
- no alternate pairwise rule
- no tie-break expert sweep
- no majority threshold
- no rank temperature
- no scalar weight
- no learned gate
- no training/capacity/optimizer change
- no native-gradient leakage
- no second encoder
- no retry
- no gate weakening
- no second DEV.

Scientific failure is valid.
