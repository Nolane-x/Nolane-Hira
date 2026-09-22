# R23 handoff — finite teacher-KL trust region

Status: **protocol frozen before empirical evaluation in issue #42; implementation in progress on `feat/r23-finite-functional-trust-region`.**

## Evidence entering R23

R22 is a completed negative result:
- exact-R15 same-slice matched: 800 / 1500, accuracy 0.5333333611;
- exact-R15 structural: 0.740;
- rank-32 functional damage subspace remained numerically valid;
- only 0.3134% of the exact-R15 structural-gradient energy survived the rank-32 projection;
- the smallest R16-L2-normalized null candidate reached 0.774 structural (+0.034) but only 751 / 1500 matched correct;
- R22 selected exact R15 and failed the primary gate.

The key successor constraint is finite: a direction can be first-order orthogonal to the measured damage subspace and still leave the safe functional neighborhood at a finite step.

## Frozen R23 mechanism

Issue #42 is authoritative.

R23:
1. reconstructs exact R19/R20/R21/R22 scored-train source indices, requiring a 40,000-example union;
2. selects new disjoint R23 structural proposal train (4,000);
3. selects new disjoint R23 functional trust train (6,000, seed 31);
4. builds the inherited rank-32 teacher-KL damage subspace at the fixed beta=0.25 R16 probe;
5. computes a new exact-R15 structural gradient;
6. removes the rank-32 damage component;
7. keeps the **native unnormalized** null tangent;
8. evaluates only the frozen eta grid {1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4} on the trust train;
9. selects the largest eta that satisfies the finite trust region.

## Finite trust region

A candidate is train-safe only if:
- mean KL(R15 teacher || candidate) <= 1e-4 on all 6,000 trust examples;
- exact teacher argmax agreement is 6000 / 6000;
- logits/probabilities are finite;
- non-relation state remains bit-identical.

Validation cannot participate in eta selection.

## New final validation

Structural:
- validation_mismatched per-label zero-based ranks 1250:1500;
- exactly 500 examples;
- disjoint from all earlier structural windows.

Matched:
- validation_matched shuffle seed 14 positions 7500:9000;
- exactly 1,500;
- disjoint from all earlier matched windows.

Final retention remains zero-drop:
- candidate correct count must be >= exact R15 correct count on the same 1,500 examples.

Primary structural gain remains >= +0.020.

No HANS, Breaking NLI, XNLI, MASSIVE, Banking77, Laya, or Jev/JEV final benchmark cell may select R23.
