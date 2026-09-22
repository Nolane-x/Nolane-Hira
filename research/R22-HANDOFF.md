# R22 handoff — functional-null structural tangent

Status: **protocol frozen before empirical evaluation in issue #40; implementation in progress on `feat/r22-functional-null-structural-tangent`.**

## Why R22 exists

R21 is a completed negative result with a retained mechanism:
- exact R15 same-slice matched: 0.5326666832;
- exact R15 same-slice ranked structural: 0.696;
- functional teacher-KL mechanism validity: PASS;
- selected R21 candidate matched: 0.5306666493;
- selected R21 structural: 0.702;
- selected structural gain: +0.006;
- frozen primary requirement: +0.020;
- primary scientific gate: FAIL.

The useful R21 signal is that teacher-KL projection recovered about +1.07 matched-accuracy points relative to the same-alpha unprojected R16 update. The remaining failure is not lack of projection strength: stronger projected R21 candidates reached up to 0.732 structural while losing competence.

R22 therefore stops tuning the R16 delta and asks whether **new structural utility can be generated directly in the measured functional-damage nullspace**.

## Frozen protocol authority

Issue #40 is authoritative.

Core mechanism:
1. reconstruct exact R19/R20/R21 scored-train source-index union (must be exactly 30,000);
2. select new disjoint R22 structural train (4,000) and retention train (6,000; seed 29);
3. compute exact-R15 structural gradient using CE + 0.5 anti-entailment margin loss;
4. build rank-32 functional teacher-KL damage subspace at the inherited beta=0.25 R16 probe;
5. remove the rank-32 damage component from the raw structural descent direction;
6. normalize the residual tangent to the frozen R16 relation-delta L2;
7. evaluate exactly four alphas: 0.125, 0.25, 0.50, 1.00.

## Stronger competence gate

R22 removes R21's -0.002 same-slice tolerance.

An R22 candidate is not eligible if it loses even one correct prediction relative to exact R15 on the new 1,500-example matched authority.

This is an exact same-slice correct-count comparison, not a float32 threshold approximation.

## New validation

Structural:
- validation_mismatched;
- per-label zero-based ranks 1000:1250;
- 500 examples total;
- disjoint from starts 0, 250, 500, 750.

Matched:
- validation_matched shuffled with seed 14;
- positions 6000:7500;
- exactly 1,500;
- disjoint from all prior positions 0:6000.

## Mechanism safeguards

The rank-32 null tangent is valid only if:
- probe KL > 1e-4;
- functional rank >= 32;
- gradient rows finite;
- basis orthonormal error <= 1e-7;
- structural gradient finite and nonzero;
- at least 5% of raw structural-gradient energy remains after projection;
- re-projected leakage energy <= 1e-8.

## Primary pass

Selected candidate must:
- be eligible;
- preserve zero-drop matched correct count;
- structural non-entailment accuracy >= 0.60;
- structural gain >= +0.020 versus same-slice exact R15;
- preserve non-relation bit identity and 422,159 total HIRA parameters.

No HANS, Breaking NLI, XNLI, MASSIVE, Banking77, Laya, or Jev/JEV final cell may select R22.

A workflow success is not a scientific PASS. A failed empirical run must be preserved unchanged.
