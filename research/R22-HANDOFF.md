# R22 handoff — functional-null structural tangent

Status: **completed empirical negative result; merged to `main` in `0a948cf57abb4e019998806ec855343d3098317e`.**

## Why R22 existed

R21 established a useful mechanism but failed its primary gate:
- functional teacher-KL projection materially recovered matched retention;
- selected structural gain remained only +0.006 versus the frozen +0.020 requirement.

R22 therefore tested whether a **new exact-R15 structural descent direction**, projected into the measured functional-damage nullspace, could create structural utility without any matched-accuracy loss.

Issue #40 was the preregistered protocol authority.

## Frozen protocol

R22 reconstructed the exact R19/R20/R21 scored-train authorities:
- 4,000 structural + 6,000 retention for each round;
- exact prior union required to contain 30,000 distinct MultiNLI train source indices.

R22 then selected:
- new structural train: 4,000;
- new functional-retention train: 6,000, seed 29;
- new structural validation: validation_mismatched per-label ranks 1000:1250;
- new matched validation: seed-14 positions 6000:7500, exactly 1,500.

The structural direction was:
1. exact-R15 gradient of CE + 0.5 anti-entailment margin;
2. raw descent `-g_struct`;
3. rank-32 teacher-KL functional-damage projection removed;
4. residual tangent normalized to the exact R15→R16 relation-delta L2;
5. exactly four alphas: 0.125, 0.25, 0.50, 1.00.

Unlike R21, R22 had a strict **zero-drop matched correct-count gate**.

## Authoritative empirical result — PRIMARY FAIL

Authoritative workflow run:
- `35721515966`;
- empirical head: `3661abaa698eb9f583cd010ecaa78d074d75fca9`.

Artifacts:
- evidence ID `10691374526`;
- evidence digest `sha256:12ae9f968229f41a45fd593554c0934828fcdf0dc6266a07effd55e39c9aef38`;
- full bundle ID `10691704213`;
- full bundle digest `sha256:acefaa02026e68b03b5494b3acd6e87bc498ac70e0910920b38077a613a2cccc`.

Execution correctness:
- unit tests: PASS;
- CI Python 3.10: PASS;
- CI Python 3.12: PASS;
- preflight: PASS;
- exact R15/R16 endpoint hashes: PASS;
- tournament execution: PASS;
- artifacts: PASS.

Scientific primary gate: **FAIL**.

## Same-slice exact R15 baseline

Matched:
- accuracy: **0.5333333611**;
- exact correct count: **800 / 1,500**;
- label support: entailment 546, neutral 488, contradiction 466.

Structural:
- ranked non-entailment accuracy: **0.740**.

Baseline validity: **PASS**.

## Functional-damage geometry

The teacher-KL authority remained numerically healthy:
- probe mean KL: **0.0010211151**;
- functional subspace rank: **63**;
- basis orthonormal error: **1.2537e-10**;
- null re-projection leakage fraction: **3.0178e-23**.

The decisive result is the structural-gradient overlap.

Exact-R15 structural gradient L2:
- **5.3597995**.

Energy remaining after removing the rank-32 functional-damage component:
- **0.0031340588 = 0.3134%**.

Frozen minimum:
- **5%**.

Therefore approximately **99.6866% of the R22 structural-gradient energy lies inside the measured rank-32 functional-damage span**.

The mechanism-validity gate fails here.

## Candidate anatomy

No R22 candidate is eligible.

Smallest null-tangent candidate:
- alpha: 0.125;
- applied update L2: **0.4951517**;
- first-order predicted structural benefit: positive;
- structural: **0.774**;
- same-slice structural gain: **+0.034**;
- matched: **0.5006667**;
- exact correct count: **751 / 1,500**;
- competence loss: **49 correct predictions**;
- zero-drop retention: FAIL.

The larger null-tangent candidates lose progressively more matched competence.

This means R22 simultaneously shows:
1. a real structural signal exists in the tiny measured null residual;
2. amplifying that tiny residual into a finite step leaves the safe functional neighborhood.

## Diagnosis controls

Raw structural tangent:
- collapses strongly even at the smallest preregistered scale;
- confirms that unprotected structural descent is overwhelmingly competence-damaging.

Protected R16 control, rank 32 / removal 0.5 / alpha 0.25:
- matched: **0.5320**;
- structural: **0.738**.

It remains close to R15 but provides no structural gain on this R22 slice.

## What R22 falsifies

R22 falsifies this specific hypothesis:

> a single first-order rank-32 functional-damage nullspace, measured at the fixed R16 probe and followed by R16-L2 normalization, is sufficient to produce a finite structural update that preserves exact matched competence.

R22 does **not** falsify functional teacher-KL damage modelling itself.

The new evidence says the problem is now a **finite-step constraint problem**, not merely a direction-separation problem.

## Successor constraint

A successor must not:
- extend the R22 alpha grid post hoc;
- lower the zero-drop gate;
- reuse R22 validation for selection;
- normalize a tiny null residual to the full R16 relation-delta norm without a train-only finite-drift check.

A principled R23 direction is:
- new disjoint structural + retention train authorities;
- generate a structural proposal direction from exact R15;
- use the teacher-KL retention set itself to enforce a **finite functional trust region** at each candidate step;
- choose step size using train-only teacher-KL / teacher-decision preservation, never validation;
- evaluate exactly once on new disjoint validation windows;
- preserve zero-drop matched competence as the final gate.

No HANS, Breaking NLI, XNLI, MASSIVE, Banking77, Laya, or Jev/JEV final benchmark cell was used for R22 selection.
