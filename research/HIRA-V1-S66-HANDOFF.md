# HIRA V1 S66 → S67 handoff

Parent verdict: **S66 Case B**

## What changed at S66

S66 proved that view-specific responsibility exists:
- final TRAIN target disagreement **0.305338542**
- reference mean alpha **0.0785312**
- treatment mean alpha **0.0668238**
- reference alpha std **0.0028190**
- treatment alpha std **0.0078778**.

Yet selected DEV improved JS only **0.00064353** and left correctness/agreement unchanged.

## Bottleneck

S66 asks only one binary question per view:

> Is the view safe and stability-improving at alpha=0.35?

That discards the most important remaining information:

> **How much** pairwise residual is safe and useful for this view?

A view that is unsafe at 0.35 may still benefit at 0.0875 or 0.175. A view that is safe at 0.35 may have its best stability point at a smaller alpha.

## Required S67 question

> Can a TRAIN-only per-view safe oracle-alpha target teach the existing gate the appropriate residual strength, without changing capacity or inference architecture?

## Preregistered direction

Keep:
- S51 native authority
- shared correction/head trajectory
- S59 pairwise head
- S64 60-param contextual single-view gate
- single-view inference
- alpha max 0.35
- hard correctness safety
- detached ownership
- frozen selector.

Reference:
- exact S66 binary per-view responsibility BCE.

Treatment:
- evaluate a frozen alpha lattice for each view independently while the other view remains baseline:
  `{0, 0.0875, 0.175, 0.2625, 0.35}`.
- candidate is feasible only if own-view CE <= baseline CE + 1e-8.
- among feasible candidates, choose alpha giving the lowest paired JS.
- require JS improvement >1e-8; otherwise choose alpha*=0.
- ties resolve to the **smaller alpha**.
- soft target `y = alpha*/0.35`, therefore y in `{0,0.25,0.5,0.75,1}`.
- train the same gate with BCEWithLogits against this continuous target separately for canonical and paraphrase.

No learned target model.  
No coefficient.  
No temperature.  
No DEV target.  
No parameter advantage.

## Why this is the next controlled experiment

S67 changes only **supervision granularity**:
- S66: safe/useful at full probe? yes/no.
- S67: which preregistered safe residual strength gives the best one-sided stability?

If S67 still produces only microscopic transfer, the next bottleneck is not label granularity and should not be attacked by another target refinement sweep.

## Discipline

S67 must preregister A0, fresh authority and one DEV only before any scientific exposure.
