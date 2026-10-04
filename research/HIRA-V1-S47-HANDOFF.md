# HIRA V1 S47 handoff — to S48 Query-Quotient Option Evidence

S47 is frozen as **Case C**.

## What S47 established

At the exact same selected checkpoint:
- legacy S45 agreement: **0.6432292**
- S47 ordinal agreement: **0.5078125**
- legacy JS: **0.0441499**
- S47 ordinal JS: **0.3377110**

S47 was mathematically invariant to score magnitude:
- affine transform error 0
- nonlinear monotonic transform error 0
- extreme magnitude error 0

Yet stability became substantially worse.

Therefore the downstream fusion layer is no longer the main bottleneck.

## New diagnosis

The experts do not merely assign different score scales to the same ordering.

They frequently **change the ordering itself across question views**.

Evidence:
- primary cross-view ordering is not stable enough to support consensus;
- native relation has very low JS but only **0.3932** selected-choice agreement;
- corrected private relation has **0.53125** agreement;
- only **32-37%** of S47 expert pairwise comparisons are unanimous across the three experts at each view.

A low-JS expert can still be semantically wrong and rank-inconsistent.

## S48 scientific direction — Query-Quotient Option Evidence

Do not add another downstream aggregator.

Instead split each option's evidence into:

1. **state-option invariant content** — what the state says about this option;
2. **query relation direction** — what semantic relation the question asks for.

The question branch must be quotiented so paraphrases that request the same latent relation map to the same relation code before option ranking.

### Proposed frozen family

For each query:
- derive the existing question-conditioned relation representation;
- derive a **relation-direction code** by projecting the query representation onto the option-signature difference subspace;
- normalize this code by direction only, discarding query-specific magnitude;
- compute option evidence from the dot/cosine alignment between this normalized relation direction and each option's state-conditioned signature.

Then the private correction branch may act on this quotient evidence, not on raw question tokens.

The scientific variable is upstream:
**raw question-token-conditioned correction vs relation-direction-quotiented correction**.

### Why this family is different

S44-S47 preserved the same question-conditioned correction pathway and only changed:
- representation capacity;
- JS objective;
- score fusion;
- rank fusion.

S48 changes the **semantic interface into the correction branch** so paraphrase surface form is factored out before option ordering.

## S48-A0 requirements

Before any DEV:
- one encoder pass/state-once;
- no extra LLM/encoder;
- option permutation equivariance;
- query paraphrase quotient identity on synthetic relation-equivalent probes;
- relation-distinct queries remain separable;
- quotient scale/sign conventions are deterministic;
- no DEV-derived threshold;
- no increased native trainable surface unless separately preregistered;
- native trajectory ownership remains exact;
- corrected branch receives no raw question-token path in the treatment family;
- K=3/7/255;
- finite outputs/probability mass.

## Fresh scientific court

Use wholly fresh S48 A0 and TRAIN/DEV rows.

Primary comparison must be matched:
- same native runtime/checkpoint-selection mechanics;
- reference correction = existing S45 raw-question-token private correction;
- treatment correction = S48 query-quotient private correction;
- compare correctness + cross-view ordering/agreement.

One DEV only.

No post-DEV quotient dimension/normalization/temperature sweep.

If S48 still fails, the bottleneck is likely deeper in state-option signatures/native semantic core rather than the query surface.
