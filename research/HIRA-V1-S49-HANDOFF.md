# HIRA V1 S49 handoff — to S50 Shared-Native Forked Private Readouts

S49 is closed as **INVALID MATCHED COURT / INCONCLUSIVE**.

## Why S49 cannot answer its scientific question

Both arms eventually exposed full DEV, but the required native runtime trajectory identity failed for all 24 epochs.

Reference selected epoch 6.
Treatment selected epoch 5.

Although selected metrics are available, the difference cannot be attributed solely to:
question-conditioned native relation signature vs query-free state↔option identity.

The private-variable comparison is confounded by different native trajectories.

## S50 design principle

Do **not** train two native arms.

Create exactly **one shared native training trajectory** and make it the common immutable evidence source for both private correction branches.

### S50 — Shared-Native Forked Private Readouts

Phase 1 — shared native authority:
- train the frozen native S45/S49 runtime once on fresh S50 TRAIN;
- use the frozen checkpoint-selection rule for native authority;
- no private correction branch participates in native optimizer updates;
- materialize/cache, for every TRAIN/DEV row and both state/question views:
  - native logits
  - native relation signatures
  - adapted state tokens/masks
  - option view tokens/masks
  - raw question tokens/masks
  - gold indices
  - row/order identity.

Phase 2 — forked private correction:
- instantiate two corrections from **bit-identical A/B/W initialization**;
- reference private input = question-conditioned native relation signature;
- treatment private input = query-free state↔option identity;
- both consume the exact same cached native evidence;
- native runtime is frozen and not present in either correction optimizer;
- both correction branches use identical TRAIN order, optimizer, LR, weight decay, grad clip, CE/JS coefficients, epochs and selector.

This turns native equality from a post-hoc assertion into a structural fact.

## S50 primary scientific question

> Holding native evidence bit-identical by construction, does query-free state↔option identity improve cross-view option identity/stability without sacrificing useful correctness versus question-conditioned native relation signatures?

## Required A0 before DEV

- shared-native cache deterministic roundtrip;
- cache content digest stable;
- reference/treatment receive byte-identical native logits, query tokens and row identities;
- only private signature source differs;
- correction initialization bit-identical;
- native parameters absent from private optimizers;
- no gradient path from private loss into cached/native tensors;
- query-free treatment identity receives no question tensor;
- K=3/7/255;
- option permutation;
- probability mass;
- full-K;
- one native encode authority per source row/view before caching;
- replay produces identical metrics independent of branch evaluation order.

## Freshness

S50 must use wholly fresh A0/TRAIN/DEV rows.
No S49 exposed row may be reused.

## Stop rule

One S50 DEV only.

No:
- private branch tuning after DEV
- temperature/mixing sweep
- learned identity projector
- native retraining per branch
- cache regeneration after DEV
- selector change
- seed/LR/epoch retry
- gate weakening
- second S50 DEV.

If S50 still shows no stable gain, the query-free identity family can finally be rejected on a valid matched-native court.
