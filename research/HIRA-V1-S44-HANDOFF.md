# HIRA V1 S44 handoff — to S45 Cross-View Consistent Private Correction

S44 is frozen as:

`HIRA_V1_S44_MATCHED_PRIVATE_CORRECTION_REPRESENTATION_DEV_COMPLETE`

Interpretation:
**Case A — private representation ownership recovers correctness while native transport remains exact.**

## What S44 solved

S38-S43 repeatedly exposed a conflict between correctness adaptation and native transport geometry.

S44 removes that conflict structurally:
- the native runtime follows the exact matched reference trajectory;
- correction receives detached native state;
- private GELU representation + full W learns correctness;
- no correction gradient enters native LoRA/projection/signatures;
- one encoder pass is retained.

Fresh S44 treatment-reference:
- relation canonical **+0.1770833**
- relation paraphrase **+0.1223958**
- fused canonical **+0.0546875**
- fused paraphrase **+0.0078125**
- native representation deltas **exactly zero**

## What S44 did not solve

The new failure surface is private correction consistency, not native ownership.

Treatment-reference:
- corrected relation agreement **-0.1458333**
- fused JS **+0.0310173** worse

So the correction branch can identify the right option more often, but its decisions move too differently across semantically equivalent question views. The fixed downstream fusion converts only a fraction of the relation-level gain into stable fused gain.

## S45 scientific question

> Can Hira keep S44's exact native/private ownership split and relation-correctness capacity while making the private correction decision itself cross-view consistent, without adding parameters or another encoder pass?

## Proposed controlled variable

Keep the complete S44 architecture unchanged:
- native runtime exact reference trajectory
- detached native signature/query
- private GELU A/B: 49,152 params
- full bilinear W: 65,536 params
- correction-only: 114,688 params
- treatment total: 163,840 params
- one encoder pass
- no learned fusion gate
- no width/capacity change

Change only the private correction objective.

S44 correction objective:
`0.10 * CE_corr`

S45 candidate:
`0.10 * CE_corr + 0.25 * JS_corr_cross_view`

where:
- `CE_corr` remains the exact S39/S44 canonical+paraphrase correction CE;
- `JS_corr_cross_view` is symmetric Jensen-Shannon divergence between corrected relation distributions for the canonical and paraphrase views of the same semantic query;
- coefficient **0.25** reuses the already-frozen cross-view JS coefficient from the existing Hira decision objective rather than selecting a new scalar from S44 DEV.

No other loss term changes.

## Why this is the next family

This is not an S44 retry:
- S44 is permanently closed;
- S45 uses wholly fresh authority;
- the new scientific variable is cross-view consistency of the detached private correction expert.

It tests whether the direct relation gains already demonstrated by S44 can become wording-stable before considering any new fusion/routing family.

## Required S45-A0

Before any S45 DEV, prove:
- S44 native reference/treatment trajectory identity remains exact;
- correction JS gradients reach A/B/W;
- correction JS gradients to native LoRA/projection are exactly zero;
- permutation equivariance;
- canonical/paraphrase pair indexing exact;
- identical semantic pair receives zero JS when logits match;
- non-identical deterministic probe receives finite positive JS;
- no second encoder path;
- arbitrary K=3/K=7;
- checkpoint roundtrip;
- full-K / probability mass / state-once;
- no extra trainable parameters;
- no learned fusion gate.

## Fresh-authority rule

S45 must use:
- wholly fresh A0 diagnostic rows;
- wholly fresh TRAIN/DEV rows;
- no exact S0-S44 exposed rows;
- no S45-A0 rows in matched court;
- one DEV only.

## Frozen interpretation before DEV

A — S44 correctness is retained while relation agreement / fused stability materially recover:
cross-view consistency is the missing private-branch constraint.

B — consistency recovers but correctness collapses:
private correctness and consistency conflict; close.

C — correctness remains but consistency does not improve:
loss-level consistency is insufficient; move to a new decision/fusion family.

D — treatment DEV_READY:
freeze immediately and open a separate fresh confirmation before external Laya/Jev evaluation.

No scalar winner score.

## Stop rule

After one S45 DEV:
- no JS coefficient sweep;
- no temperature sweep;
- no alternate divergence;
- no architecture width change;
- no learned fusion gate;
- no native-gradient leakage;
- no second encoder;
- no seed/LR/epoch/batch retry;
- no gate weakening;
- no second DEV.

If S45 fails, change family rather than tune exposed DEV.
