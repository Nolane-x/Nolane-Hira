# HIRA V1 S41 handoff — to S42 Cross-View Relational Signature-Geometry Anchoring

S41 closed as:

`HIRA_V1_S41_MATCHED_OPTIMIZER_STEP_ANCHORED_DEV_COMPLETE`

Frozen interpretation:
**Case B — correctness retained but transport is not fully protected.**

## Residual evidence

S41 solved much of the S38 tradeoff:
- fused canonical +12.24pp vs reference
- fused paraphrase +8.33pp
- relation canonical +24.22pp
- relation paraphrase +18.23pp
- fused agreement +1.56pp
- relation agreement +12.76pp

But:
- signature discrimination margin **-12.01pp**
- treatment absolute signature margin **0.08083**
- frozen signature-margin gate requires **>=0.15**

The per-signature cosine-to-reference anchor therefore preserves local correspondence but not enough **relative option separation**.

## S42 scientific question

> Can S38/S41 joint correctness co-adaptation retain its gain when the actual AdamW runtime movement is constrained against the full cross-view relative option geometry rather than independent per-signature cosine drift?

## S42 frozen candidate mechanism

Keep exactly:
- S35 native 256D signatures
- S17 primary/fusion shell
- S38 full bilinear W 256x256 / 65,536 params
- treatment total 114,688 params
- matched native reference total 49,152 params
- S41 exact stateful AdamW candidate engine
- S41 actual-step projection
- S41 float-quantized movement guards
- no learned anchor params
- no anchor coefficient
- no W projection

Change only the anchor target.

For each matched batch:
- treatment canonical signature tensor: `S_tc [B,K,256]`
- treatment paraphrase signature tensor: `S_tp [B,K,256]`
- detached reference tensors: `S_rc`, `S_rp`

L2-normalize each option signature.

Construct full cross-view option-similarity matrices:
- `G_t = normalize(S_tc) @ normalize(S_tp)^T`
- `G_r = normalize(S_rc) @ normalize(S_rp)^T`

Shape:
`[B,K,K]`.

Anchor:
`A_rel = mean((G_t - stopgrad(G_r))^2)`

This includes:
- diagonal same-option correspondence;
- all off-diagonal wrong-option similarities;
- therefore the relative diagonal-vs-strongest-wrong geometry underlying signature discrimination.

No learned params.
No coefficient.
No margin hyperparameter.
No option labels/gold labels enter the anchor.
Permutation of option order conjugates the KxK matrix and leaves the scalar anchor invariant.

## Actual-step semantics

Let:
`a = grad(A_rel)`
on treatment runtime params.

Use exact S41 candidate AdamW delta.

For runtime candidate delta `delta_r`:
- if `a dot delta_r <= 0`: identity;
- otherwise remove only the conflicting component.

W candidate delta remains unchanged.

Persist AdamW moments/state from original clipped raw gradient exactly as S41.

## Required S42-A0

Fresh S42 diagnostic authority must prove:
- exact zero-init reference/treatment relation/signature/fused identity;
- W 65,536 / reference 49,152 / treatment 114,688;
- reference detach;
- relational anchor -> reference runtime gradient 0;
- relational anchor -> W gradient 0;
- relational anchor -> treatment LoRA finite/nonzero under synthetic drift;
- joint correctness -> W/off-diagonal-W/LoRA finite/nonzero;
- full KxK anchor responds to an off-diagonal-only signature perturbation;
- full KxK anchor responds to a diagonal correspondence perturbation;
- logical-option permutation leaves anchor scalar invariant;
- actual AdamW candidate/state parity exact as S41;
- conflicting actual-step projection post-dot approximately zero;
- non-conflict identity;
- W delta unchanged;
- actual float-applied movement within rounding bounds;
- actual applied anchor dot within rounding bound;
- arbitrary K=3/K=7;
- question/padding invariance;
- native projection independence;
- checkpoint/probability/full-K/state-once mechanics.

A0 semantic values diagnostic only.

## Fresh authority

S42 must use:
- wholly fresh S42-A0 rows;
- wholly fresh matched TRAIN/DEV rows;
- no S41 DEV rows;
- no post-S41 tuning on exposed data;
- one DEV only.

No Laya/Jev benchmark before a separately confirmed DEV_READY candidate.
