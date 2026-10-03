# HIRA V1 S43 contract — Cross-View Relative-Gap Signature Geometry Anchoring

Status: **PREREGISTERED / NO S43-A0 EXPOSURE**

Issue: #269

Parent:
- S42 merged main `3e455f39de2b43d69d3e72c8b4b9e6098fd92fa9`
- S42 fresh DEV run `37107502716`
- S42 outcome `HIRA_V1_S42_MATCHED_CROSS_VIEW_RELATIONAL_GEOMETRY_DEV_COMPLETE`
- frozen interpretation **Case B**

## Scientific question

> Can S38/S41-style full-bilinear correctness co-adaptation retain its gain when actual AdamW runtime movement is constrained against the full set of cross-view same-option-vs-wrong-option similarity gaps?

S42 proved that full KxK absolute similarity MSE is too indirect:
- correctness remained positive;
- signature discrimination margin still fell materially;
- agreement/JS also regressed.

S43 changes only the anchor target.

## Shared model shell

Keep exactly:
- S35 native 256D relation/signature geometry;
- S17 primary/fusion shell;
- A13 final-attention LoRA 16,384;
- shared primary projection 32,768;
- original A13 frozen;
- HIRACore frozen;
- S38 full bilinear W 256x256 / 65,536;
- treatment total trainable 114,688;
- matched reference trainable 49,152;
- masked-mean + L2-normalized query summary, eps 1e-12;
- S14 equal standardized full-K fusion;
- S15 relation-logit detach in fused-primary objective;
- S17 primary/relation norm-balanced runtime gradient;
- state-once / full-K / opaque option IDs;
- no projected relation path.

## Relative-gap geometry

Let normalized canonical/paraphrase option signatures be:
- `S_c [B,K,256]`
- `S_p [B,K,256]`

Construct:
`G[b,i,j] = dot(normalize(S_c[b,i]), normalize(S_p[b,j]))`

For every `j != i` define:
`R[b,i,j] = G[b,i,i] - G[b,i,j]`

Diagonal entries are excluded from the anchor mean because they are identically zero.

Matched reference:
`R_r = stopgrad(relative_gap(G_reference))`

Treatment:
`R_t = relative_gap(G_treatment)`

Anchor:
`A_gap = mean_{b,i,j != i}((R_t - R_r)^2)`

Properties:
- preserves each logical option's same-option advantage over every wrong option;
- no gold answer labels;
- no strongest-wrong max;
- no numeric margin target;
- no diagonal/off-diagonal weighting;
- no learned anchor parameters;
- no coefficient/slack;
- shared logical-option permutation leaves scalar anchor invariant.

## Optimizer-step semantics

Keep exact S41/S42 qualified engine:
- AdamW lr 2e-4;
- betas (0.9, 0.999);
- eps 1e-8;
- weight decay .01;
- treatment runtime+W global clip 1.0;
- foreach false;
- fused false;
- amsgrad false;
- maximize false;
- capturable false;
- differentiable false.

Derive exact candidate parameter deltas and AdamW state using standard PyTorch AdamW on shadow trainable tensors.

Let:
`a = grad(A_gap)`
on treatment runtime params only.

For candidate runtime movement `delta_r`:
- if `a dot delta_r <= 0`: identity;
- if `a dot delta_r > 0`: remove exactly the conflicting component using eps 1e-12.

W candidate delta remains unchanged.

Apply float-representable parameter targets.
Persist AdamW moments/step state produced from the original clipped correctness gradient.

## S43-A0 authority

A0 is diagnostic only and MUST use wholly fresh S43-A0 cases.

Required:
- exact zero-init reference/treatment relation/signature/fused identity;
- selected-choice identity 1.0;
- surfaces reference 49,152 / treatment 114,688 / W 65,536;
- relative-gap tensor shape exact;
- off-diagonal cardinality B*K*(K-1);
- identical treatment/reference gaps -> anchor exactly 0;
- diagonal same-option perturbation -> anchor > 0;
- one wrong-option off-diagonal perturbation -> anchor > 0;
- a constructed pair with comparable full-matrix MSE but different gap distortion is distinguishable by A_gap;
- shared option permutation scalar error <= 1e-7;
- anchor -> reference runtime gradient exactly 0;
- anchor -> W gradient exactly 0;
- anchor -> treatment runtime and LoRA finite/nonzero under synthetic gap drift;
- joint correctness -> W/off-diagonal-W/LoRA finite/nonzero;
- exact AdamW candidate parameter/exp_avg/exp_avg_sq/step parity;
- conflicting actual-step pre-dot > 0 and projected post-dot approx 0;
- safe actual step exact identity;
- W candidate delta unchanged;
- float-applied runtime/W movement within explicit rounding bounds;
- actual applied gap-anchor dot within rounding bound;
- arbitrary K=3/K=7;
- option permutation equivariance;
- question permutation / padding invariance;
- native projection independence;
- checkpoint/probability/full-K/state-once mechanics.

A0 semantic relation/fused accuracy is diagnostic and MUST NOT tune S43.

## Intended fresh matched TRAIN/DEV

Only after:
1. contract frozen;
2. interpretation plan frozen;
3. S43-A0 qualified;
4. A0 receipt frozen;
5. wholly fresh S43 authority frozen;
6. matched trainer/workflow frozen;
7. exact staged-head generic CI PASS;
8. separate one-shot marker.

Intended:
- seed **64001**
- TRAIN 768
- DEV 192
- 12 wholly fresh S43 domains
- K=4
- two state views
- two question wording views
- two option views
- 24 epochs
- batch 16
- identical semantic rows/order
- independent runtime/optimizer state.

Freshness:
- no exact S0-S42 exposed rows;
- no S43-A0 rows;
- no M5 final/confirmation rows;
- no W29-W34 sealed rows.

## Stop rule

After one S43 DEV:
- no strongest-wrong variant;
- no hinge/numeric target margin;
- no gap weighting;
- no mixing with S41/S42 anchors;
- no anchor coefficient/slack;
- no alternate norm;
- no optimizer reinterpretation;
- no partial detach / gradient mixing;
- no W-only LR/scheduler;
- no rank/factorization;
- no residual scale/bias/nonlinearity;
- no seed/LR/epoch/batch retry;
- no gate weakening;
- no second DEV.

Scientific FAIL is valid.
No external Laya/Jev benchmark before a separately confirmed DEV_READY candidate.
