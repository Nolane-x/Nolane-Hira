# HIRA V1 S64 contract — Parameter-Matched Context-Injected Reliability Gate

Status: **FROZEN / PRE-A0**

Issue: #315

Parent:
- S63 merged main `4fcb58f0c35e2feca841b08f6f560c092ae600fd`
- fresh run `37314015283`
- artifact `11347706059`
- digest `sha256:dcb69358dfa97f3644198534faa1c2f5f214b4f1773d79008eec13cafaf2a97e`
- verdict **Case B**.

## Scientific question

Does detached state/query/option context contain reliability information that decision-surface geometry does not, under a parameter-matched gate and the exact S62 correctness-veto target?

## Controlled variable

Both arms:
- 60 trainable params
- tensors `W_phi,b_phi,w_out,b_out`
- exact same initialization
- exact same TRAIN-only reliability BCE target
- exact same optimizer/LR/weight decay
- exact same correction/head trajectory
- exact same alpha max 0.35 and initial alpha 0.10
- exact same frozen S17 selector.

Reference option input:
`[surface4, zero4]`.

Treatment option input:
`[surface4, contextual4]`.

No parameter advantage.

## Surface channels

Exact S63 normalized option features:
`[zf, zp, zf-zp, zf*zp]`.

## Context channels

Source:
- exact detached S59 representation `[B,K,512]`
- query-free option identity + joint state/query/option context
- no new encoder call.

Fixed projection:
- `P_ctx [4,512]`
- seed `64064`
- row-normalized
- registered buffer
- zero trainable parameters.

After projection, each channel is centered and RMS-normalized across options.

## Learned gate

Option input dim: 8.

Shared option encoder:
- `W_phi [5,8]`
- `b_phi [5]`
- tanh
- hidden width 5.

Pooling:
- mean 5
- max 5
- pooled dim 10.

Append exact four S61 scalar features.

Final reliability representation dim: 14.

Output:
- `w_out [14]`
- scalar `b_out`.

Total trainable parameters:
- 40 + 5 + 14 + 1 = **60**.

Initialization:
- option encoder seed `64164`
- `w_out=0`
- `b_out=-0.916290731874155`
- initial alpha = **0.10**.

## Reliability authority

Exact S62 target:
- alpha probe 0.35
- tolerance 1e-8
- positive iff probe paired CE is non-worse AND paired JS strictly improves.

No teacher, no DEV target, no self-anchor.

## Ownership

All fused/pairwise/context inputs are detached.

Reliability gradient may update only:
- W_phi
- b_phi
- w_out
- b_out.

It may not update:
- native runtime
- cache
- correction
- pairwise head.

## Required A0

Must prove:
- both arms exactly 60 trainable params
- added treatment params 0
- parameter initialization bit-identical
- fixed projection identical and non-trainable
- reference contextual channels exact zero
- treatment contextual channels finite/non-degenerate
- option permutation invariance
- treatment context sensitivity under a mechanical nonzero-output probe with fixed fused/pairwise logits
- positive-scale/offset invariance of decision-surface component
- flat finite mechanics
- initial alpha exact 0.10
- output gradient live at step 1
- staged phi gradient becomes live after output warm step
- upstream gradient zero
- alpha override 0 exact fused identity
- bounded residual
- K=3/7/255
- probability mass error <=1e-6
- exact S62 target semantics
- deterministic checkpoint replay
- one encoder/state-once
- fresh S64 TRAIN/DEV exposed = false.

## Fresh court

Only after A0 and exact pre-DEV CI:
- seed 85001
- TRAIN 768
- DEV 192
- 12 fresh S64 domains
- exact S63 overlap 0
- K=4
- 24 epochs
- one DEV.

## Stop rule

No projection seed/dim sweep, hidden-width sweep, feature retrofit, pooling change, init sweep, target change, BCE weighting, optimizer change, regularizer, gradient coupling, native retraining, selector change, retry, second S64 DEV or external Laya/Jev evaluation after scientific exposure.
