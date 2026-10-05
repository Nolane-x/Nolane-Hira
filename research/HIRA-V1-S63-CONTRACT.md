# HIRA V1 S63 contract — Learned Permutation-Invariant Set Reliability Gate

Status: **FROZEN / PRE-A0**

Issue: #311

## Scientific variable

Reference:
- exact S62 reliability-supervised gate;
- exact four S61 scalar features;
- linear gate 4 -> 1;
- 5 trainable parameters;
- exact S62 TRAIN-only reliability BCE.

Treatment:
- learned permutation-invariant set representation over the full detached fused/pairwise surfaces;
- 61 trainable parameters;
- exact same S62 TRAIN-only reliability BCE target.

Treatment adds exactly **56 trainable parameters**.

No other decision/training mechanism changes.

## Treatment representation

For each option:
- standardized fused score `zf`;
- standardized pairwise score `zp`;
- difference `zf-zp`;
- interaction `zf*zp`.

Per-option dimension: **4**.

Shared option encoder:
- `W_phi [8,4]`
- `b_phi [8]`
- `tanh`
- hidden width **8**.

Permutation-invariant pooling:
- mean = 8
- max = 8
- pooled dimension = **16**.

Concatenate exact S61 scalar features = +4.

Final representation dimension = **20**.

Output:
- `w_out [20]`
- scalar `b_out`.

Total treatment parameters:
- 32 + 8 + 20 + 1 = **61**.

## Initialization

- representation seed: **63063**
- `w_out=0` exactly
- `b_out=-0.916290731874155`
- initial alpha exactly **0.10**
- alpha max **0.35**.

The first reliability step updates output parameters; after a nonzero output update, gradients may reach the learned set encoder.

## Reliability authority

Exact S62 target:
- fixed probe alpha **0.35**
- tolerance **1e-8**
- target 1 iff probe paired CE is non-worse AND paired JS strictly improves
- otherwise target 0
- TRAIN only
- detached
- no DEV target dependency.

## Ownership

Gate surfaces are detached.

No reliability-gate gradient may enter:
- correction
- pairwise head
- native runtime
- immutable cache.

## Mechanics

Required:
- option permutation invariance
- fused offset invariance
- fused positive-scale invariance
- pairwise offset invariance
- pairwise positive-scale invariance
- flat-logit finiteness
- K=3/7/255
- bounded residual
- alpha override 0 exact fused identity
- probability mass error <=1e-6
- deterministic checkpoint replay
- one encoder/state-once.

## Scientific discipline

Fresh intended S63:
- seed 84001
- TRAIN 768
- DEV 192
- 12 wholly fresh domains
- exact S62 state/question/option overlap 0
- K=4
- epochs 24
- one DEV only.

No fresh S63 TRAIN/DEV may be exposed before A0 is qualified and a separate exact-head pre-DEV CI gate passes.
