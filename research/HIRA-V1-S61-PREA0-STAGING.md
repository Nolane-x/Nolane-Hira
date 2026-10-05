# HIRA V1 S61 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #307

Parent S60 merged main:
`c795c6c448af942f597a8d33131557338def7fa9`

Parent S60 verdict:
**Case B**

Fresh S60 evidence:
- run `37297784471`
- artifact `11340397535`
- digest `sha256:c55205e9b74aac6196346a0ce9b705900aaec7fe4c6b30380af8ee0093d1b18b`.

## Frozen S61 adaptive gate

Inputs:
- detached fused logits
- detached explicit pairwise aggregate.

Features exactly **4**:
1. normalized fused top1-top2 confidence
2. normalized pairwise top1-top2 confidence
3. fused/pairwise top1 agreement in {+1,-1}
4. centered standardized surface alignment.

Gate:
- `w:[4]`
- `b:scalar`
- total **5 trainable params**
- no hidden layer
- alpha max **0.35**
- w init **0**
- b init **-0.916290731874155**
- every initial query alpha **0.10**.

Treatment composition:
`fused + alpha(query) * fused_rms * tanh(standardized_pairwise)`.

Reference:
exact fused surface.

Forbidden:
- pairwise-only path
- teacher
- pseudo-target
- self-anchor
- upstream gate gradient.

## A0 staged

Core:
`src/nmd/v1_confidence_adaptive_bounded_hybrid.py`

A0:
`scripts/hira_v1_s61_a0_confidence_adaptive_bounded_hybrid.py`

Workflow:
`.github/workflows/hira-v1-s61-a0-confidence-adaptive-bounded-hybrid.yml`

Marker:
`research/HIRA-V1-S61-ENABLE-A0`

A0 proves:
- exact 5-param surface
- initial alpha exactly 0.10
- 4 detached feature mechanics
- affine/option-permutation invariance
- flat-input finiteness
- manual adaptive alpha separation
- w/b gradients live
- upstream gradients zero
- alpha bounds and exact identity override
- K=3/7/255
- full-K mass
- one encoder/state-once
- checkpoint replay exact.

## Authorization rule

The A0 marker MUST remain absent until this exact final staging head passes generic CI on Python 3.10 and 3.12.

A0 is mechanical only:
- no fresh S61 TRAIN/DEV
- no S61 model selection
- no external Laya/Jev evaluation.
