# HIRA V1 S60 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #305

Parent S59 merged main:
`867056d999bab707a1d8f4b1ea9ca774ccaf514d`

Parent S59 verdict:
**Case C**

Fresh S59 evidence:
- run `37289522292`
- artifact `11335428630`
- digest `sha256:72a348c27f95c1f7ea2dc0dbcba8db7afac56fa5cd4d284d0593f7321d603e4c`.

## Frozen S60 composer

Base retained:
- correction params **114,688**
- explicit pairwise params **32,832**
- immutable cache
- one encoder/state-once
- no teacher / pseudo-target / self-anchor
- pairwise gradients isolated from correction/native.

Composer:
- fused logits + pairwise aggregate only
- centered RMS epsilon **1e-6**
- pairwise direction `tanh(z_pair)`
- exactly **1 trainable scalar a**
- `alpha = 0.35 * sigmoid(a)`
- alpha max **0.35**
- alpha initial **0.10**
- frozen a initial **-0.916290731874155**
- hybrid `fused + alpha * fused_rms * tanh(z_pair)`
- explicit alpha override 0 returns exact fused logits
- residual per coordinate bounded by alpha * fused RMS.

TRAIN calibration:
- gold CE only
- composer inputs detached
- AdamW
- existing S35 LR
- weight decay **0.0**
- composer gradients cannot enter correction/pairwise/native/cache.

## A0 staged

Core:
`src/nmd/v1_train_calibrated_bounded_hybrid.py`

A0:
`scripts/hira_v1_s60_a0_train_calibrated_bounded_hybrid.py`

Workflow:
`.github/workflows/hira-v1-s60-a0-train-calibrated-bounded-hybrid.yml`

Marker:
`research/HIRA-V1-S60-ENABLE-A0`

A0 proves:
- composer params exactly **1**
- trainable tensor exactly **a**
- alpha initial 0.10
- alpha in (0,0.35)
- exact fused identity at alpha override 0
- adversarial pairwise residual cannot exceed frozen bound
- flat pairwise gives zero residual
- option permutation equivariance
- fused offset / positive-scale equivariance
- pairwise offset / positive-scale invariance
- K=3/7/255
- full-K probability mass
- composer scalar gradient live
- upstream gradients zero
- no pairwise-only final path
- checkpoint roundtrip exact
- one encoder/state-once.

## Authorization rule

The marker `research/HIRA-V1-S60-ENABLE-A0` MUST remain absent until this exact final staging head passes generic CI on Python 3.10 and 3.12.

A0 is mechanical only:
- no fresh S60 TRAIN/DEV exposure
- no S60 model selection
- no external Laya/Jev evaluation.
