# HIRA V1 S36 A0 receipt — Native Signature Linear Correctness Readout

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #255
PR: #256

## Canonical authority

- run `37025445955`
- artifact `11235297229`
- artifact digest `sha256:33c0961945b0f4cb3c56022107fa0fe726481e6e5f6d5602445e78220da78070`
- authority head `4ca77a3659dfa1b3f2676a6bff41cb4afd78ed0b`
- result integrity `sha256:5dfbf107c534c5f783044844ebcf02f8621683c8526881f1b6ac6d51f5b9cf09`
- outcome `HIRA_V1_S36_A0_NATIVE_SIGNATURE_READOUT_READY`

No replacement S36-A0 is authorized.

## Exact zero-init identity

Treatment added learned params:
- shared readout vector: **256**
- residual scale: **1.0**
- bias/nonlinearity: **none**

Capacity:
- control trainable surface: **49,152**
- treatment trainable surface: **49,408**
- LoRA: **16,384**
- shared primary projection: **32,768**
- frozen A0 runtime trainable: **0**
- original A13 trainable: **0**
- HIRACore trainable: **0**

At zero initialization:
- relation-logit max abs difference: **0**
- native signature max abs difference: **0**
- primary-logit max abs difference: **0**
- fused-logit max abs difference: **0**
- selected-choice identity rate: **1.0**

## Gradient ownership

Treatment:
- relation block -> readout L1: **0.0340911485**
- primary block -> readout L1: **0**
- native relation block -> shared projection direct L1: **0**
- primary block -> shared projection L1: **1318.4007568359**
- native relation block -> A13 LoRA L1: **0.2824096456**

This is the preregistered ownership split.

## Projection independence

After strong shared-projection perturbation:
- native relation-logit max abs change: **0**
- native signature max abs change: **0**
- primary-logit max abs change: **0.0021466089**

The correctness readout consumes native signatures and does not create a hidden projection dependency.

## Equivariance / arbitrary K / mechanics

- logical-option permutation logit error: **1.8626451492e-8**
- logical-option permutation signature error: **0**
- K=3 output shapes: logits `[3,3]`, signatures `[3,3,256]`
- K=7 output shapes: logits `[3,7]`, signatures `[3,7,256]`
- checkpoint key count: **1**
- readout checkpoint roundtrip: PASS
- probability-mass error: **1.1920928955e-7**
- full-K: PASS
- expected state views: **32**

## A0 semantic diagnostics

Diagnostic only:
- native relation accuracy: **29.6875%**
- fused accuracy: **26.5625%**
- relation CE: **1.3826909065**
- canonicalization: **0.3684265018**
- relation block: **0.1935330778**

These values MUST NOT tune:
- readout scale
- bias/nonlinearity
- readout rank/width
- readout-only LR
- native geometry
- projected/native mixing
- optimizer
- selector/gates
- seed/LR/epochs/batch

## Consequence

S36-A0 is **QUALIFIED**.

Fresh matched TRAIN/DEV may open only after:
1. this receipt is frozen;
2. wholly fresh S36 authority is frozen;
3. trainer/workflow are frozen;
4. exact-head generic CI passes;
5. a separate one-shot TRAIN/DEV marker is committed.

No second S36-A0 run is authorized.
