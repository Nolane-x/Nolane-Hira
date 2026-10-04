# HIRA V1 S50 contract — Shared-Native Forked Private Readouts

Status: **PREREGISTERED / NO S50-A0 EXPOSURE**

Issue: #283

Parent:
- S49 closed as **INVALID MATCHED COURT / INCONCLUSIVE**
- initial run `37178974852`
- treatment continuation `37181138393`
- native trajectory identity failed at all 24 epochs.

## Scientific question

> Holding native evidence bit-identical by construction, does query-free state↔option identity improve cross-view option identity/stability without sacrificing useful correctness versus question-conditioned native relation signatures?

## Structural rule

S50 MUST NOT execute independent native training inside reference and treatment branches.

There is exactly:
1. one shared native training authority;
2. one selected/frozen native checkpoint;
3. one immutable native evidence cache;
4. two private correction branches trained only against that cache.

## Shared native phase

Use the frozen S45/S49 native mechanics once:
- M4 base bundle
- existing native LoRA/projection trainable surface
- existing native losses
- existing optimizer/LR/weight decay/grad clip.

No private correction module participates in this phase.

Native authority is **fixed epoch 24**.
S50 DEV MUST NOT be encoded, scored, or used for checkpoint selection during this phase.

After epoch 24, native runtime is frozen.

## Shared evidence cache

For each query row, cache:
- case/row identity
- gold index
- raw/triadic logits
- native relation logits
- native relation signature
- adapted state tokens/mask
- option-view tokens/token mask/view mask
- raw question tokens/mask.

Cache tensors:
- are detached;
- are cloned;
- are contiguous;
- require no gradient;
- carry a stable SHA-256 digest.

After cache creation, source tensor mutation must not change cached bytes.

Reference and treatment consume the exact same cache object/bytes.

## Private forks

Reference:
- signature source = cached question-conditioned native relation signature;
- raw query remains live in the correction readout.

Treatment:
- signature source = query-free state↔option identity computed from cached state/option evidence;
- the identity API accepts no question tensor;
- raw query remains live only in the correction readout.

Both:
- correction parameters **114,688**
- A/B/W shapes exact S44/S45
- bit-identical initialization
- native trainable params **0**
- identity added trainable params **0**
- identical private optimizer
- identical LR / weight decay / grad clip
- identical CE + JS coefficients
- identical TRAIN order
- identical private checkpoint selector
- no second encoder.

## Required S50-A0

Cache:
- deterministic digest;
- roundtrip exact;
- source mutation isolation;
- cache tensors require_grad false;
- branch-order replay exact;
- row/gold/triadic/native-relation/query bytes shared exactly;
- final fused replay is reconstructable from cached triadic + private relation logits only.

Ownership:
- native parameters absent from private optimizer by construction;
- no live native graph in private phase;
- reference/treatment correction initialization bit-identical;
- reference/treatment correction count both 114,688;
- query-free identity params 0.

Treatment mechanics:
- question inputs absent from identity API;
- K=3/7/255;
- option/state/token/view permutation properties;
- full-K corrected logits;
- probability mass <=1e-6;
- raw query remains live in correction readout.

A0 semantic values are diagnostic only.

## Fresh S50 authority

Only after qualified A0 and exact-head CI.

Intended:
- seed **71001**
- TRAIN 768
- DEV 192
- 12 wholly fresh S50 domains
- K=4
- private epochs 24
- batch 16
- one DEV only.

No exact S49 A0/TRAIN/DEV row may be reused.

## Prohibited

No:
- branch-specific native retraining
- cache regeneration after DEV
- identity operator variant
- pair-temperature/direct-relative sweep
- query leak
- raw/native identity blend
- learned projector
- correction/loss/capacity change
- selector change
- retry
- gate weakening
- second S50 DEV.

Scientific failure is valid.
