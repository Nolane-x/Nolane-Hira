# HIRA V1 S40 handoff — to S41 Optimizer-Step-Anchored Joint Co-Adaptation

S40 closes **before TRAIN/DEV exposure**.

## Why

S40 correctly projects the raw treatment runtime gradient against the matched-reference signature-anchor gradient.

But the real optimizer is AdamW.

AdamW transforms raw gradients through:
- first moments;
- second moments;
- bias correction;
- coordinate-wise preconditioning;
- decoupled weight decay.

Therefore raw-gradient orthogonality to the anchor gradient does not guarantee the **actual AdamW parameter delta** is non-anchor-increasing.

No S40 DEV was opened.

## Canonical S40 evidence

Qualified A0:
- run `37095924764`
- artifact `11264530873`
- digest `sha256:38a3179453383491f8640c9e44dd4d2be33bf16730b81725f072e016cf67c3b8`

Staged matched court:
- head `6aedd1bdb8644bcf9f366f1c120dff973eff8330`
- CI `37096245409` PASS

No `HIRA-V1-S40-ENABLE-TRAIN-DEV` exists.

## S41 scientific question

> Can S38-style joint correctness co-adaptation retain its gain when the actual stateful AdamW runtime parameter movement is prevented from increasing native-signature drift to first order?

## S41 mechanism

Reference:
- exact matched S35/S17 native runtime;
- same initialization/rows/order;
- independently optimized;
- detached reference signatures.

Treatment:
- exact S38 full bilinear correctness family;
- W 256x256 / 65,536 params;
- joint correctness co-adaptation.

Anchor:
`A_sig = mean(1 - cosine(s_treatment, stopgrad(s_reference)))`

Optimizer-faithful step:
1. compute exact frozen treatment raw gradients;
2. apply the frozen grad-clip rule;
3. advance AdamW moment/state equations to obtain the exact candidate parameter delta;
4. split runtime delta and W delta;
5. compute anchor gradient `a` on treatment runtime params;
6. if `a dot delta_runtime > 0`, remove only that conflicting actual-step component;
7. apply projected runtime delta;
8. apply W candidate delta unchanged;
9. persist the AdamW state derived from the frozen raw clipped gradients.

No anchor coefficient.
No projection slack.
No partial detach.
No W-only LR.
No additional learned capacity.

## Required S41-A0

Must prove:
- exact zero-init identity;
- exact 65,536 W params / 49,152 reference / 114,688 treatment;
- reference detach;
- anchor -> reference/W gradients zero;
- joint correctness -> W and LoRA live;
- exact candidate AdamW delta matches standard AdamW on an unprojected control probe;
- optimizer moment/state transition matches standard AdamW;
- conflicting actual AdamW runtime delta has `a dot delta > 0`;
- projected actual runtime delta has post-dot approximately zero;
- non-conflicting actual delta is exact identity;
- W actual delta unchanged bitwise;
- applied runtime parameter movement equals projected candidate delta;
- sufficiently small optimizer-faithful projected step does not increase anchor;
- option/question/padding invariance;
- arbitrary K;
- projection independence;
- checkpoint/probability/full-K/state-once mechanics.

Fresh S41 diagnostic authority only.
No S40 A0 semantic score may tune S41.
No external Laya/Jev benchmark before a separately confirmed DEV_READY candidate.
