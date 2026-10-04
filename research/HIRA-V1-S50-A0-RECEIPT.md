# HIRA V1 S50 A0 receipt — Shared-Native Forked Private Readouts

Status: **QUALIFIED**

Run: `37183981097`  
Artifact: `11296132080`  
Artifact digest: `sha256:364545c53977aba290faacb4a064b59b104a3d9f67e1ea0ad8a6b0cfe543ab1b`  
Authorization head: `d1f4d5d8e47d3446b610aedf942988735ae38e03`

Outcome:
`HIRA_V1_S50_A0_SHARED_NATIVE_FORKED_PRIVATE_READOUTS_READY`

## Shared-cache authority

- canonical cache digest: `81871db154ac009a53b3426306398fc5e854af50cb9ae04cd52cc186475b9cce`
- paraphrase cache digest: `5f4bfb10a786cbda141d58ca93009219e25a4fadd0833ac6a86fd3b6ef66bb73`
- canonical replay digest exact: **true**
- cache tensors with requires_grad: **0**
- branch-order replay max abs error: **0**
- one native output authority: **true**
- state-view encodes: **32**
- fused replay from cache only: **true**
- shared triadic logits exact: **true**
- shared native relation logits exact: **true**
- shared question tensors exact: **true**

## Private ownership

- native training arms in private phase: **0**
- private optimizer native parameter count: **0**
- reference correction: **114,688**
- treatment correction: **114,688**
- correction initialization exact: **true**
- query-free identity params: **0**
- query inputs absent from identity API: **true**
- second encoder in private phase: **false**

## Liveness / validity

- raw-query treatment sensitivity max abs: **0.00396538**
- probability-mass max error: **1.192e-7**
- identity cross-state-view same-option cosine: **0.828975**
- identity same-vs-strongest-wrong margin: **0.015966**

Identity semantic diagnostics are A0-only and are not model-selection evidence.

## Qualification

S50 mechanically qualifies.

Unlike S49, reference and treatment no longer need separately trained native trajectories. The S50 private comparison can be built over a single immutable native evidence cache, so matched-native equality is structural rather than post-hoc.

Fresh S50 TRAIN/DEV remains separately gated.

Shared native production phase is preregistered as:
- TRAIN-only
- fixed epoch 24
- zero S50 DEV exposure before private comparison.
