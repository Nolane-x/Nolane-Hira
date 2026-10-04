# HIRA V1 S50 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #283  
PR: #284

## Parent scientific status

S49 is merged and closed as:
**INVALID MATCHED COURT / INCONCLUSIVE**

Main after S49:
`4273fa0848093385374c0157ea7f3a1ce667b6e3`

S50 exists specifically to eliminate the S49 native-trajectory confound by construction.

## Qualified S50-A0

Run: `37183981097`  
Artifact: `11296132080`  
Digest: `sha256:364545c53977aba290faacb4a064b59b104a3d9f67e1ea0ad8a6b0cfe543ab1b`  
Authorization head: `d1f4d5d8e47d3446b610aedf942988735ae38e03`

Outcome:
`HIRA_V1_S50_A0_SHARED_NATIVE_FORKED_PRIVATE_READOUTS_READY`

A0 proves:
- immutable cache replay
- canonical cache digest `81871db154ac009a53b3426306398fc5e854af50cb9ae04cd52cc186475b9cce`
- paraphrase cache digest `5f4bfb10a786cbda141d58ca93009219e25a4fadd0833ac6a86fd3b6ef66bb73`
- cache requires_grad count 0
- branch-order replay error 0
- fused replay from cached triadic evidence only
- shared triadic/native/query evidence exact
- private optimizer native params 0
- correction initialization bit-identical
- reference/treatment correction params 114,688 each
- query-free identity params 0
- raw query remains live
- no second encoder in private phase.

## Fresh S50 authority

Frozen:
- seed **71001**
- TRAIN **768**
- DEV **192**
- **12** wholly fresh S50 domains
- K=4
- 2 option views
- no exact S49 TRAIN/DEV overlap
- no S50-A0 overlap.

Files:
- `src/nmd/v1_s50_authority.py`
- `tests/test_v1_s50_authority.py`

## Phase 1 — one shared native TRAIN-only authority

Exactly one native runtime is trained.

Mechanics:
- base M4 runtime unchanged
- trainable native surface **49,152**
- exact S35 native relation mechanics
- AdamW
- LR **2e-4**
- weight decay **0.01**
- grad clip **1.0**
- batch **16 semantic cases**
- seed **71001**
- exactly **24 epochs**
- native authority = **terminal epoch 24**

Critical:
- S50 DEV is **not passed to the native training function**
- S50 DEV is not encoded or scored during native epochs
- no native DEV checkpoint selection exists
- no private correction participates in native training.

At epoch 24:
- native checkpoint is frozen
- all native parameters set requires_grad false
- native optimizer is destroyed.

## Phase 2 — immutable shared cache

Only after native epoch 24 is frozen:
- TRAIN cache is materialized in **48 batches × 16 cases**
- DEV cache is materialized in **12 batches × 16 cases**
- each batch performs exactly one native output call producing both canonical and paraphrase evidence.

Cached:
- triadic logits
- native relation logits
- native relation signatures
- state tokens/masks
- option-view tokens/token/view masks
- raw question tokens/masks
- row IDs and gold labels.

Reference and treatment consume the same cache bytes.

Fused output is reconstructed only from:
cached triadic logits + branch-private corrected relation logits.

No encoder/native runtime is called during private training/evaluation.

## Phase 3 — matched private forks

Reference:
question-conditioned native relation signature.

Treatment:
query-free state↔option identity.

Matched:
- correction capacity **114,688**
- bit-identical A/B/W initialization
- native optimizer params **0**
- identity added params **0**
- private epochs **24**
- batch **16**
- LR **2e-4**
- weight decay **0.01**
- grad clip **1.0**
- correction CE coefficient **0.10**
- correction cross-view JS coefficient **0.25**

TRAIN cache order:
- 48 immutable cache batches
- epoch e uses `random.Random(71001 + e)`
- identical batch permutation for reference and treatment.

DEV:
- scored only from frozen DEV cache
- same DEV rows/cache for both branches
- no native encode during DEV scoring.

## Frozen checkpoint selector

Exact S17 lexicographic selector:

1. paired both-correct
2. fused canonical accuracy
3. corrected relation canonical accuracy
4. corrected relation canonical gold margin
5. fused canonical gold margin
6. question-swap choice-change
7. fused cross-view agreement
8. same-option signature cosine
9. signature same-vs-wrong margin
10. negative fused canonical CE
11. negative epoch.

No alternate selector.

## Staged implementation

- `scripts/hira_v1_s50_train_dev.py`
- `tests/test_v1_s50_matched_harness.py`
- `.github/workflows/hira-v1-s50-shared-native-forked-private-train-dev.yml`

Workflow marker:

`research/HIRA-V1-S50-ENABLE-TRAIN-DEV`

The marker MUST remain absent until the exact final staging head passes generic CI on Python 3.10 and 3.12.

## One-shot rule

After authorization:
- exactly one fresh S50 scientific court;
- no cache regeneration after DEV;
- no branch-specific native retraining;
- no identity variant;
- no pair-temperature/direct-relative sweep;
- no query leak/blend/projector;
- no loss/capacity/optimizer change;
- no alternate selector;
- no retry for scientific weakness;
- no gate weakening;
- no second S50 DEV;
- no external Laya/Jev evaluation unless separately confirmed after valid DEV_READY.
