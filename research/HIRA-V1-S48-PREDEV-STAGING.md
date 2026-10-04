# HIRA V1 S48 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #279  
PR: #280

## Parent

S47 merged main:
`b48ce16faa721a79e998db0c7d585544e4281c9a`

S47 fresh court:
- run `37168305687`
- artifact `11291161358`
- interpretation **Case C**
- ordinal DEV_READY false

## Qualified S48-A0

Run: `37171579714`  
Artifact: `11291393129`  
Digest: `sha256:de5ae3d7c936539c19d4b1a9a695fbd27a44fbeac7d1f5d46f26d5ab959090e7`  
Authorization head: `eeb9e328d28125723c3e92941ef48eb1d532ea12`

Outcome:
`HIRA_V1_S48_A0_QUERY_QUOTIENT_OPTION_EVIDENCE_READY`

A0 proves:
- reference/treatment correction capacity both 114,688
- treatment total 163,840
- zero quotient trainable parameters
- K=3/7/255
- option permutation equivariance
- orthogonal query nuisance removal
- no raw-query bypass
- relevant relation direction remains live
- distinct relation direction separation
- deterministic zero-subspace behavior
- one encoder batch/state-once
- exact native/private ownership
- W -> B -> A warm-start.

## Frozen S48 matched variable

Reference:
**raw normalized mean query private correction**

Treatment:
**option-difference covariance direction quotient private correction**

Both:
- native trainable **49,152**
- correction **114,688**
- total **163,840**
- identical correction initialization
- exact same native initialization/update rule
- exact same LR/weight decay/grad clip
- exact same `0.10 CE + 0.25 JS`
- exact same S35 checkpoint-selection rule per arm
- 24 epochs
- batch 16
- no second encoder.

## Fresh authority

Frozen:
- seed **69001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S48 domains
- K=4
- identical rows/order across arms
- one DEV authority only.

Freshness guards reject:
- prior exposed surfaces inherited through S45
- S47 TRAIN/DEV exact rows
- S48-A0 exact rows.

## Matched evidence

Primary comparison:
`treatment_query_quotient - reference_raw_query`

Report:
- fused canonical/paraphrase correctness
- paired both-correct
- question-swap
- fused agreement / JS / margins
- corrected relation correctness
- corrected relation agreement / JS / margins
- native trajectory fingerprints
- treatment quotient norm / zero fraction
- raw-vs-quotient cosine
- quotient cross-view cosine
- corrected option-ranking cross-view agreement.

## Staged files

- `research/HIRA-V1-S48-CONTRACT.md`
- `research/HIRA-V1-S48-INTERPRETATION-PLAN.md`
- `research/HIRA-V1-S48-A0-RECEIPT.json`
- `research/HIRA-V1-S48-A0-RECEIPT.md`
- `src/nmd/v1_query_quotient_private_correction.py`
- `src/nmd/v1_s48_authority.py`
- `scripts/hira_v1_s48_train_dev.py`
- `tests/test_v1_s48_authority.py`
- `tests/test_v1_s48_matched_harness.py`
- `.github/workflows/hira-v1-s48-matched-query-quotient-option-evidence-train-dev.yml`

## Authorization rule

Matched workflow is marker-gated on:

`research/HIRA-V1-S48-ENABLE-TRAIN-DEV`

The marker MUST NOT exist until the exact final staging head passes generic CI on Python 3.10 and 3.12.

After authorization:
- exactly one fresh S48 matched TRAIN/DEV court;
- no quotient blend/sweep;
- no raw-query bypass;
- no learned projector;
- no checkpoint-rule change;
- no loss/capacity/optimizer tuning;
- no second S48 DEV;
- no external Laya/Jev evaluation unless separately confirmed DEV_READY.
