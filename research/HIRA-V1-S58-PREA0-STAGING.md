# HIRA V1 S58 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #299  
PR: #300

## Parent

S57 merged main:
`27521eb426371b96a10c9cfcadf1fbf69ead9f50`

S57 scientific verdict:
**Case B**

Fresh S57 evidence:
- run `37271509208`
- artifact `11327849211`
- artifact digest `sha256:868faf45b1994286e52cbac4c35adfbbb420e2ba8a6c561545067da013c05147`
- reference selected epoch **19**
- reference checkpoint SHA:
  `804dc0b31ca75a77400c4a658ae893058d39916ed287d9fe2d45643798b739aa`

## Frozen S58 teacher

Teacher:
- S57 reference branch only
- seed **78001**
- selected epoch **19**
- correction params **114,688**
- ordinal coefficient **0**
- checkpoint file `reference-private-candidate.pt`
- checkpoint SHA pinned above
- non-trainable in S58
- no S58-based teacher selection.

## S58 consensus mechanics

Teacher pair is active only if:
- canonical teacher margin abs >= **0.25**
- paraphrase teacher margin abs >= **0.25**
- both teacher signs agree
- gold-involving pair ranks gold correctly.

Student target:
- preserve detached teacher consensus sign
- target margin floor **0.05**

Reference coefficient:
- **0.0**

Treatment coefficient:
- **0.05**

Added trainable params:
- **0**

No full-distribution JS.
No student-self anchor.

## A0 staged

Core:
`src/nmd/v1_teacher_consensus_ranking.py`

A0:
`scripts/hira_v1_s58_a0_consensus_teacher_pairwise_ranking.py`

Workflow:
`.github/workflows/hira-v1-s58-a0-consensus-teacher-pairwise-ranking.yml`

Marker:
`research/HIRA-V1-S58-ENABLE-A0`

A0 proves:
- exact teacher artifact/checkpoint authority
- teacher frozen/no gradients
- deterministic teacher replay
- strong consensus active
- weak teacher inactive
- teacher sign disagreement inactive
- wrong gold consensus filtered
- non-gold consensus remains eligible
- detached consensus sign
- satisfied student target lower loss than violation
- offset/positive-scale teacher invariance
- flat-teacher anti-collapse
- flat-student violation live
- option permutation
- student treatment gradient live
- reference auxiliary exact zero
- native/cache isolation
- K=3/7/255
- full-K probability mass
- one encoder/state-once.

## Authorization rule

The A0 marker MUST remain absent until the exact final staging head passes generic CI on Python 3.10 and 3.12.

A0 is mechanical only:
- no fresh S58 TRAIN/DEV
- no S58 model selection
- no external Laya/Jev evaluation.
