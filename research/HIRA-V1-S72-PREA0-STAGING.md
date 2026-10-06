# HIRA V1 S72 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #331

Parent S71:
- merged main `ef2e8a4b71ba9a46dd9caea07698df742abb6461`
- run `37459447097`
- artifact `11412505173`
- digest `sha256:aa116237d5508131db18f88a23893d9107d05ae67f46ccffe638520fe8098a50`
- verdict **Case C**.

## Frozen controlled variable

Both arms:
- exact S69 representation
- exact S59 pairwise head
- same pairwise/correction trajectory
- exact S71 mean-only composer
- **80 trainable composer params**
- bit-identical initialization
- exact S66 per-view responsibility objective.

Reference residual direction:
- normalized uniform pairwise row mean.

Treatment residual direction:
- fused-prior opponent-profile signed/absolute confidence ratio.

Direction params:
**0 vs 0**.

Treatment parameter advantage:
**0**.

## Staged A0

Core:
`src/nmd/v1_opponent_profile_vector_residual.py`

Unit mechanics:
`tests/test_v1_opponent_profile_vector_residual.py`

A0 court:
`scripts/hira_v1_s72_a0_opponent_profile_vector_residual.py`

Guards:
- `tests/test_v1_s72_a0_harness.py`
- `tests/test_v1_s72_a0_workflow.py`

Workflow:
`.github/workflows/hira-v1-s72-a0-opponent-profile-vector-residual.yml`

A0 is mechanical only and may not expose fresh S72 TRAIN/DEV.

## Authorization rule

Marker:
`research/HIRA-V1-S72-ENABLE-A0`

The marker MUST remain absent until exact final pre-A0 head passes generic CI on Python 3.10 and 3.12.
