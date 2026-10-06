# HIRA V1 S71 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #329

Parent:
- S70 merged main `751161746887b795149fcd924735ee2d043b43b3`
- scientific run `37423476142`
- artifact `11394610000`
- digest `sha256:8af711f789efb9717eaed0eff882fade7bab37b546f86a35502db82242ad221f`
- verdict **Case C**.

Staged core:
- `src/nmd/v1_multistat_pairwise_row_composer.py`

Staged unit mechanics:
- `tests/test_v1_multistat_pairwise_row_composer.py`

Frozen variable:
- reference: normalized row mean + three zero channels
- treatment: normalized mean/max/min/RMS
- composer capacity: 80 vs 80
- same residual direction
- exact S66 target retained
- no target/representation/head change.

A0 marker:
`research/HIRA-V1-S71-ENABLE-A0`

Marker must remain absent until exact final pre-A0 head passes generic CI on Python 3.10 and 3.12.
