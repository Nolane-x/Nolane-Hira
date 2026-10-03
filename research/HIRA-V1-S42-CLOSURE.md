# HIRA V1 S42 closure — Cross-View Relational Signature-Geometry Anchoring

Status: **CLOSED — ONE FRESH DEV COMPLETE / PREREGISTERED CASE B**

Issue: #267
PR: #268

## Qualified A0

- run `37105795175`
- artifact `11267578716`
- digest `sha256:d87c03d7ec07c98cdf1922f6fa7eb210559a4929d962243848cf7bac65f12542`
- authority head `6f12bdef22d09fa205e09dbb10f9f577be5b400d`
- outcome `HIRA_V1_S42_A0_CROSS_VIEW_RELATIONAL_GEOMETRY_READY`

## Canonical fresh matched DEV

- run `37107502716`
- artifact `11269981431`
- digest `sha256:ec32ad2c7f01dab73ca54e8f15957553d2e7a3352f674442899cc908adb74f47`
- scientific head `018cf7bb50f81a180faeec608689d37104082418`
- seed **63001**
- outcome `HIRA_V1_S42_MATCHED_CROSS_VIEW_RELATIONAL_GEOMETRY_DEV_COMPLETE`

Exact frozen receipts:
- `research/HIRA-V1-S42-MATCHED-RECEIPT.json`
- `research/HIRA-V1-S42-MATCHED-RECEIPT.md`

All scientific-workflow steps passed:
- TRAIN/DEV
- receipt verification
- integrity freeze
- artifact upload

No second DEV.

## Selected reference — epoch 19

- fused canonical/paraphrase **0.4401042 / 0.4505208**
- relation canonical/paraphrase **0.3125000 / 0.3359375**
- fused agreement **0.5729167**
- fused JS **0.0278331**
- relation agreement **0.6197917**
- signature cosine/margin **0.8660154 / 0.1755615**
- gates **14/22**
- DEV_READY false

## Selected treatment — epoch 14

- fused canonical/paraphrase **0.5234375 / 0.5442708**
- relation canonical/paraphrase **0.4036458 / 0.5000000**
- fused agreement **0.5312500**
- fused JS **0.0465635**
- relation agreement **0.4921875**
- signature cosine/margin **0.8439764 / 0.0839071**
- gates **18/27**
- DEV_READY false

## Treatment minus reference

Correctness:
- fused canonical **+0.0833333**
- fused paraphrase **+0.0937500**
- relation canonical **+0.0911458**
- relation paraphrase **+0.1640625**
- primary canonical **+0.0468750**
- primary paraphrase **-0.0520833**
- fused canonical margin **+0.0451230**
- fused paraphrase margin **+0.1744329**

Transport/stability:
- fused agreement **-0.0416667**
- relation agreement **-0.1276042**
- fused JS **+0.0187304** worse
- same-option cosine **-0.0220390**
- signature discrimination margin **-0.0916545**

## Actual-step relational anchor

- mean anchor **0.02451545**
- conflict rate **0.33680556**
- mean pre-dot **-0.0000202769**
- mean post-dot **-0.0001360411**
- mean applied anchor dot **-0.0001360411**
- mean applied rounding bound **0.0001360467**
- max runtime rounding ratio **0.4995122**
- max W rounding ratio **0**
- selected W norm **11.1203089**

The optimizer-faithful mechanism remained within its frozen precision guards.

## Frozen interpretation

**Case B — correctness retained but relative geometry still collapses.**

The full KxK MSE anchor is mechanically valid and correctness remains materially better than the matched native reference.

However the central S42 target is not solved:
- signature discrimination margin falls **0.0916545**
- treatment absolute margin is only **0.0839071**, below the frozen 0.15 gate
- fused agreement falls **0.0416667**
- relation agreement falls **0.1276042**
- JS increases **0.0187304**

Therefore:

> Preserving the full cross-view similarity matrix in average squared error is too indirect. It does not sufficiently preserve the relative same-option-vs-wrong-option gaps that drive discrimination.

## Closed family

No post-DEV:
- S42 anchor coefficient/slack
- per-signature + matrix mixing
- diagonal/off-diagonal weighting
- margin addition
- alternate matrix norm
- optimizer reinterpretation
- partial detach / gradient mixing
- W-only LR/scheduler
- rank/factorization
- residual scale/bias/nonlinearity
- seed/LR/epoch/batch retry
- gate weakening
- second DEV

No confirmation.
No multilingual probe.
No Laya/Jev evaluation.
Production-ready remains false.

## Next direction

**S43 — Cross-View Relative-Gap Signature Geometry Anchoring**

Change the anchor target only:
preserve each matched option's cross-view similarity **relative to every wrong option**, rather than preserving all absolute pairwise similarities equally.
