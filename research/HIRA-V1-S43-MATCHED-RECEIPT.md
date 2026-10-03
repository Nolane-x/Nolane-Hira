# HIRA V1 S43 matched DEV receipt — Cross-View Relative-Gap Signature Geometry

Status: **FROZEN / ONE FRESH DEV COMPLETE / CASE C**

Issue: #269
PR: #270

## Canonical court

- run `37112732199`
- artifact `11270379708`
- digest `sha256:c1be142d6b14f3d1b80f154fb1e99822d5f624ab5262594d8ecbdd432f8d8d2a`
- scientific head `6928551af8e2702de2b047cfa5a60430d49b29b5`
- seed **64001**
- outcome `HIRA_V1_S43_MATCHED_CROSS_VIEW_RELATIVE_GAP_GEOMETRY_DEV_COMPLETE`
- authority `V1_S43_FRESH_MATCHED_NATIVE_REFERENCE_VS_CROSS_VIEW_RELATIVE_GAP_GEOMETRY_FULL_BILINEAR`

All scientific workflow steps passed:
- fresh TRAIN/DEV
- receipt verifier
- integrity freeze
- artifact upload

## Reference — selected epoch 22

- gates: 13/22
- DEV_READY: **false**
- checkpoint: `3b1c8fd458dfdfbc97ba903330cc6affffbbfdb2d1d9656a5fc053f6d7eba7d7`

Fused:
- canonical **0.5052083333333334**
- paraphrase **0.46875**
- paired **0.265625**
- question-swap **0.7083333333333334**
- agreement **0.65625**
- JS **0.0195333834271878**

Primary:
- canonical **0.5052083333333334**
- paraphrase **0.5078125**

Relation:
- canonical **0.3098958333333333**
- paraphrase **0.3046875**
- agreement **0.5729166666666666**

Signature:
- same-option cosine **0.8953376561403275**
- discrimination margin **0.13020227942615747**

## Treatment — selected epoch 17

- gates: 18/27
- DEV_READY: **false**
- checkpoint: `e56d6d51c5894db30ddae7fbd7e9e721ce84087b69c8ba2364595864acf055e7`

Fused:
- canonical **0.46875**
- paraphrase **0.4348958333333333**
- paired **0.20833333333333334**
- question-swap **0.53125**
- agreement **0.6171875**
- JS **0.033541803481057286**

Primary:
- canonical **0.390625**
- paraphrase **0.421875**

Relation:
- canonical **0.4010416666666667**
- paraphrase **0.4192708333333333**
- agreement **0.4921875**

Signature:
- same-option cosine **0.7783455699682236**
- discrimination margin **0.09878427876780431**

## Treatment minus reference

Correctness:
- fused canonical **-0.03645833333333337**
- fused paraphrase **-0.033854166666666685**
- primary canonical **-0.11458333333333337**
- primary paraphrase **-0.0859375**
- relation canonical **0.09114583333333337**
- relation paraphrase **0.11458333333333331**
- paired **-0.05729166666666666**
- question-swap **-0.17708333333333337**

Transport:
- fused agreement **-0.0390625**
- relation agreement **-0.08072916666666663**
- fused JS **0.014008420053869486**
- same-option cosine **-0.11699208617210388**
- signature discrimination margin **-0.031418000658353165**

## Relative-gap optimizer-step diagnostics

- mean anchor **0.07101457443176899**
- conflict rate **0.4366319444444444**
- mean pre-dot **0.00015790095446713235**
- mean post-dot **-0.00011759933297973994**
- mean applied anchor dot **-0.0001175993687084734**
- mean applied rounding bound **0.00011761250206525176**
- max runtime rounding ratio **0.4995121951219512**
- max W rounding ratio **0**
- selected W norm **13.348763465881348**

## Frozen interpretation

**Case C — protecting the relative-gap direction suppresses the co-adaptation needed for fused/primary correctness.**

Relative-gap anchoring materially improves the specific signature-discrimination failure seen in S41/S42:
- S41 signature-margin delta: about **-0.1201**
- S42 signature-margin delta: about **-0.0917**
- S43 signature-margin delta: **-0.031418000658353165**

However the treatment loses the broader decision gain:
- fused canonical **-0.03645833333333337**
- fused paraphrase **-0.033854166666666685**
- primary canonical **-0.11458333333333337**
- primary paraphrase **-0.0859375**
- paired **-0.05729166666666666**
- question-swap **-0.17708333333333337**

Relation-only correctness still rises, but it no longer converts to the fused/primary decision endpoint.

Broader transport is also not fully solved:
- fused agreement **-0.0390625**
- relation agreement **-0.08072916666666663**
- fused JS **0.014008420053869486**
- treatment signature margin remains **0.09878427876780431**, below the frozen 0.15 gate.

Therefore S43 is not Case A and not DEV_READY. The evidence supports the frozen Case-C direction: the parameter movement that relative-gap projection removes overlaps materially with the movement required for useful joint correctness co-adaptation.

## Safeguards

- post-DEV tuning: **false**
- second DEV: **false**
- sealed confirm: **false**
- multilingual probe: **false**
- production-ready claim: **false**

No second S43 DEV.
No strongest-wrong/hinge/gap-weighting/anchor-mixing retry.
No external Laya/Jev evaluation.
