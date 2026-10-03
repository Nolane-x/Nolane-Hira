# HIRA V1 S43 closure — Cross-View Relative-Gap Signature Geometry Anchoring

Status: **CLOSED — ONE FRESH DEV COMPLETE / CASE C**

Issue: #269
PR: #270

## Qualified A0

- run `37112131898`
- artifact `11269189976`
- digest `sha256:8116aec424a7852a0749a0880f66ecdcfe633ee4c1bb6b570f322f409e338429`
- authority head `ff45888f822f8b6b06218e4758b15d1d57087462`
- outcome `HIRA_V1_S43_A0_CROSS_VIEW_RELATIVE_GAP_GEOMETRY_READY`

## Canonical fresh matched DEV

- run `37112732199`
- artifact `11270379708`
- digest `sha256:c1be142d6b14f3d1b80f154fb1e99822d5f624ab5262594d8ecbdd432f8d8d2a`
- scientific head `6928551af8e2702de2b047cfa5a60430d49b29b5`
- seed **64001**
- outcome `HIRA_V1_S43_MATCHED_CROSS_VIEW_RELATIVE_GAP_GEOMETRY_DEV_COMPLETE`

All scientific workflow steps passed:
- fresh TRAIN/DEV
- receipt verifier
- integrity freeze
- artifact upload

Frozen receipts:
- `research/HIRA-V1-S43-MATCHED-RECEIPT.json`
- `research/HIRA-V1-S43-MATCHED-RECEIPT.md`

No second DEV.

## Reference — selected epoch 22

- gates **13/22**
- DEV_READY false
- fused canonical/paraphrase **0.5052083 / 0.4687500**
- primary canonical/paraphrase **0.5052083 / 0.5078125**
- relation canonical/paraphrase **0.3098958 / 0.3046875**
- fused agreement **0.6562500**
- relation agreement **0.5729167**
- fused JS **0.0195334**
- signature cosine/margin **0.8953377 / 0.1302023**

## Treatment — selected epoch 17

- gates **18/27**
- DEV_READY false
- fused canonical/paraphrase **0.4687500 / 0.4348958**
- primary canonical/paraphrase **0.3906250 / 0.4218750**
- relation canonical/paraphrase **0.4010417 / 0.4192708**
- fused agreement **0.6171875**
- relation agreement **0.4921875**
- fused JS **0.0335418**
- signature cosine/margin **0.7783456 / 0.0987843**

## Treatment minus reference

Correctness:
- fused canonical **-0.0364583**
- fused paraphrase **-0.0338542**
- primary canonical **-0.1145833**
- primary paraphrase **-0.0859375**
- relation canonical **+0.0911458**
- relation paraphrase **+0.1145833**
- paired **-0.0572917**
- question-swap **-0.1770833**

Transport:
- fused agreement **-0.0390625**
- relation agreement **-0.0807292**
- fused JS **+0.0140084** worse
- same-option cosine **-0.1169921**
- signature discrimination margin **-0.0314180**

## What S43 did solve

The signature-discrimination degradation is much smaller than in the two preceding anchored stages:

- S41: about **-0.1201**
- S42: about **-0.0917**
- S43: **-0.0314**

Thus the relative-gap target is much closer to the transport quantity that actually matters than:
- per-signature cosine anchoring;
- full KxK absolute-similarity MSE.

## What S43 did not solve

The broader decision endpoint collapses:
- fused canonical/paraphrase become negative vs reference;
- raw primary canonical/paraphrase fall strongly;
- paired and question-swap fall;
- agreement falls and JS worsens.

Relation-only correctness remains positive, but it does not convert into useful fused decision correctness.

Treatment is not DEV_READY and still fails the absolute signature-margin gate:
- treatment signature margin **0.0987843**
- frozen gate **>=0.15**

## Frozen interpretation

**Case C — protecting the relative-gap direction suppresses movement required for useful correctness co-adaptation.**

This is not evidence that the relative-gap target is wrong. It is evidence that forcing one shared native representation to serve both:
1. transport-stable option identity geometry; and
2. aggressive correctness co-adaptation

creates a structural conflict.

S38/S41 show correctness wants representation co-adaptation.
S43 shows constraining the transport-critical gap direction reduces that correctness effect.

Therefore the next stage must change **representation ownership**, not tune another anchor.

## Closed S43 family

No post-DEV:
- strongest-wrong variant
- hinge/numeric target margin
- gap weighting
- S41/S42 anchor mixing
- coefficient/slack
- alternate norm
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
No external Laya/Jev evaluation.
Production-ready remains false.

## Next direction

**S44 — Private Correction Representation Fork**

Goal:
- keep the native representation on the exact matched transport trajectory;
- create a private post-encoder correction representation that may co-adapt with the full bilinear correctness head;
- never feed the private representation back into native signatures, native relation geometry, or primary transport;
- retain state-once encoder execution.
