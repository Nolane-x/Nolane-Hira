# HIRA V1 S27 closure — Blockwise Cross-View Factor Canonicalization

Status: **CLOSED — DEV FAIL / BLOCKWISE OUTPUT CANONICALIZATION IMPROVES TRANSPORT BUT NOT RELATION DISCRIMINATION**

Issue: #237  
PR: #238

## Authority

Canonical A0:
- run `36859567980`
- artifact `11161461301`
- digest `sha256:2c9f61c218679f8ed24f557fda56436a63c2109520385b48c6583e77d3cb06d0`
- authority head `93fc8ebef17ca936872c5c72899bacfd232cfd67`
- outcome `HIRA_V1_S27_A0_BLOCKWISE_CANONICALIZATION_READY`

Fresh TRAIN/DEV:
- run `36860808313`
- artifact `11162995529`
- artifact digest `sha256:7e4ec757d3b252c15d7296fb3ed535373326646a5da5e4f0d045cc093694272f`
- scientific head `a7d778b8aa6fa5153c48d366f09c65e12b2a33e4`
- outcome `HIRA_V1_S27_BLOCKWISE_CANONICALIZATION_DEV_FAIL`
- selected epoch **7**
- checkpoint SHA256 `7ee131ad80109c572d5165541bc684ee138921bfcddfd58d6702dda362070a15`

Exact-head generic CI on authority head:
- run `36860817752`: PASS

No second DEV run.
No post-DEV tuning.
No sealed confirmation.
No multilingual probe.
No Laya/Jev reopening.

## Frozen intervention

S27 changes **training canonicalization only**.

Inference is exactly S26:
- primary logits identity max abs **0**
- relation logits identity max abs **0**
- factorized signature identity max abs **0**
- fused logits identity max abs **0**

Inherited capacity:
- shared A13 LoRA **16,384**
- primary-private projection **32,768**
- relation-private projection **32,768**
- exact trainable surface **81,920**
- no added learned parameters

S27 loss:
- split 256D factorized signature into **128D role + 128D value**
- normalize blocks independently
- same-option alignment + wrong-option separation per block
- fixed block weights **0.50 / 0.50**
- separation margin **0.20**
- outer canonicalization coefficient **0.15**

## A0 result

Mechanics are qualified:
- identical-view blockwise loss **0**
- role-only mismatch: role **0.35**, value **0**
- value-only mismatch: role **0**, value **0.35**
- both-factor mismatch: role **0.35**, value **0.35**
- option-permutation loss error **0**
- cross-private leakage **0 / 0**
- all intended private/shared gradients nonzero
- full-K/state-once PASS

Thus fresh DEV failure is not an inactive objective.

## Selected DEV — epoch 7

Fused:
- canonical accuracy **0.5885416667**
- paraphrase accuracy **0.5338541667**
- paired both-correct **0.3333333333**
- question-swap change **0.6614583333**
- cross-view selected-choice agreement **0.5000000000**
- cross-view mean JS **0.0644452251**
- canonical signed margin **+0.0555786624**
- paraphrase signed margin **-0.0332792709**

Relation:
- canonical accuracy **0.46875**
- paraphrase accuracy **0.4713541667**
- canonical signed margin **-0.0795914428**
- paraphrase signed margin **-0.0566431495**
- cross-view agreement **0.5755208333**

Factorized signature:
- same-option cosine **0.7606588602**
- same-vs-strongest-wrong margin **0.0006603413**

Primary:
- canonical accuracy **0.4505208333**
- paraphrase accuracy **0.4453125**
- cross-view agreement **0.4244791667**

Mechanical:
- option-order flip **0**
- probability-mass error **1.1920929e-7**
- full-K PASS
- state-once PASS
- cross-private leakage **0 / 0**
- exact surface **81,920**
- canonicalization added params **0**

## Gate result

PASS **21 / 31**.

FAIL **10 / 31**:
- fused canonical >= **0.85**
- paired >= **0.75**
- question-swap >= **0.80**
- fused cross-view agreement >= **0.95**
- fused JS <= **0.05** at selected epoch
- fused canonical margin >= **0.15**
- relation canonical >= **0.80**
- relation canonical margin >= **0.15**
- same-option signature cosine >= **0.90**
- signature discrimination margin >= **0.15**

DEV_READY is not authorized.

## Best observed DEV values across 24 epochs

- fused canonical **0.5885416667**, epoch 7
- fused paraphrase **0.6223958333**, epoch 12
- paired **0.3333333333**, epoch 7
- question-swap **0.6614583333**, epoch 7
- fused agreement **0.5755208333**, epoch 11
- fused JS minimum **0.0452790880**, epoch 23
- fused canonical margin **+0.0594560029**, epoch 9
- fused paraphrase margin **+0.1084950299**, epoch 15
- relation canonical **0.4713541667**, epoch 2
- relation paraphrase **0.5338541667**, epoch 21
- best relation canonical margin **-0.0767449367**, epoch 12
- best relation paraphrase margin **-0.0084951470**, epoch 15
- relation cross-view agreement **0.609375**, epoch 11
- same-option signature cosine **0.7830425799**, epoch 9
- signature discrimination margin **0.0193961563**, epoch 20

Important: S27 can push **transport/alignment** farther, and fused JS can eventually pass its <=0.05 gate, yet relation discrimination remains weak.

## Optimization dynamics

Epoch 1 → 24:
- total loss **1.7176638072 → 0.7321661375**
- decision loss **1.4440640906 → 0.5715864549**
- relation binding loss **1.2452835416 → 0.6405849420**
- canonicalization loss **0.4498850840 → 0.1872296284**
- fused consistency JS **0.0547475986 → 0.0027572203**

Gradient ownership remains healthy:
- minimum primary-private gradient L1 **50.4835891724**
- minimum relation-private gradient L1 **9.2530698776**
- minimum primary shared-LoRA gradient L1 **3.8688578606**
- minimum relation shared-LoRA gradient L1 **0.9553121924**
- cross-private leakage **0 / 0**
- mean shared-LoRA conflict rate **0.4305555556**

Training strongly optimizes the frozen objective. This is not an optimizer crash.

## Scientific interpretation

S27 partly validates the S26 diagnosis: factor-specific canonicalization improves wording-view transport.

Evidence:
- selected relation cross-view agreement reaches **57.55%**
- best relation agreement reaches **60.94%**
- selected signature cosine reaches **0.7607**
- best signature cosine reaches **0.7830**
- fused JS reaches **0.0453** at epoch 23

But this transport does **not** become semantic discrimination:
- selected whole-signature discrimination margin is effectively zero (**0.00066**)
- best margin is only **0.0194**
- relation canonical margin remains negative at every epoch
- relation paraphrase margin also remains negative at every epoch
- relation accuracy stays near 47–53%

Therefore the dominant residual is no longer simply output-signature cross-view drift.

The output relation signature can be made substantially more stable while the underlying **query-conditioned role/value anchors** remain insufficiently semantically anchored. Canonicalizing only the final relation vector is too late in the computation: it can align outputs without guaranteeing that equivalent wording extracts the same role anchor and same value/content anchor before option scoring.

## Track closure

Close:
**blockwise output-signature canonicalization as a sufficient solution**.

Do not create S27b by:
- increasing blockwise coefficient
- changing 0.50/0.50 block weights
- increasing separation margin
- changing seed/LR/epochs
- weakening gates
- changing inference under the S27 label

## Next controlled direction

S28 should move the cross-view constraint **upstream** while keeping inference unchanged.

Proposed track:
**S28 — Anchor-Level Cross-View Factor Transport**

Controlled hypothesis:
- equivalent wording should extract the same query-conditioned role anchor before option scoring;
- equivalent wording should extract the same state value/content anchor before option scoring;
- canonicalize those latent anchors directly rather than only the final option relation signature.

Freeze:
- exact S26/S27 inference
- exact 81,920 trainable surface
- S14 fusion
- S25 gradient ownership
- no learned new params/state

S28 training-only objective:
- expose normalized state-role anchor and state-value anchor from the factorized relation operator;
- align canonical/paraphrase role anchors directly;
- align canonical/paraphrase value anchors directly;
- add negative/anti-collapse structure using mismatched semantic-query anchors within batch;
- fixed role/value anchor weighting preregistered before A0;
- preserve existing relation CE and inference.

A0 must prove inference identity, anchor-view localization, anti-collapse behavior, gradient ownership, option permutation, full-K/state-once and zero added params.

Production-ready remains false.
Laya/Jev parity remains unestablished.
