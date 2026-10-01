# HIRA V1 S28 closure — Anchor-Level Cross-View Factor Transport

Status: **CLOSED — DEV FAIL / ANCHOR TRANSPORT SUCCEEDS, ANCHOR DISCRIMINATION DOES NOT**

Issue: #239  
PR: #240

## Authority

Canonical A0:
- run `36867793857`
- artifact `11164614334`
- digest `sha256:b3033c40357ba4b9b8cca378377a6d7e87f27f7e8449489e0beef20eb7fdec22`
- authority head `73cc786ee59f74098665024648955704718aa8a0`
- outcome `HIRA_V1_S28_A0_ANCHOR_FACTOR_TRANSPORT_READY`

Fresh TRAIN/DEV:
- run `36872373034`
- artifact `11168765249`
- artifact digest `sha256:5a5e28ea906cd5b86bff11177818382a1487934ab7ed99d53eaacf1e8a67bb5e`
- scientific head `38905d245caaf4fced86ccecacf47e665f52ae03`
- outcome `HIRA_V1_S28_ANCHOR_FACTOR_TRANSPORT_DEV_FAIL`
- selected epoch **20**
- checkpoint SHA256 `b9e0f59b5e624981db79f6e0e2525fe4f350ddfaf727164fdd8aefd2717fd8cd`

Enable-head generic CI:
- run `36872380061`
- PASS

No second DEV run.
No post-DEV tuning.
No sealed confirmation.
No multilingual probe.
No Laya/Jev reopening.

## Frozen intervention

S28 changes training only:
- exact S26/S27 production inference
- exact **81,920** trainable surface
- shared A13 LoRA **16,384**
- primary-private projection **32,768**
- relation-private projection **32,768**
- zero added anchor-transport parameters
- S14 equal standardized full-K fusion unchanged
- S25 private/shared gradient ownership unchanged

Training-only anchor transport:
- query-conditioned state role anchor
- role-orthogonal state value/content anchor
- same-query cross-view alignment
- hardest wrong cross-query separation
- role/value weights **0.50 / 0.50**
- separation margin **0.20**
- outer coefficient **0.15**
- replaces S27 blockwise output canonicalization

## A0 result

A0 qualified the mechanism:
- production primary/relation/signature/fused identity max abs **0**
- role-anchor reference error **0**
- value-anchor reference error **0**
- identical well-separated anchor loss **0**
- role-only mismatch localized to role factor
- value-only mismatch localized to value factor
- constant-anchor collapse pays ~**0.20** separation
- option permutation has **0** effect on state-anchor objective
- cross-private leakage **0 / 0**
- intended private/shared gradients nonzero
- full-K/state-once PASS

The DEV result is therefore not caused by an inactive anchor objective.

## Selected DEV — epoch 20

Fused:
- canonical accuracy **0.5260416667**
- paraphrase accuracy **0.734375**
- paired both-correct **0.203125**
- question-swap change **0.453125**
- cross-view selected-choice agreement **0.5026041667**
- cross-view mean JS **0.0705156003**
- canonical signed margin **-0.1302195030**
- paraphrase signed margin **+0.3248527013**

Relation:
- canonical accuracy **0.4296875**
- paraphrase accuracy **0.4583333333**
- canonical signed margin **-0.0796973606**
- paraphrase signed margin **-0.0394903719**
- cross-view agreement **0.6015625**

Final factorized signature:
- same-option cosine **0.6421900640**
- same-vs-strongest-wrong margin **-0.0111360058**

Anchor transport diagnostics:
- role-anchor cross-view cosine **0.9185716013**
- role-anchor same-vs-wrong margin **-0.0099708166**
- value-anchor cross-view cosine **0.8572883606**
- value-anchor same-vs-wrong margin **-0.0366763414**

Primary:
- canonical accuracy **0.4479166667**
- paraphrase accuracy **0.5911458333**

Mechanics:
- option-order flip **0**
- probability-mass error **1.1920929e-7**
- full-K PASS
- state-once PASS
- cross-private leakage **0 / 0**
- exact trainable surface **81,920**

## Gate result

PASS **21 / 31**.

FAIL **10 / 31**:
- fused canonical >= **0.85**
- paired >= **0.75**
- question-swap >= **0.80**
- fused agreement >= **0.95**
- fused JS <= **0.05**
- fused canonical margin >= **0.15**
- relation canonical >= **0.80**
- relation canonical margin >= **0.15**
- same-option final-signature cosine >= **0.90**
- final-signature discrimination margin >= **0.15**

DEV_READY is not authorized.

## Best observed DEV values across 24 epochs

- fused canonical **0.5260416667**, epoch 20
- fused paraphrase **0.734375**, epoch 20
- paired both-correct **0.203125**, epoch 19/20
- question-swap **0.453125**, epoch 17/20
- fused agreement **0.5026041667**, epoch 20
- fused JS minimum **0.0705156003**, epoch 20
- fused canonical margin best **-0.1302195030**, epoch 20
- fused paraphrase margin **+0.3248527013**, epoch 20
- relation canonical **0.4296875**, epoch 19/20
- relation paraphrase **0.4713541667**, epoch 19
- relation canonical margin best **-0.0796973606**, epoch 20
- relation paraphrase margin best **-0.0394903719**, epoch 20
- relation agreement **0.6015625**, epoch 20
- final-signature cosine **0.6651761979**, epoch 21
- final-signature discrimination margin **0.0020386775**, epoch 2
- role-anchor cosine **0.9266317983**, epoch 21
- role-anchor discrimination margin best **-0.0088916677**, epoch 22
- value-anchor cosine **0.8881069670**, epoch 21
- value-anchor discrimination margin best **-0.0205161761**, epoch 21

No epoch makes anchor same-vs-wrong discrimination positive.

## Optimization dynamics

Epoch 1 → 24:
- total loss **1.8619660512 → 0.7402530101**
- decision loss **1.5604186778 → 0.5710716657**
- relation binding loss **1.4886878580 → 0.6985140666**
- anchor transport loss **0.4513919391 → 0.2073300844**
- fused consistency JS **0.0694033029 → 0.0027071679**

Gradient ownership remains healthy:
- minimum primary-private gradient L1 **60.2588500977**
- minimum relation-private gradient L1 **6.0638070107**
- minimum primary shared-LoRA gradient L1 **6.1991176605**
- minimum relation shared-LoRA gradient L1 **0.7505410314**
- cross-private leakage **0 / 0**
- mean shared-LoRA conflict rate **0.4574652778**

This is not an optimizer/harness failure.

## Scientific interpretation

S28 confirms that **latent cross-view transport can be learned**:
- role-anchor cosine exceeds **0.92**
- value-anchor cosine approaches **0.89**

However, the same anchors remain non-discriminative:
- role same-vs-wrong margin remains negative at every epoch
- value same-vs-wrong margin remains negative at every epoch
- relation margins remain negative
- final-signature discrimination remains essentially zero

The selected model also develops a large view asymmetry:
- fused canonical **52.60%**
- fused paraphrase **73.44%**
- canonical margin **-0.130**
- paraphrase margin **+0.325**

Therefore the residual is not merely wording transport.

The key missing property is **query-specific discriminative binding**. Equivalent wording can map to similar latent anchors while different semantic queries also occupy overlapping anchor neighborhoods. The current state-only anchor representation lacks enough explicit query identity in the relation score.

## Track closure

Close:
**state-anchor cross-view transport as a sufficient solution**.

Do not create S28b by:
- increasing anchor coefficient
- changing role/value 0.50/0.50 weights
- increasing separation margin
- changing negative mining only
- retrying seed/LR/epochs
- weakening gates
- stacking S27 and S28 canonicalization losses

## Next controlled direction

S11/S12/S21 already exercised query-role localization / query-conditioned state-option role anchors, and the older W7 line already falsified an explicit smooth-AND/conjunctive factor branch against an equal-data free-form control. Those families must not be reopened under a new S29 label.

S6-S28 share a more important limitation: every v1 track that adapts A13 has only adapted the **four attention linears of the final BERT block**. The block's feed-forward semantic transform remains frozen.

S29 should therefore perform a controlled architectural reset to the strongest semantic frontier, S17:

**S29 — Final-Block Attention+FFN LoRA Semantic Adaptation**

Controlled hypothesis:
- S17 already establishes strong semantic discrimination with the existing attention-LoRA + shared projection surface;
- S26-S28 show that downstream factor/canonicalization/transport rules can align evidence without producing enough semantic separation;
- the missing capacity may therefore be inside the token-wise nonlinear semantic transform of the last A13 block, not another downstream scoring heuristic.

Freeze from S17:
- S14 equal standardized full-K inference/fusion;
- S13 relation expert;
- S17 loss partition;
- S17 relation-priority norm-balanced shared-gradient rule;
- same state-once/full-K/opaque-option semantics;
- same 32,768-parameter shared relation projection.

Change only the final A13 adaptation surface:
- retain rank-8 LoRA on final-block Q/K/V/attention-output: **16,384 params**;
- add rank-8 LoRA on final-block `intermediate.dense` (256→1024): **10,240 params**;
- add rank-8 LoRA on final-block FFN `output.dense` (1024→256): **10,240 params**;
- total A13 LoRA: **36,864 params**;
- shared projection: **32,768 params**;
- exact physical trainable surface: **69,632 params**;
- all original A13 weights remain frozen;
- HIRACore frozen;
- no new downstream scorer/router/gate/calibrator.

A0 must prove:
- zero-init exact S17 inference identity before training;
- exact module paths/shapes and **69,632** trainable surface;
- original A13/HIRACore frozen;
- FFN LoRA receives nonzero gradients on a real semantic court;
- existing attention LoRA and projection also receive gradients;
- full-K/state-once/option permutation/probability mass;
- S17 gradient-balancing contract remains valid;
- checkpoint roundtrip/replay includes all six LoRA modules.

Only a qualified A0 may open wholly fresh S29 TRAIN/DEV.

This is an encoder-capacity intervention, not permission to widen ranks, adapt more layers, retry seeds, or tune against exposed S28/S17 DEV.

Production-ready remains false.
Laya/Jev parity remains unestablished.
