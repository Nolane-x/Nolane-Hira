# HIRA V1 S27 handoff — to S28 Anchor-Level Cross-View Factor Transport

S27 is frozen as:

`HIRA_V1_S27_BLOCKWISE_CANONICALIZATION_DEV_FAIL`

Canonical authority:
- A0 run `36859567980`
- A0 artifact `11161461301`
- A0 digest `sha256:2c9f61c218679f8ed24f557fda56436a63c2109520385b48c6583e77d3cb06d0`
- TRAIN/DEV run `36860808313`
- artifact `11162995529`
- artifact digest `sha256:7e4ec757d3b252c15d7296fb3ed535373326646a5da5e4f0d045cc093694272f`
- selected epoch **7**
- checkpoint `7ee131ad80109c572d5165541bc684ee138921bfcddfd58d6702dda362070a15`

Selected DEV:
- fused canonical/paraphrase **0.5885416667 / 0.5338541667**
- paired **0.3333333333**
- question-swap **0.6614583333**
- fused agreement **0.5**
- fused JS **0.0644452251**
- fused canonical/paraphrase margins **+0.0555786624 / -0.0332792709**
- relation canonical/paraphrase **0.46875 / 0.4713541667**
- relation margins **-0.0795914428 / -0.0566431495**
- relation agreement **0.5755208333**
- signature cosine **0.7606588602**
- signature discrimination margin **0.0006603413**

Best residual probes:
- fused JS reaches **0.0452790880**
- relation agreement reaches **0.609375**
- signature cosine reaches **0.7830425799**
- signature discrimination margin never exceeds **0.0193961563**
- relation canonical margin remains negative; best **-0.0767449367**
- relation paraphrase margin remains negative; best **-0.0084951470**

## Stop rule applied

Do not retry S27 with higher canonicalization strength, different block weights, margin, seed, LR or epoch count.

## S28 target

**Anchor-Level Cross-View Factor Transport**

Why:
S27 improves final-signature alignment but not correct-vs-wrong relation separation. The final signature is downstream of query-conditioned role/value anchor extraction. Equivalent wordings may still be producing different latent anchors and then merely being forced into similar final relation vectors.

Freeze before A0:
- exact S26/S27 inference
- exact **81,920** trainable surface
- no new learned state
- S21 primary unchanged
- factorized relation logits/signatures unchanged
- S14 equal fusion unchanged
- S25 gradient ownership unchanged

Training-only S28 mechanism:
- expose normalized query-conditioned **state role anchor**
- expose normalized **state value/content anchor**
- optionally expose corresponding option-side anchors for diagnostics, but do not change inference
- canonical/paraphrase anchor alignment occurs before option relation construction
- anti-collapse negatives use other semantic queries/cases, not option IDs
- role/value anchor weights frozen before A0
- existing downstream relation CE remains

Required A0:
- inference bit-identity vs S27/S26
- zero added parameters
- identical wording anchors produce zero/near-zero alignment loss
- role-anchor mismatch activates role component
- value-anchor mismatch activates value component
- negative/cross-case court prevents constant-anchor collapse
- option permutation does not affect state-anchor objective
- relation-private/shared-LoRA gradients nonzero
- cross-private leakage zero
- full-K/state-once/checkpoint replay preserved

Use wholly fresh S28 authority.
No S27 DEV rows may enter S28 TRAIN/DEV.
No Laya/Jev until DEV_READY.
