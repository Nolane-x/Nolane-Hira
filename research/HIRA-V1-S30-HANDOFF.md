# HIRA V1 S30 handoff — to S31 Matched Local-vs-Global Relation Canonicalization

S30 is frozen as:

`HIRA_V1_S30_MATCHED_ADAPTATION_DEV_COMPLETE`

No arm reached DEV_READY.

## Canonical authority

A0:
- run `36940968046`
- artifact `11199314198`
- digest `sha256:2ec456fc02ca498729358d21cc7a8205aa8807230514902ac73f2ddaaebad013`
- outcome `HIRA_V1_S30_A0_MATCHED_ADAPTATION_READY`

Matched TRAIN/DEV:
- run `36941827140`
- artifact `11201303347`
- digest `sha256:aa7779664411acdf3e44d3bbef8f80688f1a55c3d20240b99050b7757e6a150b`
- scientific head `41a62c63eff9244c7ccc6bddd57ddf11d2a64f3a`
- outcome `HIRA_V1_S30_MATCHED_ADAPTATION_DEV_COMPLETE`

Attention-only:
- selected epoch **15**
- checkpoint `04fad98500ebc802f6d9441e6cdbc0274a170f7d3eb23d2f12655772c274883a`
- fused canonical/paraphrase **0.5078125 / 0.2578125**
- paired **0.2447916667**
- relation canonical/paraphrase **0.4036458333 / 0.2734375**
- relation margins **-0.2553525219 / -1.0135451568**
- signature cosine/discrimination **0.8607332458 / 0.1111970699**
- gates **12/22 PASS**

FFN-only:
- selected epoch **17**
- checkpoint `f7b6667c0c009776fe2910646b8d27e4ac26b64613e989c3a68ffc3b06304620`
- fused canonical/paraphrase **0.546875 / 0.2994791667**
- paired **0.2552083333**
- relation canonical/paraphrase **0.4401041667 / 0.2630208333**
- relation margins **-0.0848894926 / -0.5236856043**
- signature cosine/discrimination **0.8562618246 / 0.1187967975**
- gates **13/22 PASS**

## Matched interpretation

FFN-only improves some canonical/fused endpoints but loses on several transport/sensitivity endpoints.

The evidence is split.

Do not select a final-block sublayer family from S30.

## S31 target

**Matched Local-vs-Global Relation Canonicalization Court**

Base both arms on exact S17:
- final attention LoRA 16,384
- shared projection 32,768
- total 49,152
- original A13 frozen
- HIRACore frozen
- S13 relation expert
- S14 equal fusion
- S15 relation detach
- S17 relation-priority norm-balanced gradients
- S17 primary block unchanged

Control relation block:
- `0.10 * relation CE + 0.15 * local cross-view signature canonicalization`

Treatment relation block:
- `0.10 * relation CE + 0.15 * global cross-case relation contrastive`

Global contrastive operator:
- select the gold relation signature for every semantic query
- canonical view and paraphrase view of the same semantic query are positives
- every other semantic query in the batch is a negative
- L2 normalize signatures
- symmetric InfoNCE
- fixed temperature **0.10**
- no added parameters/state

A0 must prove:
- inference identity control vs treatment
- global loss permutation equivariance under matched query permutation
- low loss on well-separated matched signatures
- materially higher loss when positive pairing is shuffled
- nonzero finite gradients to both views
- no added params
- exact 49,152 surface
- full-K/state-once/mechanics

Use one wholly fresh matched S31 authority.
No S30 DEV rows.
No post-DEV coefficient/temperature tuning.
No second DEV.
No Laya/Jev reopening before DEV_READY.
