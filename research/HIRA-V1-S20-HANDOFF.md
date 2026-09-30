# HIRA V1 S20 handoff — to S21 Semantic Role-Content Factorization

S20 is frozen as:

`HIRA_V1_S20_STANDARDIZED_TRIADIC_CONSISTENCY_DEV_FAIL`

Canonical authority:
- A0 run `36677124443`
- A0 artifact `11080820457`
- A0 digest `sha256:39769042b9620b5444f5a478a1ef12d63acb43c7abbd5332b8978b5177be4b06`
- TRAIN/DEV run `36677868762`
- artifact `11080892962`
- digest `sha256:7f217c8bd86c4844328fd31b27ab6a97f1daa04e4186b5cd340112090320c220`
- selected epoch **17**
- checkpoint `dc87723defa224f012b513fcb679fe44794e0eb3a7d28cb0a10deb9f5772047d`

Selected DEV:
- fused canonical **0.5963541667**
- fused paraphrase **0.4036458333**
- paired **0.3385416667**
- question-swap **0.734375**
- fused agreement **0.4583333333**
- fused margin **+0.1902903815**
- raw triadic canonical **0.5208333333**
- raw triadic paraphrase **0.3932291667**
- relation canonical **0.4427083333**
- relation paraphrase **0.3515625**
- relation margin **-0.0772121996**
- signature cosine **0.7979244739**
- signature margin **0.0933141845**

## Key result

S20 standardized consistency is not inert:
- A0 standardized mismatch loss **0.4353144765**
- A0 gradients nonzero to both views
- TRAIN consistency term **0.0633 -> peak 0.1474 -> 0.0929**

Despite that, S20 regresses versus S17.

This rules out the simple hypothesis that stronger cross-view evidence consistency is the missing ingredient.

## S21 direction

Return to **S17** as the scientific frontier.

Target:
**semantic role-content factorization / binding generalization**.

Do not carry forward:
- S18 paired-margin term
- S19 raw-softmax JS
- S20 standardized-evidence consistency

Keep:
- S17 norm-balanced shared-gradient optimizer
- S17 relation CE/signature canonicalization
- S14 equal-weight inference fusion
- exact 49,152 physical trainable params
- A13/HIRACore frozen
- zero learned downstream head/router
- full-K/state-once
- wholly fresh authority

Before S21 contract is frozen, inspect:
- the projection triadic scorer's query/state/option composition;
- the relation canonicalizer's role extraction and binding path;
- where role identity and value/content are currently entangled.

The S21 objective must explicitly improve role/content discrimination, not generic view similarity.

No S20 DEV reuse.
No post-DEV tuning.
No Laya/Jev reopening before DEV_READY.
