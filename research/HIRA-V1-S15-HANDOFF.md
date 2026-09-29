# HIRA V1 S15 handoff — to S16 Shared-Surface Gradient Surgery

S15 is frozen as:

`HIRA_V1_S15_GRADIENT_ISOLATED_FUSION_DEV_FAIL`

Canonical authority:
- run `36557765610`
- artifact `11029253333`
- digest `sha256:c15025d02b0b2e4afc3b2f93fdb104c1a8037420a84bb8530d49ee3df2b003c4`
- selected epoch **12**
- checkpoint `b14b644d739cff5559d3580e15f33e53b09f3e6ccbb6a3e6b6ee49d2ae1cc64e`

Selected DEV:
- fused canonical accuracy **0.51822917**
- paired both-correct **0.26041667**
- question-swap **0.671875**
- fused cross-view agreement **0.625**
- fused signed margin **0.01068631**
- raw triadic canonical accuracy **0.57552083**
- relation canonical accuracy **0.2890625**
- relation signed margin **-0.15570846**
- signature cosine **0.87221908**
- signature margin **0.02479495**

Key diagnosis:
- direct fused-primary gradient to relation logits is exactly zero;
- relation auxiliary remains trainable;
- nevertheless relation quality collapses versus S14;
- therefore interference remains in shared A13-LoRA/projection parameters.

S16 target: **Shared-Surface Gradient Surgery**

Controlled mechanism:
1. retain S14/S15 inference exactly;
2. split training objective into primary block and relation-preservation block;
3. obtain gradients of each block over the same 49,152 physical parameters;
4. compute global or preregistered per-block cosine/dot conflict diagnostics;
5. if gradients conflict, project the primary gradient away from the relation gradient (or use a preregistered symmetric PCGrad rule);
6. apply one combined parameter update;
7. add zero learned parameters.

S16 must preregister the exact projection formula before A0/TRAIN exposure.

Do not:
- reuse S15 DEV;
- change fusion weights or epsilon;
- increase model capacity by default;
- fit a learned optimizer/router;
- reopen Laya/Jev before DEV_READY.
