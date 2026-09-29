# HIRA V1 S16 handoff — to S17 Norm-Balanced Shared Gradient Optimization

S16 is frozen as:

`HIRA_V1_S16_SHARED_GRADIENT_SURGERY_DEV_FAIL`

Canonical authority:
- run `36565350686`
- artifact `11031938669`
- digest `sha256:518dbe1474abea876a314555b8b648357dce0744d798ff454fe7c8e58eb221ef`
- selected epoch **24**
- checkpoint `2b46263a9a6b0f66c64b0a4d6f486b7303637de49c67531ef29a822962939bd3`

Selected DEV:
- fused canonical **0.5520833333**
- paired **0.3020833333**
- question-swap **0.6302083333**
- fused agreement **0.546875**
- fused margin **0.0233745320**
- raw triadic canonical **0.5598958333**
- relation canonical **0.34375**
- relation margin **-0.0640710592**
- signature cosine **0.8635843198**
- signature margin **0.0359043240**

TRAIN:
- mean gradient conflict rate **0.3767361111**
- epoch-24 mean primary gradient norm **4.2758044302**
- epoch-24 mean relation gradient norm **0.0354909398**

## New bottleneck

S16 proved anti-alignment exists, but relation gradients are also dramatically smaller than primary gradients.

Conflict projection only removes anti-relation components. It does not guarantee comparable directional influence when gradients are compatible.

## S17 target

**Norm-Balanced Shared Gradient Optimization**

Keep:
- S14/S15/S16 inference exactly
- exact 49,152 trainable physical params
- original A13 and HIRACore frozen
- zero learned optimizer/router params
- same primary/relation loss partition
- same loss coefficients

Preregister a deterministic parameter-free rule:

1. compute raw shared gradients `g_p` and `g_r`;
2. convert each nonzero gradient vector to unit direction:
   `u_p = g_p / (||g_p|| + eps)`
   `u_r = g_r / (||g_r|| + eps)`;
3. if `dot(u_p,u_r) < 0`, relation-priority project `u_p` away from `u_r`;
4. combine the two directions with exact equal weight;
5. rescale the combined direction by a fixed preregistered reference norm derived only from the current raw gradients, not DEV history;
6. then apply existing global clip and AdamW.

A strong non-tuned reference is the arithmetic mean raw norm:
`s = 0.5 * (||g_p|| + ||g_r||)`

and:
`g = s * normalize(u_p' + u_r)`

This preserves current-batch scale while preventing one objective from dominating direction solely due to gradient magnitude.

S17 must freeze the exact formula before A0.

Do not:
- use S16 DEV for fitting;
- choose gradient balance coefficient after exposure;
- add capacity;
- add learned optimizer state beyond AdamW;
- reopen Laya/Jev before DEV_READY.
