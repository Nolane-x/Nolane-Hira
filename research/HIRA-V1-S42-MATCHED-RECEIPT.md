# HIRA V1 S42 matched DEV receipt — Cross-View Relational Signature Geometry

Status: **FROZEN / ONE FRESH DEV COMPLETE / CASE B**

Issue: #267
PR: #268

## Canonical court

- run `37107502716`
- artifact `11269981431`
- digest `sha256:ec32ad2c7f01dab73ca54e8f15957553d2e7a3352f674442899cc908adb74f47`
- scientific head `018cf7bb50f81a180faeec608689d37104082418`
- seed **63001**
- outcome `HIRA_V1_S42_MATCHED_CROSS_VIEW_RELATIONAL_GEOMETRY_DEV_COMPLETE`
- authority `V1_S42_FRESH_MATCHED_NATIVE_REFERENCE_VS_CROSS_VIEW_RELATIONAL_GEOMETRY_FULL_BILINEAR`

All court steps passed:
- fresh TRAIN/DEV
- receipt verifier
- integrity freeze
- artifact upload

## Reference — selected epoch 19

- gates: 14/22
- DEV_READY: **false**
- checkpoint: `1321bcc19ab465ac8613f09b7f7d18bd24cd27d0c657249b3b7ae700fda77b48`

Fused:
- canonical **0.4401041666666667**
- paraphrase **0.4505208333333333**
- agreement **0.5729166666666666**
- JS **0.02783307945355773**

Relation:
- canonical **0.3125**
- paraphrase **0.3359375**
- agreement **0.6197916666666666**

Signature:
- same-option cosine **0.8660154342651367**
- discrimination margin **0.17556150878469148**

## Treatment — selected epoch 14

- gates: 18/27
- DEV_READY: **false**
- checkpoint: `5db620bcf4b92c8b1a45341842e608f4becf55449eb114eedd39231094433671`

Fused:
- canonical **0.5234375**
- paraphrase **0.5442708333333334**
- agreement **0.53125**
- JS **0.04656350084890922**

Relation:
- canonical **0.4036458333333333**
- paraphrase **0.5**
- agreement **0.4921875**

Signature:
- same-option cosine **0.8439763933420181**
- discrimination margin **0.0839070538058877**

## Treatment minus reference

Correctness:
- fused canonical **0.08333333333333331**
- fused paraphrase **0.09375000000000006**
- relation canonical **0.09114583333333331**
- relation paraphrase **0.1640625**
- primary canonical **0.046875**
- primary paraphrase **-0.052083333333333315**

Transport:
- fused agreement **-0.04166666666666663**
- relation agreement **-0.12760416666666663**
- fused JS **0.01873042139535149**
- same-option cosine **-0.02203904092311859**
- signature discrimination margin **-0.09165445497880378**

## Relational-anchor diagnostics

- mean anchor **0.024515447988890163**
- conflict rate **0.3368055555555556**
- mean pre-dot **-0.000020276932636712444**
- mean post-dot **-0.00013604106213845982**
- mean applied anchor dot **-0.00013604105968897514**
- mean applied rounding bound **0.00013604671327339618**
- max runtime rounding ratio **0.4995121951219512**
- max W rounding ratio **0**
- selected W norm **11.120308876037598**

## Frozen interpretation

**Case B — correctness retained but relative geometry still collapses.**

S42 retains a meaningful correctness gain:
- fused canonical **+8.33pp**
- fused paraphrase **+9.38pp**
- relation canonical **+9.11pp**
- relation paraphrase **+16.41pp**

But its central transport target is not protected:
- signature discrimination margin **−9.17pp**
- treatment absolute discrimination margin **0.08391**, below the frozen 0.15 gate
- fused agreement **−4.17pp**
- relation agreement **−12.76pp**
- fused JS **+0.01873** worse

Therefore full-matrix KxK MSE is too indirect: it preserves average similarity geometry but does not sufficiently preserve the **relative same-option-vs-wrong-option gaps** that define discrimination.

No post-DEV S42 tuning is authorized.
No second DEV.
No confirmation.
No external Laya/Jev evaluation.
Production-ready remains false.
