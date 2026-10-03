# HIRA V1 S37 closure — Query-Gated Native Signature Correctness Readout

Status: **CLOSED — MATCHED DEV COMPLETE / PREREGISTERED CASE C / DIAGONAL QUERY-SIGNATURE INTERACTION INSUFFICIENT**

Issue: #257
PR: #258

## Canonical authority

A0:
- run `37078578256`
- artifact `11257573101`
- digest `sha256:adbff1f2af80978615f437d2350f4dd132905a42aeb18b39225c7d50fc0c31e6`
- authority head `00d166ca9950359aeada6b2f18e94a5471bfb102`
- outcome `HIRA_V1_S37_A0_QUERY_GATED_READOUT_READY`

Fresh matched TRAIN/DEV:
- run `37079989756`
- artifact `11259139433`
- artifact digest `sha256:1f6fbd32881f3ac01fae93250a1407ee519adab6a8f516cdf6564174203661af`
- scientific head `fc88dc9f8d8a4e94a8ef04c1b9c974b18f1054f7`
- outcome `HIRA_V1_S37_MATCHED_QUERY_GATED_DEV_COMPLETE`

No second DEV.
No post-DEV query normalization / scale / bias / nonlinearity / rank tuning.
No projected/native mixing.
No sealed confirmation.
No multilingual probe.
No Laya/Jev evaluation.

## Frozen matched setup

Both arms:
- exact S35 native 256D relation representation/signature
- exact S17 primary path/fusion/optimizer shell
- A13 LoRA **16,384**
- shared primary projection **32,768**
- original A13 frozen
- HIRACore frozen
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S37 domains
- K=4
- seed **58001**
- 24 epochs
- batch 16
- AdamW 2e-4 / wd .01 / clip 1.0
- identical rows/order
- independent runtime/optimizer state

Control:
- exact native relation logits/signatures
- trainable surface **49,152**

Treatment:
- masked-mean native question summary
- L2 normalize eps **1e-12**
- `feature_k = signature_k * query_summary`
- `residual_k = feature_k @ w`
- one shared zero-init `w in R^256`
- no bias/nonlinearity/rank expansion/learned scale
- added params **256**
- trainable surface **49,408**

## A0 mechanism result

A0 cleanly isolated query conditioning:
- zero-init relation/signature/primary/fused differences **0**
- selected-choice identity **1.0**
- query intervention residual change **0.0220453572**
- question-token permutation logit error **7.4505806e-9**
- masked query-padding error **0**
- logical-option permutation logit error **5.9604645e-8**
- readout relation gradient L1 **0.0012450286**
- readout primary gradient L1 **0**
- native relation -> projection gradient L1 **0**
- primary -> projection gradient L1 **969.4917602539**
- relation -> LoRA gradient L1 **0.2209856110**
- native relation remains projection-independent
- arbitrary K=3/K=7 PASS
- checkpoint/probability/full-K mechanics PASS

DEV outcome is therefore not a wiring failure.

## Selected DEV — control

Selected epoch: **21**
Checkpoint:
`d738a1ae68c34846d89f1f249e49e11382035b5952714df222db2edfab36fc87`

Fused:
- canonical **0.5208333333**
- paraphrase **0.4348958333**
- paired both-correct **0.234375**
- question-swap **0.53125**
- cross-view agreement **0.6276041667**
- JS **0.0315472766**
- canonical margin **+0.0018336633**
- paraphrase margin **-0.1845829493**

Primary:
- canonical **0.4583333333**
- paraphrase **0.46875**

Relation:
- canonical **0.3411458333**
- paraphrase **0.28125**
- canonical margin **-0.0383368110**
- paraphrase margin **-0.0467969875**
- agreement **0.5911458333**

Signatures:
- same-option cosine **0.8558237304**
- same-vs-strongest-other margin **0.1971760833**

Gate:
- PASS **15/23**
- FAIL **8/23**
- DEV_READY false

## Selected DEV — query-gated treatment

Selected epoch: **20**
Checkpoint:
`8f5b4da3ab68c0d316c2271a73cfc60c0366684ef2cd8e6ed774e31c8e5d83c0`

Fused:
- canonical **0.5182291667**
- paraphrase **0.4244791667**
- paired both-correct **0.2395833333**
- question-swap **0.484375**
- cross-view agreement **0.5755208333**
- JS **0.0355339245**
- canonical margin **-0.0605559849**
- paraphrase margin **-0.1832927659**

Primary:
- canonical **0.4322916667**
- paraphrase **0.4140625**

Relation:
- canonical **0.34375**
- paraphrase **0.2916666667**
- canonical margin **-0.0352677380**
- paraphrase margin **-0.0445126866**
- agreement **0.6015625**

Signatures:
- same-option cosine **0.8502296259**
- same-vs-strongest-other margin **0.1887374539**

Gate:
- PASS **15/23**
- FAIL **8/23**
- DEV_READY false

## Exact selected delta — treatment minus control

Direct relation correctness changes are tiny:
- relation canonical accuracy **+0.0026041667**
- relation paraphrase accuracy **+0.0104166667**
- relation canonical margin **+0.0030690730**
- relation paraphrase margin **+0.0022843008**
- relation agreement **+0.0104166667**

Fused/primary endpoints do not show coherent gain:
- fused canonical **-0.0026041667**
- fused paraphrase **-0.0104166667**
- paired **+0.0052083333**
- question-swap **-0.046875**
- fused agreement **-0.0520833333**
- fused JS **+0.0039866479** (worse)
- fused canonical margin **-0.0623896482**
- fused paraphrase margin **+0.0012901834**
- raw primary canonical **-0.0260416667**
- raw primary paraphrase **-0.0546875**

Native transport is slightly worse:
- signature cosine **-0.0055941045**
- signature discrimination margin **-0.0084386294**

## Preregistered interpretation

S37 matches **Case C — little/no direct correctness gain**.

Therefore:

> Making the S36 readout query-dependent through a single diagonal interaction `signature * query_summary` does not expose a useful correctness map.

The result localizes the residual further:
- a single global direction is insufficient (S36);
- making that direction depend on the query only through coordinate-wise multiplication is also insufficient (S37).

This does **not** show query conditioning is irrelevant.
It shows the missing interaction is not well represented by one diagonal 256D bilinear form.

## S37 closure

Close the S37 diagonal query-gated family on exposed DEV.

Do not:
- tune query normalization
- alter residual scale
- add bias/nonlinearity on S37 rows
- retry rank/multiple vectors
- change readout LR
- mix projected/native paths
- rerun seed/LR/epoch/batch
- weaken gates
- run a second DEV

## Next controlled direction

**S38 — Full Bilinear Native Query-Signature Correctness Readout**

Scientific question:

> Does correctness require cross-coordinate interactions between the native option signature and native query summary?

Control:
- exact S35 native relation logits/signatures.

Treatment:
- same masked-mean + L2-normalized 256D native query summary;
- one shared zero-initialized matrix `W in R^(256x256)`;
- residual `s_k^T W q`;
- no bias, MLP, nonlinearity, option/domain/K-specific state;
- residual scale 1.0;
- added params exactly **65,536**;
- treatment total trainable surface **114,688**.

Why this is the next clean test:
- S37 tested only the diagonal of a bilinear map;
- S38 exposes off-diagonal cross-coordinate interactions directly;
- zero-init still gives exact control identity;
- every matrix entry receives a first-order gradient without asymmetric factor initialization;
- arbitrary-K and option-permutation equivariance remain intact.

Production-ready remains false.
Laya/Jev parity remains unestablished.
