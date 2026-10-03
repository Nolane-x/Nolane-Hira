# HIRA V1 S39 closure — Gradient-Isolated Full Bilinear Correctness Readout

Status: **CLOSED — MATCHED DEV COMPLETE / PREREGISTERED CASE C / TRANSPORT RECOVERS BUT CORRECTNESS GAIN LARGELY DISAPPEARS**

Issue: #261
PR: #262

## Canonical authority

A0:
- run `37089648986`
- artifact `11261657743`
- digest `sha256:afc20e60cabac8de5db33eeee15574207d7e0592547a1787e4980cc0b81f6e20`
- authority head `6f1eb465cfcefc7bc368218378fc144ea3f6c347`
- outcome `HIRA_V1_S39_A0_GRADIENT_ISOLATED_BILINEAR_READY`

Fresh matched TRAIN/DEV:
- run `37092496411`
- artifact `11263399222`
- artifact digest `sha256:7c11deed7760bddbb559aec5e608469e8fb6e2787b2e7d22b416b5f394311574`
- scientific head `2b8b789ad2729f7ee185f83185e54fffeb29ad83`
- outcome `HIRA_V1_S39_MATCHED_GRADIENT_ISOLATED_DEV_COMPLETE`
- seed **60001**

No second DEV.
No post-DEV partial detach / gradient mixing.
No W-only LR/scheduler.
No rank/factorization retry.
No residual scale/bias/nonlinearity retry.
No projected/native mixing.
No sealed confirmation.
No multilingual probe.
No Laya/Jev evaluation.

## Pre-DEV mechanical abort

Run `37091887607` failed at step 7 `py_compile` before:
- canonical A0 verification
- M4 integrity
- fresh TRAIN/DEV step 10

Step 10 was skipped.
No DEV metric or artifact was exposed.

The repaired stack passed exact-head CI `37092044550` on Python 3.10 and 3.12 before the successful scientific re-arm.

## Frozen matched setup

Both arms:
- exact S35 native 256D relation representation/signature
- exact S17 primary/fusion shell
- A13 LoRA **16,384**
- shared primary projection **32,768**
- original A13 frozen
- HIRACore frozen
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S39 domains
- K=4
- 24 epochs
- batch 16
- AdamW 2e-4 / wd .01
- identical rows/order across arms
- independent optimizer state

Control:
- exact native S35/S17 runtime training
- trainable surface **49,152**

Treatment:
- same native runtime training as control
- same full bilinear W as S38: **256x256 / 65,536 params**
- correction CE sees detached native logits/signatures/query
- correction gradient updates W only
- runtime and W clipped separately at 1.0
- total trainable surface **114,688**

## A0 mechanism result

A0 proved hard isolation:
- correction -> W gradient L1 **0.3294754326**
- correction -> off-diagonal W gradient L1 **0.3281979561**
- correction -> LoRA **0**
- correction -> projection **0**
- correction -> HIRACore **0**
- primary -> W **0**
- native relation -> W **0**
- matched runtime gradient max abs **0**
- matched synthetic runtime update max abs **0**
- W update max abs **0.0001999941**

Therefore any S39 treatment/control native-runtime difference would be a contract failure, not an intended effect.

## Runtime trajectory invariant

All 24 epochs:
- runtime-state fingerprints equal: **true**
- LoRA max abs difference: **0**
- projection max abs difference: **0**
- native pre-W logit max abs difference: **0**
- native signature max abs difference: **0**

The selected treatment/control epoch is also the same: **7**.

Therefore the reported selected delta is also a pure W effect on an identical native runtime state.

## Selected DEV — control

Selected epoch: **7**
Checkpoint:
`d0eacabbce8aa345b364de669032396d3bf028d891df34abed76d41479ff116d`

Fused:
- canonical **0.5260416667**
- paraphrase **0.4166666667**
- paired both-correct **0.1979166667**
- question-swap **0.5052083333**
- cross-view agreement **0.4583333333**
- JS **0.0415225637**
- canonical margin **+0.0694033681**
- paraphrase margin **-0.1253143274**

Primary:
- canonical **0.5130208333**
- paraphrase **0.4505208333**

Relation:
- canonical **0.3515625**
- paraphrase **0.3203125**
- canonical margin **-0.0402195677**
- paraphrase margin **-0.0377369101**
- agreement **0.5286458333**

Signatures:
- same-option cosine **0.8556269805**
- discrimination margin **0.0184459677**

Gate:
- PASS **14/23**
- DEV_READY false

## Selected DEV — gradient-isolated treatment

Selected epoch: **7**
Checkpoint:
`4b87af26096469407512204ad0033cd29882d6e199f8870cd2ab5248222ce327`

Fused:
- canonical **0.5338541667**
- paraphrase **0.3958333333**
- paired both-correct **0.171875**
- question-swap **0.46875**
- cross-view agreement **0.5598958333**
- JS **0.0494064926**
- canonical margin **+0.0709558992**
- paraphrase margin **-0.1813010946**

Primary:
- canonical **0.5130208333**
- paraphrase **0.4505208333**

Relation:
- canonical **0.4244791667**
- paraphrase **0.3307291667**
- canonical margin **-0.0394929176**
- paraphrase margin **-0.0658865968**
- agreement **0.5859375**

Signatures:
- same-option cosine **0.8556269805**
- discrimination margin **0.0184459677**

Gate:
- PASS **15/24**
- DEV_READY false

## Exact pure-W delta — treatment minus same-epoch control

Correctness:
- relation canonical **+0.0729166667**
- relation paraphrase **+0.0104166667**
- fused canonical **+0.0078125**
- fused paraphrase **-0.0208333333**
- paired **-0.0260416667**
- question-swap **-0.0364583333**

Transport/stability:
- relation agreement **+0.0572916667**
- fused agreement **+0.1015625**
- fused JS **+0.0078839289** (slightly worse)
- same-option signature cosine **0**
- signature discrimination margin **0**

Margins:
- relation canonical **+0.0007266502**
- relation paraphrase **-0.0281496868**
- fused canonical **+0.0015525311**
- fused paraphrase **-0.0559867672**

Primary:
- canonical **0**
- paraphrase **0**

## Relation to S38

S38 unrestricted co-adaptation had:
- relation canonical **+0.1223958333**
- relation paraphrase **+0.1744791667**
- fused canonical **+0.0651041667**
- fused paraphrase **+0.0364583333**
- but fused agreement **-0.1197916667**
- relation agreement **-0.0729166667**
- signature discrimination margin **-0.1329070317**
- fused JS **+0.0514968857**

S39 hard isolation removes the representation drift:
- signature deltas become exactly **0**
- relation agreement becomes positive
- fused agreement becomes strongly positive

But most S38 correctness gain disappears, especially paraphrase and fused endpoints.

## Preregistered interpretation

S39 matches **Case C — transport recovers but correctness gain largely disappears**.

Therefore:

> S38's strongest correctness gain depended on representation co-adaptation. A W-only detached correction adapter is not sufficient.

This closes the hard-isolated full-bilinear family on exposed S39 DEV.

## S39 closure

Do not:
- tune detach coefficient
- add gradient mixing coefficient
- alter W LR/scheduler
- change rank/factorization
- alter scale/bias/nonlinearity
- mix projected/native paths
- retry seed/LR/epoch/batch
- weaken gates
- run second DEV

Production-ready remains false.
Laya/Jev parity remains unestablished.

## Next controlled direction

**S40 — Reference-Anchored Joint Bilinear Co-Adaptation**

Scientific question:

> Can S38's co-adaptive correctness gain be retained while explicitly constraining native representation drift relative to a matched reference trajectory?

This changes the optimization constraint, not readout capacity.

Treatment:
- exact S38 full bilinear W capacity: **65,536**
- joint runtime + W co-adaptation remains enabled
- a matched reference runtime, initialized identically and trained on the exact native S35/S17 objective, provides detached reference native logits/signatures on the same rows
- treatment native representation is constrained toward the matched reference geometry
- W remains evaluated as `signature^T W query`
- no partial detach
- no gradient-mixing coefficient
- no W-only LR
- no rank/factorization
- no extra learned adapter capacity

The S40 mechanism must be preregistered and mechanically validated on fresh A0 before any fresh DEV.
