# HIRA V1 S38 closure — Full Bilinear Native Query-Signature Correctness Readout

Status: **CLOSED — MATCHED DEV COMPLETE / PREREGISTERED CASE B / CORRECTNESS GAIN WITH TRANSPORT-FUSION REGRESSION**

Issue: #259
PR: #260

## Canonical authority

A0:
- run `37084720616`
- artifact `11259982918`
- digest `sha256:d6e6cfc05cdc65ab5486979a4bd69775a41fd6ff4c6debabdc35c0d0d528d03b`
- authority head `2d0fc9bc26fe3610e9c40bc55dc2986f91a207a2`
- outcome `HIRA_V1_S38_A0_FULL_BILINEAR_READOUT_READY`

Fresh matched TRAIN/DEV:
- run `37085497287`
- artifact `11261486050`
- artifact digest `sha256:0c717b73dd75e94beca43bede784ba22cb06b291fbd8ff57d6a0adcc24ddadf4`
- scientific head `19672166b808f23eaac88d324b2fba2252ba31a5`
- outcome `HIRA_V1_S38_MATCHED_FULL_BILINEAR_DEV_COMPLETE`

No second DEV.
No post-DEV W regularization / factorization / rank / scale / bias / nonlinearity tuning.
No W-only LR/scheduler.
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
- 12 wholly fresh S38 domains
- K=4
- seed **59001**
- 24 epochs
- batch 16
- AdamW 2e-4 / wd .01 / clip 1.0
- identical rows/order
- independent runtime/optimizer state

Control:
- exact native relation logits/signatures
- trainable surface **49,152**

Treatment:
- exact S37 masked-mean + L2-normalized 256D native query summary
- one shared zero-init `W in R^(256x256)`
- residual `signature_k^T W query`
- no bias/nonlinearity/factorization/learned scale
- added params **65,536**
- trainable surface **114,688**

## A0 mechanism result

A0 cleanly proved a genuinely broader cross-coordinate family:
- zero-init relation/signature/primary/fused differences **0**
- selected-choice identity **1.0**
- relation -> full W gradient L1 **0.3336779475**
- off-diagonal W gradient L1 **0.3323763907**
- primary -> W gradient L1 **0**
- native relation -> projection direct gradient L1 **0**
- primary -> projection gradient L1 **1517.2321777344**
- relation -> LoRA gradient L1 **0.2464779774**
- query intervention residual change **0.0210716370**
- signature intervention residual change **0.0132539095**
- question-token permutation logit error **2.9802322e-8**
- masked query-padding error **0**
- logical-option permutation logit error **5.9604645e-8**
- arbitrary K=3/K=7 PASS
- checkpoint/probability/full-K mechanics PASS

The DEV result is therefore not a wiring failure.

## Selected DEV — control

Selected epoch: **17**
Checkpoint:
`d33ec7a4f45192ec6178c33935a70ad75907a4eaea51525277ed544e06fd1da7`

Fused:
- canonical **0.4635416667**
- paraphrase **0.4739583333**
- paired both-correct **0.171875**
- question-swap **0.4270833333**
- cross-view agreement **0.5963541667**
- JS **0.0378480734**
- canonical margin **-0.0381862478**
- paraphrase margin **-0.1112954083**

Primary:
- canonical **0.4895833333**
- paraphrase **0.5026041667**

Relation:
- canonical **0.3567708333**
- paraphrase **0.3125**
- canonical margin **-0.0482126276**
- paraphrase margin **-0.0507723217**
- agreement **0.5234375**

Signatures:
- same-option cosine **0.8359404455**
- same-vs-strongest-other margin **0.2052292128**

Gate:
- PASS **15/23**
- DEV_READY false

## Selected DEV — full bilinear treatment

Selected epoch: **16**
Checkpoint:
`5f35d9082e3e0ed318632395d4e548b3f18da85b89e8a022d9c7bffa22812e78`

Fused:
- canonical **0.5286458333**
- paraphrase **0.5104166667**
- paired both-correct **0.265625**
- question-swap **0.7395833333**
- cross-view agreement **0.4765625**
- JS **0.0893449590**
- canonical margin **-0.1975861347**
- paraphrase margin **-0.2084601584**

Primary:
- canonical **0.4895833333**
- paraphrase **0.5208333333**

Relation:
- canonical **0.4791666667**
- paraphrase **0.4869791667**
- canonical margin **-0.3075006741**
- paraphrase margin **-0.0764008686**
- agreement **0.4505208333**

Signatures:
- same-option cosine **0.8625788788**
- same-vs-strongest-other margin **0.0723221811**

Gate:
- PASS **13/23**
- DEV_READY false

## Exact selected delta — treatment minus control

Correctness improved materially:
- relation canonical accuracy **+0.1223958333**
- relation paraphrase accuracy **+0.1744791667**
- fused canonical accuracy **+0.0651041667**
- fused paraphrase accuracy **+0.0364583333**
- paired both-correct **+0.09375**
- question-swap **+0.3125**
- raw primary paraphrase **+0.0182291667**
- raw primary canonical **0**

But transport/calibration regressed materially:
- relation canonical margin **-0.2592880465**
- relation paraphrase margin **-0.0256285469**
- relation agreement **-0.0729166667**
- fused agreement **-0.1197916667**
- fused JS **+0.0514968857** (worse)
- fused canonical margin **-0.1593998869**
- fused paraphrase margin **-0.0971647501**
- signature discrimination margin **-0.1329070317**

Same-option signature cosine rises **+0.0266384333**, but that does not offset the collapse in discrimination margin and agreement.

## Preregistered interpretation

S38 matches **Case B — direct correctness improves but transport/fusion materially regresses**.

The result establishes an important positive fact:

> Native signature/query geometry contains useful cross-coordinate correctness information that S36/S37 could not extract.

However, the unrestricted jointly-trained bilinear correction distorts the optimization balance:
- accuracy rises;
- relation/fused margins become more negative;
- cross-view agreement worsens;
- fused JS crosses the frozen <=0.05 gate;
- signature discrimination collapses.

Therefore the full bilinear hypothesis is informative but not acceptable as a DEV_READY candidate.

## S38 closure

Close unrestricted jointly-coupled full bilinear readout on exposed S38 DEV.

Do not:
- tune W regularization on S38 rows
- factorize or change rank on S38 rows
- change residual scale/bias/nonlinearity
- give W a separate LR/scheduler
- mix projected/native paths
- rerun seed/LR/epoch/batch
- weaken selector/gates
- run a second DEV

## Next controlled direction

**S39 — Gradient-Isolated Full Bilinear Correctness Readout**

Scientific question:

> Can S38's cross-coordinate correctness gain be retained when the correction head is prevented from steering native representation geometry?

Keep:
- exact same 256x256 full bilinear readout capacity (**65,536 params**);
- exact same native query summary;
- exact native relation representation;
- exact S17 primary/fusion shell.

Change the optimization graph, not capacity:
- native/base relation objective continues to train LoRA exactly through the native relation operator;
- bilinear correction objective sees **detached native signatures and detached native query summary**;
- bilinear correction gradients update **W only**;
- bilinear correction contributes no direct gradient to A13 LoRA, native relation geometry, shared projection, or HIRACore;
- evaluation still uses native logits + bilinear residual.

This directly tests whether S38's Case-B failure is caused by correctness-head co-adaptation damaging transport geometry.

Production-ready remains false.
Laya/Jev parity remains unestablished.
