# HIRA V1 S38 handoff — to S39 Gradient-Isolated Full Bilinear Readout

S38 is frozen as:

`HIRA_V1_S38_MATCHED_FULL_BILINEAR_DEV_COMPLETE`

Preregistered interpretation:
**Case B — strong correctness gain with material transport/fusion regression.**

## Canonical evidence

A0:
- run `37084720616`
- artifact `11259982918`
- digest `sha256:d6e6cfc05cdc65ab5486979a4bd69775a41fd6ff4c6debabdc35c0d0d528d03b`

Fresh matched DEV:
- run `37085497287`
- artifact `11261486050`
- digest `sha256:0c717b73dd75e94beca43bede784ba22cb06b291fbd8ff57d6a0adcc24ddadf4`
- scientific head `19672166b808f23eaac88d324b2fba2252ba31a5`

Control selected epoch 17:
- fused canonical/paraphrase **0.4635417 / 0.4739583**
- relation canonical/paraphrase **0.3567708 / 0.3125**
- relation agreement **0.5234375**
- signature cosine/margin **0.8359404 / 0.2052292**
- gates **15/23**

Treatment selected epoch 16:
- fused canonical/paraphrase **0.5286458 / 0.5104167**
- relation canonical/paraphrase **0.4791667 / 0.4869792**
- relation agreement **0.4505208**
- signature cosine/margin **0.8625789 / 0.0723222**
- gates **13/23**

Key deltas:
- relation canonical **+0.1223958**
- relation paraphrase **+0.1744792**
- fused canonical **+0.0651042**
- fused paraphrase **+0.0364583**
- question-swap **+0.3125**
- fused agreement **-0.1197917**
- fused JS **+0.0514969**
- relation agreement **-0.0729167**
- signature discrimination margin **-0.1329070**
- canonical relation margin **-0.2592880**

## Residual

S38 shows the missing correctness information is present in cross-coordinate native query-signature interactions.

The failure is no longer "cannot learn correctness".
The problem is now **correctness-vs-transport interference**.

The next stage must preserve the full bilinear expressivity while changing gradient ownership.

## S39 hypothesis

**Gradient-Isolated Full Bilinear Correctness Readout**

Shared base:
- exact S35 native 256D relation representation/signature
- exact S17 primary/fusion shell
- exact S38 native query summary
- no projected relation path

Control:
- exact native relation logits.

Treatment evaluation:
- `residual_k = signature_k^T W query`
- `logit_k = native_logit_k + residual_k`
- `W in R^(256x256)`
- added params **65,536**
- total surface **114,688**

Treatment training isolation:
1. native/base relation losses train the normal native relation path and LoRA;
2. correction loss computes bilinear residual from **detached signature/query features**;
3. correction-loss gradients may update **W only**;
4. correction-loss gradient into A13/LoRA, native relation geometry, shared projection and HIRACore must be exactly zero;
5. primary block remains unable to update W.

Do not reduce rank or change capacity.
Do not add regularization/MLP/bias/learned scale.

## Required S39-A0

Must prove:
- exact zero-init identity
- W params exactly 65,536 / treatment total 114,688
- native control relation gradient ownership unchanged
- correction loss -> W gradient nonzero, including off-diagonal entries
- correction loss -> A13 LoRA gradient exactly zero
- correction loss -> projection gradient exactly zero
- correction loss -> HIRACore gradient exactly zero
- primary block -> W gradient exactly zero
- detached feature tensors have no correction-gradient path upstream
- nonzero-W query/signature interventions remain live
- logical-option permutation equivariance
- question permutation / masked padding invariance
- arbitrary K
- projection independence
- checkpoint/probability/full-K/state-once mechanics

Use wholly fresh S39 authority.
No S38 rows.
One DEV only.
No post-DEV gradient-mixing coefficient tuning.
No Laya/Jev benchmark before a separately confirmed DEV_READY candidate.
