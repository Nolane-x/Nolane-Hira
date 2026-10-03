# HIRA V1 S41 contract — Optimizer-Step-Anchored Joint Bilinear Co-Adaptation

Status: **OPEN / PREREGISTERED BEFORE S41-A0 EXPOSURE**

Issue: #265

Parent:
- S40 PR #264
- merged main `83d42cde0f535c7ca9ca644eede5a41aa3cf47dd`
- S40 closed pre-DEV on optimizer-faithfulness invalidation
- no S40 TRAIN/DEV exposure

## 1. Scientific question

S38 showed that unrestricted full-bilinear co-adaptation can recover material correctness but damages transport geometry.

S39 showed that hard isolation preserves transport but loses most S38 correctness gain.

S40 introduced a matched-reference native-signature anchor, but projected the **raw gradient** before AdamW. Because AdamW applies stateful moment/preconditioning and decoupled weight decay, that raw-gradient projection did not guarantee the actual parameter movement was anchor-safe.

S41 asks:

> Can joint co-adaptation retain S38 correctness while the **actual stateful AdamW runtime parameter delta** is prevented from increasing matched-reference native-signature drift to first order?

## 2. Frozen shared shell

Reference and treatment use:
- exact S35 native 256D relation representation/signature
- exact S17 primary/fusion shell
- final A13 attention LoRA **16,384**
- shared primary projection **32,768**
- original A13 frozen
- HIRACore frozen
- exact S38/S39 masked-mean + L2 native query summary
- no projected relation path
- state-once
- full-K
- opaque option IDs
- exact existing selector/gates

Reference trainable surface:
**49,152**.

Treatment:
- runtime **49,152**
- W **65,536**
- total **114,688**.

## 3. Reference arm

Reference:
- exact native S35/S17 objective
- same initialization seed
- same TRAIN rows/order
- no W
- independent runtime/optimizer state
- standard AdamW with the frozen S41 optimizer semantics below

Reference native signatures used by the anchor are detached.

## 4. Treatment correctness family

Treatment keeps exact S38 full-bilinear expressivity:

`residual_k = signature_k^T W q_hat`

where:
- W shape **256x256**
- W params **65,536**
- zero initialization
- no bias
- no nonlinearity
- no factorization/rank constraint
- no learned residual scale
- residual scale **1.0**
- no option/domain/K-specific params

Joint correctness gradients may reach treatment runtime and W.

## 5. Signature anchor

For matched batch/views:

`A_sig = mean(1 - cosine(s_treatment, stopgrad(s_reference)))`

Rules:
- reference side fully detached
- W excluded from anchor target
- shared primary projection has no direct native-signature anchor path
- no learned anchor params
- no anchor coefficient

Let:
`a = grad(A_sig)`
with respect to treatment runtime params only.

## 6. Frozen optimizer semantics

S41 explicitly fixes AdamW to:
- lr **2e-4**
- betas **(0.9, 0.999)**
- eps **1e-8**
- weight decay **0.01**
- amsgrad **false**
- maximize **false**
- foreach **false**
- fused **false**
- capturable **false**
- differentiable **false**

Gradient clipping:
- exact inherited global L2 grad clip **1.0**
- applied before AdamW moment/state update

All treatment trainable params, including W, participate in the same clipping norm before the candidate AdamW step is computed.

No W-only optimizer or schedule.

## 7. Exact AdamW candidate delta

For every treatment parameter and its clipped gradient:
- advance the standard AdamW first/second moments and step counter;
- include bias correction;
- include coordinate-wise preconditioning;
- include decoupled weight decay;
- obtain the exact candidate parameter value that frozen PyTorch AdamW would produce;
- define `delta = candidate - current_parameter`.

The S41 step engine must reproduce standard PyTorch AdamW under these frozen semantics on an unprojected probe to tight numerical identity.

The next AdamW state is derived from the original clipped raw gradient, not from the projected parameter delta.

## 8. Optimizer-step anchor projection

Split candidate treatment deltas into:
- runtime candidate delta `delta_r`
- W candidate delta `delta_W`

Compute `a` on treatment runtime params.

Actual first-order anchor change is:
`a dot delta_r`.

If:
`a dot delta_r <= 0`
then:
`delta_r' = delta_r`

If:
`a dot delta_r > 0`
then:
`delta_r' = delta_r - ((a dot delta_r)/(||a||^2 + 1e-12)) a`

W:
`delta_W' = delta_W`

Apply:
- runtime params += `delta_r'`
- W += `delta_W`

Persist the AdamW moment/step state computed from the frozen clipped raw gradients.

No projection slack/margin.
No anchor coefficient.
No partial detach.
No gradient-mixing coefficient.

## 9. Required S41-A0

Fresh S41-A0 diagnostic authority only.

Must prove:
- zero-init reference/treatment native relation identity
- zero-init signature identity
- zero-init treatment fused identity with W=0
- selected-choice identity 1.0
- W params exactly 65,536
- reference/treatment surfaces 49,152 / 114,688
- reference-signature detach exact
- anchor -> reference runtime gradient exactly 0
- anchor -> W gradient exactly 0
- synthetic treatment signature drift produces anchor >0
- anchor -> treatment LoRA gradient finite/nonzero
- exact joint correctness gradient -> W finite/nonzero
- off-diagonal W gradient finite/nonzero
- exact joint correctness gradient -> treatment LoRA finite/nonzero
- candidate AdamW parameter values/deltas match standard PyTorch AdamW on an unprojected probe
- next AdamW first moments match standard PyTorch AdamW
- next AdamW second moments match standard PyTorch AdamW
- step counters match
- decoupled weight decay is included
- conflicting actual runtime candidate delta has `a dot delta_r > 0`
- projected actual runtime delta has post-dot approximately 0
- non-conflicting actual runtime delta is exact identity
- W candidate delta is unchanged by projection
- actually applied runtime movement equals projected candidate delta
- actually applied W movement equals unprojected candidate W delta
- sufficiently small optimizer-faithful projected update does not increase anchor
- logical-option permutation equivariance
- question-token permutation invariance
- masked query-padding invariance
- arbitrary K at least K=3/K=7
- native projection independence
- checkpoint/probability/full-K/state-once mechanics PASS

A0 semantic values are diagnostic only.

## 10. Fresh matched TRAIN/DEV

Only after:
1. qualified S41-A0
2. frozen A0 receipt
3. frozen interpretation plan
4. frozen trainer/workflow
5. exact-head generic CI
6. separate one-shot TRAIN/DEV marker

Intended authority:
- seed **62001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S41 domains
- K=4
- two state views
- two question wording views
- two option views
- 24 epochs
- batch 16
- frozen AdamW semantics above
- grad clip 1.0
- identical rows/order reference vs treatment
- independent runtime/optimizer state

Freshness:
- exact S0-S40 exposed rows excluded
- S41-A0 rows excluded
- M5 final/confirmatory rows excluded
- W29-W34 sealed rows excluded

## 11. Selector / gates

Use exact existing S35/S38 selector independently per arm.
Keep DEV_READY gates unchanged.

Treatment additionally requires:
- W capacity exact
- optimizer-step engine parity qualified
- reference batch/order identity
- no reference-gradient leakage
- W delta never projected

## 12. Frozen interpretation

A — correctness retained + transport protected:
material S38-like correctness gain while avoiding S38 agreement/signature collapse.
Then S41 is viable for wholly fresh confirmation.

B — correctness retained but transport still collapses:
optimizer-step anchoring is insufficient; close S41.

C — transport protected but correctness collapses toward S39:
anchor-conflicting actual movement is necessary for S38 correctness; close this family.

D — treatment DEV_READY:
freeze immediately; no second S41 DEV; separate fresh confirmation before external matched evaluation.

## 13. Stop rule

After one S41 DEV:
- no projection slack/margin
- no anchor coefficient
- no anchor-target change
- no optimizer-state reinterpretation
- no partial detach
- no gradient mixing
- no W-only LR/scheduler
- no rank/factorization retry
- no residual-scale/bias/nonlinearity retry
- no projected/native mixing
- no seed/LR/epoch/batch retry
- no selector/gate weakening
- no second DEV

Scientific FAIL is valid.
No Laya/Jev benchmark before a separately confirmed DEV_READY candidate.
