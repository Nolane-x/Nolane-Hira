# HIRA V1 S40 contract — Reference-Anchored Projected Joint Bilinear Co-Adaptation

Status: **OPEN / PREREGISTERED BEFORE S40-A0 EXPOSURE**

Issue: #263

Parent:
- S39 PR #262
- merged main `e197256fdc5dd74fc789cdfdf77ff2f1c7863042`
- S39 outcome `HIRA_V1_S39_MATCHED_GRADIENT_ISOLATED_DEV_COMPLETE`
- preregistered S39 interpretation: **Case C**

## 1. Scientific question

S38 showed that unrestricted full-bilinear co-adaptation can recover material correctness but damages transport geometry.

S39 showed that hard isolation preserves native transport but loses most S38 correctness gain.

S40 asks:

> Can joint co-adaptation retain S38 correctness while a parameter-free matched-reference constraint prevents first-order increase of native signature drift?

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

## 3. Reference arm

Reference is an exact native S35/S17 runtime:
- same initialization seed
- same TRAIN rows
- same batch order
- exact native primary/relation objectives
- exact S17 norm-balanced runtime update
- no W
- independent runtime and optimizer state

Trainable surface:
**49,152**.

Reference native signatures used by the anchor are always detached.

## 4. Treatment arm

Treatment keeps exact S38 full-bilinear expressivity:

`residual_k = signature_k^T W q_hat`

`logit_k = native_logit_k + residual_k`

where:
- `W in R^(256x256)`
- exact zero initialization
- no bias
- no nonlinearity
- no factorization
- no rank constraint
- no learned residual scale
- residual scale **1.0**
- no option/domain/K-specific parameters

Treatment runtime and W jointly co-adapt through the exact S38 treatment loss.

Added W params:
**65,536**.

Treatment total trainable surface:
**114,688**.

## 5. Reference signature anchor

For the same batch and same semantic views:

`A_sig = mean(1 - cosine(s_treatment, stopgrad(s_reference)))`

over all valid native option signatures.

Rules:
- reference signatures are detached;
- no anchor gradient may reach reference runtime;
- W is excluded from anchor targets;
- shared primary projection is not part of native signature geometry and receives no direct anchor gradient;
- no learned anchor parameters;
- no anchor coefficient.

## 6. Parameter-free projected joint update

Let:
- `g` = exact S38 joint treatment gradient after S17 primary/relation norm balancing;
- `g_runtime` = treatment runtime slice of `g`;
- `g_W` = W slice of `g`;
- `a` = gradient of `A_sig` with respect to treatment runtime parameters.

The projected runtime gradient is:

If `a dot g_runtime >= 0`:
- `g_runtime' = g_runtime`

If `a dot g_runtime < 0`:
- `g_runtime' = g_runtime - ((a dot g_runtime) / (||a||^2 + 1e-12)) * a`

W gradient is always:
- `g_W' = g_W`

Interpretation under gradient descent:
- first-order anchor change is proportional to `-a dot g_runtime'`;
- the projection therefore prevents a conflicting joint step from increasing anchor drift to first order;
- no tunable mixing coefficient exists.

After projection:
- concatenate `g_runtime'` and unchanged `g_W`;
- apply the same treatment grad clip **1.0** over the full treatment parameter set;
- AdamW lr **2e-4**, weight decay **0.01**;
- no W-only optimizer/schedule.

## 7. Required S40-A0

Fresh S40-A0 only.

Must prove:
- zero-init reference/treatment native relation identity
- zero-init signature identity
- zero-init treatment fused identity with W=0
- selected-choice identity 1.0
- W params exactly **65,536**
- reference/treatment surfaces **49,152 / 114,688**
- reference-signature detach exact
- anchor -> reference runtime gradient exactly **0**
- anchor -> W gradient exactly **0**
- synthetic treatment signature drift produces anchor >0
- anchor -> treatment LoRA gradient finite and nonzero under synthetic drift
- exact S38 joint correctness gradient -> W finite/nonzero
- off-diagonal W correctness gradient finite/nonzero
- exact S38 joint correctness gradient -> treatment LoRA finite/nonzero
- conflicting synthetic gradient has pre-projection anchor dot <0
- post-projection anchor dot approximately 0
- non-conflicting synthetic gradient is exact identity
- W gradient bitwise unchanged by projection
- sufficiently small projected synthetic step does not increase anchor
- logical-option permutation equivariance
- question-token permutation invariance
- masked query-padding invariance
- arbitrary K at least K=3/K=7
- native projection independence
- checkpoint/probability/full-K/state-once mechanics PASS

A0 semantics are diagnostic only.

## 8. Fresh matched TRAIN/DEV

Only after:
1. qualified A0
2. frozen A0 receipt
3. frozen interpretation plan
4. frozen trainer/workflow
5. exact-head generic CI
6. separate one-shot TRAIN/DEV marker

Intended authority:
- seed **61001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S40 domains
- K=4
- two state views
- two question wording views
- two option views
- epochs **24**
- batch 16
- AdamW lr **2e-4**
- weight decay **0.01**
- grad clip **1.0**
- identical rows/order for reference and treatment
- independent runtime/optimizer state

Freshness:
- exact S0-S39 exposed rows excluded
- S40-A0 rows excluded
- M5 final/confirmatory rows excluded
- W29-W34 sealed rows excluded

## 9. Selector / gates

Use exact existing S35/S38 selector independently per arm.
Keep existing DEV_READY gates unchanged.

Treatment additionally requires:
- W capacity exact
- reference batch/order identity
- no reference-gradient leakage
- projection rule active exactly as frozen

## 10. Frozen interpretation

A — correctness retained + transport protected:
S38-like correctness gains remain material while agreement/signature discrimination avoid S38 collapse.
Then S40 is viable for a wholly fresh confirmation track.

B — correctness retained but transport still collapses:
reference-signature projection is insufficient; close S40.

C — transport protected but correctness collapses toward S39:
the anchor removes a representation component required for correctness; close this family.

D — treatment DEV_READY:
freeze immediately; no second S40 DEV; separate fresh confirmation required before external matched evaluation.

## 11. Stop rule

After one S40 DEV:
- no projection slack/margin
- no anchor coefficient
- no changing anchor target
- no partial detach
- no gradient-mixing coefficient
- no W-only LR/scheduler
- no rank/factorization retry
- no residual-scale/bias/nonlinearity retry
- no projected/native mixing
- no seed/LR/epoch/batch retry
- no selector/gate weakening
- no second DEV

Scientific FAIL is valid.
No Laya/Jev benchmark before a separately confirmed DEV_READY candidate.
