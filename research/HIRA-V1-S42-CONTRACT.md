# HIRA V1 S42 contract — Cross-View Relational Signature-Geometry Anchoring

Status: **OPEN / PREREGISTERED BEFORE S42-A0 EXPOSURE**

Issue: #267

Parent:
- S41 PR #266
- merged main `8b4d0cc886cf2ee971b0e19cabfc68341d14a21d`
- S41 one fresh matched DEV completed
- S41 frozen interpretation: **Case B**
- no S41 confirmation / multilingual / external Laya-Jev evaluation

## 1. Scientific question

S41 retained strong S38-style correctness gains and repaired agreement collapse, but the selected treatment still lost **0.1201088** signature discrimination margin relative to its matched reference.

S42 asks:

> Can full-bilinear joint correctness co-adaptation retain S41 correctness while the **actual stateful AdamW runtime movement** is constrained against the full cross-view relative option signature geometry rather than independent per-signature cosine drift?

## 2. Frozen shared shell

Reference and treatment use:
- exact S35 native 256D relation representation/signature
- exact S17 primary/fusion shell
- final A13 attention LoRA **16,384**
- shared primary projection **32,768**
- original A13 frozen
- HIRACore frozen
- exact masked-mean + L2 native query summary
- state-once
- full-K
- opaque option IDs
- no projected relation path
- exact existing S35/S38 selector/gates

Reference trainable surface:
**49,152**

Treatment:
- runtime **49,152**
- full bilinear W **65,536**
- total **114,688**

No new learned S42 parameters.

## 3. Reference arm

Reference:
- exact native S35/S17 objective
- same initialization seed as treatment
- exact same TRAIN semantic rows and batch order
- independent runtime/optimizer state
- no W
- exact S41 frozen AdamW semantics

Reference signature geometry is detached when used by the S42 anchor.

## 4. Treatment correctness family

Treatment keeps exact S38/S41 full-bilinear correctness readout:

`residual_k = signature_k^T W q_hat`

Frozen:
- W shape **256x256**
- W params **65,536**
- zero init
- no bias
- no nonlinearity
- no rank/factorization
- residual scale **1.0**
- no option/domain/K-specific params
- no W-only optimizer or scheduler

Joint correctness gradients may reach treatment runtime and W.

## 5. Cross-view relational signature geometry

For each matched semantic batch:

- treatment canonical signatures `S_tc [B,K,256]`
- treatment paraphrase signatures `S_tp [B,K,256]`
- detached matched-reference signatures `S_rc`, `S_rp`

Normalize every signature over the native dimension with epsilon **1e-12**.

Define:

`G_t = normalize(S_tc) @ normalize(S_tp)^T`

`G_r = normalize(S_rc) @ normalize(S_rp)^T`

where:
- `G_t, G_r` shape `[B,K,K]`
- the last transpose swaps option/native axes so each matrix entry is a cross-view option cosine.

Anchor:

`A_rel = mean((G_t - stopgrad(G_r))^2)`

Rules:
- full matrix; diagonal and off-diagonal receive equal elementwise weight
- no diagonal/off-diagonal coefficient
- no margin
- no gold labels
- no option IDs/features
- no learned anchor params
- no anchor coefficient
- reference fully detached
- W excluded from anchor target

Interpretation:
- diagonal preserves same-option cross-view correspondence
- off-diagonal preserves wrong-option similarities
- therefore the diagonal-vs-wrong relative geometry measured by signature discrimination is constrained directly.

## 6. Frozen permutation semantics

A simultaneous logical-option permutation P transforms:

`G -> P G P^T`.

Because S42 uses the unweighted mean squared difference over all KxK entries:

`mean((P G_t P^T - P G_r P^T)^2) = mean((G_t - G_r)^2)`.

Therefore the anchor scalar must be invariant under identical logical-option permutation of treatment/reference signatures.

No canonical position is privileged.

## 7. Frozen optimizer semantics

Exact S41 optimizer semantics:
- AdamW
- lr **2e-4**
- betas **(0.9, 0.999)**
- eps **1e-8**
- weight decay **0.01**
- amsgrad false
- maximize false
- foreach false
- fused false
- capturable false
- differentiable false

Gradient clipping:
- global L2 clip **1.0**
- over treatment runtime + W together
- applied before AdamW candidate/state transition

Reference uses its own independent AdamW state.

## 8. Exact candidate AdamW movement

Reuse the qualified S41 engine:
- derive exact candidate parameter movement using standard PyTorch AdamW on shadow trainable tensors
- persist exact first/second moments and step counter
- include decoupled weight decay
- candidate/state transition must remain parity-locked to standard PyTorch AdamW

The next optimizer state is derived from the original clipped raw gradient and is not recomputed from the projected parameter movement.

## 9. Actual-step relational projection

Let:

`a = grad(A_rel)`

with respect to treatment runtime params only.

Split candidate treatment movement:
- runtime `delta_r`
- W `delta_W`

If:

`a dot delta_r <= 0`

then:

`delta_r' = delta_r`

Else:

`delta_r' = delta_r - ((a dot delta_r)/(||a||^2 + 1e-12)) a`

W:

`delta_W' = delta_W`

Apply float-representable parameter targets exactly as qualified in S41.

Persist AdamW state derived from original clipped raw gradients.

No projection slack.
No anchor coefficient.
No partial detach.
No gradient mixing.

## 10. Required S42-A0

Wholly fresh S42-A0 diagnostic authority only.

Must prove:

Identity/capacity:
- zero-init reference/treatment native relation identity
- zero-init native signature identity
- zero-init fused identity with W=0
- selected-choice identity **1.0**
- W params **65,536**
- reference/treatment trainable surfaces **49,152 / 114,688**

Relational anchor ownership:
- relational anchor -> reference runtime gradient exactly **0**
- relational anchor -> W gradient exactly **0**
- synthetic treatment drift gives anchor >0
- relational anchor -> treatment LoRA finite/nonzero

Relative geometry sensitivity:
- diagonal-only cross-view correspondence perturbation changes anchor by >0
- off-diagonal-only relative-geometry perturbation changes anchor by >0
- simultaneous logical-option permutation changes anchor scalar by <= numerical tolerance
- the anchor operates on the full KxK matrix, not only diagonal entries

Correctness path:
- joint correctness -> W finite/nonzero
- off-diagonal W gradient finite/nonzero
- joint correctness -> treatment LoRA finite/nonzero

Optimizer:
- candidate params/deltas match standard PyTorch AdamW
- first moments match
- second moments match
- step counters match
- decoupled weight decay included
- conflicting actual runtime movement has positive pre-dot
- projected post-dot approximately zero
- safe movement exact identity
- W candidate movement unchanged
- applied runtime/W movement within explicit float-rounding bounds
- actual applied relational-anchor dot within explicit bound

Mechanics:
- question-token permutation invariance
- masked query-padding invariance
- logical-option equivariance
- arbitrary K>=3 including K=7
- native projection independence
- checkpoint roundtrip
- probability mass
- full-K
- state-once

A0 semantic accuracy is diagnostic only and MUST NOT tune S42.

## 11. Fresh matched TRAIN/DEV

Only after:
1. this contract frozen
2. interpretation plan frozen
3. S42-A0 qualified
4. A0 receipt frozen
5. matched trainer/workflow frozen
6. exact staged-head generic CI PASS
7. separate one-shot TRAIN/DEV marker

Intended authority:
- seed **63001**
- TRAIN **768**
- DEV **192**
- **12** wholly fresh S42 domains
- K=4
- 2 state views
- 2 question wording views
- 2 option views
- 24 epochs
- batch 16
- exact frozen S41 AdamW semantics
- grad clip 1.0
- identical rows/order reference vs treatment
- independent runtime/optimizer state

Freshness:
- no exact S0-S41 exposed rows
- no S42-A0 rows
- no M5 final/confirmatory rows
- no W29-W34 sealed rows

## 12. Selector / gates

Use exact existing S35/S38 selector independently per arm.

DEV_READY gates remain unchanged.

Treatment additionally requires:
- W capacity exact
- optimizer-step engine parity qualified
- relational anchor ownership qualified
- reference batch/order identity
- W candidate never projected
- actual movement float guards qualified

## 13. Frozen interpretation

A — correctness retained + relative geometry protected:
material S41-like correctness gain while avoiding the S41 signature-discrimination collapse.
Then S42 is viable for wholly fresh confirmation.

B — correctness retained but relative geometry still collapses:
full cross-view KxK geometry anchoring is insufficient; close S42.

C — relative geometry protected but correctness collapses:
the constrained relational movement was necessary for S41 correctness; close this family.

D — treatment DEV_READY:
freeze immediately; no second S42 DEV; separate wholly fresh confirmation before external matched evaluation.

## 14. Stop rule

After one S42 DEV:
- no anchor coefficient/slack
- no mixing with S41 per-signature anchor
- no diagonal/off-diagonal weighting
- no margin hyperparameter
- no alternate matrix norm
- no optimizer-state reinterpretation
- no partial detach
- no gradient mixing
- no W-only LR/scheduler
- no rank/factorization
- no residual scale/bias/nonlinearity
- no seed/LR/epoch/batch retry
- no selector/gate weakening
- no second DEV

Scientific FAIL is valid.

No Laya/Jev benchmark before a separately confirmed DEV_READY candidate.
