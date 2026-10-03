# HIRA V1 S45 interpretation plan — frozen before S45-A0/DEV exposure

Status: **FROZEN**

Issue: #273

## Controlled comparison

Reference:
- exact native S35/S17 runtime.

Treatment:
- exact S44 private correction architecture and capacity;
- native runtime trajectory must remain matched-identical;
- only scientific change is adding cross-view JS inside the detached correction objective.

## Frozen capacity

Native: **49,152**

Private adapter:
- A **32,768**
- B **16,384**
- total **49,152**

W: **65,536**

Correction-only: **114,688**

Treatment total: **163,840**

No capacity alternatives are in scope.

## Frozen objective

`CE_corr = 0.5 * (CE(canonical) + CE(paraphrase))`

`JS_corr = JS(softmax(canonical_corrected_relation), softmax(paraphrase_corrected_relation))`

`L_corr = 0.10 * CE_corr + 0.25 * JS_corr`

No temperature, no alternate divergence, no fusion-weight change.

## Frozen optimizer

Exact S44 optimizer semantics:
- AdamW lr 2e-4
- weight decay .01
- correction grad clip 1.0
- separate native/correction optimizers by ownership only.

## Intended authority

- seed **66001**
- TRAIN 768
- DEV 192
- 12 fresh S45 domains
- K=4
- 24 epochs
- batch 16
- identical native rows/order
- one DEV.

## Required reporting

Native identity:
- every epoch runtime-state fingerprint
- max LoRA/projection/native-logit/native-signature deltas.

Correctness:
- fused canonical/paraphrase
- paired
- question-swap
- corrected relation canonical/paraphrase
- raw primary canonical/paraphrase
- margins.

Consistency:
- corrected relation selected-choice agreement
- corrected relation cross-view JS
- fused agreement
- fused JS
- private residual/signature cross-view diagnostics.

## Frozen interpretation

**A** — S44-level correctness is retained while corrected-relation consistency and fused stability recover:
cross-view consistency is the missing private-branch constraint.

**B** — consistency recovers but correctness materially collapses:
private correctness and consistency conflict; close.

**C** — correctness remains but consistency does not materially improve:
loss-level consistency is insufficient; move to a new decision/fusion family.

**D** — treatment DEV_READY:
freeze immediately and open a separate fresh confirmation before any external Laya/Jev evaluation.

No scalar winner score.

## Stop rule

No post-DEV:
- JS coefficient or temperature sweep
- alternate divergence
- capacity/activation change
- learned fusion gate/weight
- native-gradient leakage
- second encoder
- seed/LR/epoch/batch retry
- gate weakening
- second DEV.

Scientific failure is valid.
