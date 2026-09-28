# HIRA V1 S11 — preregistered interpretation plan

Status: **FROZEN BEFORE S11 TRAIN/DEV RESULT IS READ**

Qualified A0:
- run `36436701667`
- artifact `10975624156`
- outcome `HIRA_V1_S11_A0_ROLE_VALUE_READY`

This note does not modify the S11 contract.

## Primary hypothesis

S10 failed because a single pooled question-conditioned state vector had to preserve both query role identity and state value identity.

S11 separates them:
- role expert: query -> option;
- state-presence expert: option -> state;
- final decision: product of the two K-way experts.

The hypothesis is supported only if fresh DEV decisions improve together with the intended expert diagnostics.

## Frozen A0 baseline

Canonical A0:
- binding accuracy 0.375
- paired both-correct 0.125
- question-swap change 0.3125
- binding margin -0.2620205
- role entropy 0.97453
- state entropy 0.95708
- expert agreement 0.21875
- paired-gold state top-2 containment 0.0625

These values cannot justify S11 retuning.

## Outcome classes

### A. Both experts specialize and binding generalizes
Pattern:
- paired-gold state top-2 containment rises strongly;
- query-driven final choices change appropriately;
- binding accuracy/paired correctness rise on fresh DEV;
- signed binding margin becomes positive;
- wording stability remains high.

Interpretation:
- factorized role/value binding is supported.

Only the frozen DEV_READY gate may open sealed confirmation.

### B. State-presence expert succeeds; query-role expert remains weak
Pattern:
- paired-gold state top-2 containment becomes high;
- final question-swap sensitivity and/or final accuracy remain weak;
- role expert remains diffuse or does not separate the queried role.

Interpretation:
- state-value presence can be represented, but query-role selection remains the bottleneck.

### C. Query-role behavior improves; state-presence expert remains weak
Pattern:
- question-swap sensitivity rises;
- state top-2 containment stays weak;
- same-role wrong-value distractors remain competitive.

Interpretation:
- role selection works but state-value identity is not represented strongly enough.

### D. Both experts improve but product fusion fails
Pattern:
- state top-2 containment is high;
- role expert becomes discriminative;
- final binding accuracy/margin remains weak.

Interpretation:
- evidence is available in the separate experts, but fixed product-of-experts fusion is the bottleneck.

No S11 post-DEV fusion reweighting is authorized.

### E. TRAIN factorization improves but fresh DEV collapses
Pattern:
- TRAIN loss/accuracy and expert diagnostics improve strongly;
- fresh DEV binding, expert diagnostics or wording stability stagnate/regress.

Interpretation:
- the current representation still learns lexical/template shortcuts rather than transferable role/value structure.

### F. Representation does not learn the factorized task
Pattern:
- TRAIN decision loss barely improves;
- state top-2 and query sensitivity remain near A0 scale.

Interpretation:
- the current A13-LoRA + shared-projection surface may be insufficient for factorized role/value learning under this operator.

## Diagnostic priority

Read metrics in this order:
1. canonical paired both-correct / binding accuracy;
2. signed binding margin;
3. paired-gold state-support top-2 containment;
4. question-swap choice-change;
5. cross-view agreement / JS;
6. role and state expert entropy;
7. expert selected-choice agreement.

Low entropy alone is never success.

## Forbidden post-result actions

After S11 DEV exposure, do not:
- tune either expert temperature from DEV;
- add role/state fusion weights from DEV;
- lower gates;
- retry seed/LR/templates;
- reuse S11 DEV rows in S12;
- reinterpret expert concentration as correctness;
- reopen M5/Laya/Jev unless S11 DEV_READY is achieved.
