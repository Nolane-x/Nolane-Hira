# HIRA V1 S11 — preregistered interpretation plan

Status: **FROZEN BEFORE S11 TRAIN/DEV RESULT IS READ**

Qualified A0:
- run `36440258349`
- artifact `10978171127`
- artifact digest `sha256:95115c357f9b9174bf4f0c2b0beb3f5d5829829e432e00116165e5dcee0f080b`
- outcome `HIRA_V1_S11_A0_IDENTITY_READY`

This note does not modify S11 architecture, data, optimizer, losses, selection order or gates.

## Primary hypothesis

S10 showed that pooled state grounding can become selective on TRAIN while failing fresh semantic grounding on DEV.

S11 tests whether preserving:
`question role -> local state value -> option token`
until the final late-interaction score improves fresh role/value binding and primary decisions.

## Frozen A0 baseline

Canonical:
- binding accuracy 0.28125
- signed binding margin -0.3389761
- role entropy 0.8970667
- role max weight 0.1286201
- value entropy 0.9805641
- value max weight 0.0617842

Paraphrase:
- binding accuracy 0.34375
- signed binding margin -0.3489900

These values are diagnostic only and cannot justify retuning S11.

## Outcome classes

### A. Binding and primary decisions improve together

Pattern:
- fresh binding accuracy and signed margin rise materially;
- paired both-correct and primary accuracy rise;
- question sensitivity remains strong;
- cross-view stability is preserved.

Interpretation:
- explicit role/value token structure is supported as a missing semantic supervision mechanism.

Only the frozen DEV_READY gate may authorize sealed confirmation.

### B. Binding improves, primary decisions do not

Pattern:
- binding accuracy/margin become strong;
- primary triadic decision metrics remain weak.

Interpretation:
- the representation can support role/value binding, but the primary parameter-free triadic decision operator does not exploit it sufficiently.

A later track may integrate already parameter-free binding evidence into the decision rule. Do not increase capacity by default.

### C. Role localizes but value binding stays wrong

Pattern:
- role entropy falls / role max weight rises;
- binding accuracy or signed margin stays weak;
- value support remains diffuse or semantically wrong.

Interpretation:
- query-role identification works but fixed local role->value transport is insufficient.

Do not tune the value window from DEV. A future track must change the structural relation mechanism with fresh evidence.

### D. Role and value remain diffuse

Pattern:
- role/value concentration remains near A0 scale;
- TRAIN binding loss changes little;
- fresh DEV binding does not improve.

Interpretation:
- the current A13-LoRA + shared projection does not realize role-preserving localization under this parameter-free operator.

S11 closes without coefficient/temperature/window retries.

### E. TRAIN binding improves but fresh DEV collapses

Pattern:
- TRAIN binding loss improves strongly;
- fresh DEV binding accuracy/margin stagnate or regress;
- wording-view agreement stays weak.

Interpretation:
- the bottleneck remains generalization across lexical/template variation rather than binding capacity.

No DEV-driven retry is permitted.

### F. Primary decisions improve without binding improvement

Pattern:
- primary decision metrics rise;
- binding metrics remain near baseline.

Interpretation:
- the intended role-binding mechanism is not established as the cause; retained objectives/shared adaptation may explain the improvement.

Do not claim the S11 hypothesis succeeds from decision accuracy alone.

## Diagnostic priority

Read metrics in this order:
1. canonical paired both-correct / accuracy;
2. canonical binding accuracy;
3. canonical signed binding margin;
4. question-swap choice-change;
5. binding cross-view agreement;
6. primary cross-view agreement / JS;
7. role concentration diagnostics.

Concentration alone is never semantic success.

## Forbidden post-result actions

After S11 DEV exposure, do not:
- tune role temperature;
- tune binding contrastive temperature;
- tune value window;
- tune binding coefficient;
- lower existing gates;
- retry seed/LR/templates on exposed DEV;
- reuse S11 DEV rows in S12;
- reopen Laya/Jev unless `HIRA_V1_S11_ROLE_BINDING_DEV_READY` is achieved.
