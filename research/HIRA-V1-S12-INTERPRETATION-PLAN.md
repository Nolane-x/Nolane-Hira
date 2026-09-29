# HIRA V1 S12 — preregistered interpretation plan

Status: **FROZEN BEFORE S12 TRAIN/DEV RESULT IS READ**

Qualified A0:
- run `36497191159`
- artifact `11004450349`
- artifact digest `sha256:62340525b36377e08d5ec481d852dd9422264a43579a0347c7d126943180bbf5`
- outcome `HIRA_V1_S12_A0_IDENTITY_READY`

This note does not modify S12 architecture, data, optimizer, losses, selection order, temperatures, or gates.

## 1. Primary hypothesis

S10 failed after pooling question-conditioned state evidence.
S11 improved role localization but failed when role->value transport was defined by a fixed positional window.

S12 tests whether the missing invariant is the **semantic relation itself**:

`query role -> state role anchor -> role-relative value relation <-> option role-relative value relation`

without any fixed token-distance assumption.

## 2. Frozen A0 baseline

Canonical:
- relation-binding accuracy: 0.46875;
- signed relation-binding margin: -0.3835980594;
- state-role entropy: 0.8865347318;
- state-role max weight: 0.1507578027;
- option-role entropy: 0.8776449505;
- option-role max weight: 0.2356426856;
- mean best pair score: 0.7325994652.

Paraphrase:
- relation-binding accuracy: 0.34375;
- signed relation-binding margin: -0.3163611293;
- relation-binding cross-view agreement: 0.71875.

These values are diagnostic only. They cannot justify changing the operator or training contract.

## 3. Outcome classes

### A. Relation binding and primary decisions improve together

Pattern:
- fresh relation-binding accuracy and signed margin improve materially;
- paired both-correct and primary accuracy rise;
- question-change remains strong;
- cross-view stability is preserved.

Interpretation:
- role-relative relation geometry is supported as a useful missing semantic mechanism.

Only the frozen DEV_READY gate may authorize sealed confirmation.

### B. Relation binding improves, primary decisions do not

Pattern:
- relation-binding accuracy/margin become strong;
- primary triadic decision quality remains weak.

Interpretation:
- the representation contains usable relation evidence, but the primary parameter-free triadic decision operator does not consume it effectively.

A later track may integrate already parameter-free relation evidence into the primary decision rule. Do not increase capacity by default.

### C. Role anchors improve but relation pairs stay wrong

Pattern:
- state/option role entropy falls or max weight rises;
- signed relation margin remains negative;
- pair evidence does not become correctly discriminative.

Interpretation:
- role localization is not the remaining bottleneck; role-relative residual geometry is insufficient or ambiguous.

Do not sharpen role temperature from exposed DEV.

### D. Pair scores rise without gold separation

Pattern:
- mean best pair score rises strongly for all options;
- relation-binding accuracy/margin stays weak.

Interpretation:
- the operator is exploiting generic token similarity or a high-similarity shortcut rather than correct role/value relation identity.

Do not increase pair-score weight or contrastive sharpness from DEV.

### E. TRAIN relation binding improves but fresh DEV collapses

Pattern:
- TRAIN relation-binding loss improves strongly;
- fresh DEV relation accuracy/margin stagnates or regresses;
- cross-view generalization remains weak.

Interpretation:
- relation geometry is still lexical/template-specific under the current semantic surface.

No seed/LR/template/temperature retry is permitted inside S12.

### F. Primary decisions improve without relation binding improving

Pattern:
- primary accuracy/paired correctness rise;
- relation metrics remain near A0 baseline.

Interpretation:
- improvement cannot be attributed to the intended S12 mechanism with confidence; retained objectives/shared representation adaptation may explain it.

Do not claim relation-binding success from primary accuracy alone.

## 4. Diagnostic priority

Read metrics in this order:
1. canonical paired both-correct / accuracy;
2. canonical relation-binding accuracy;
3. canonical signed relation-binding margin;
4. question-swap choice-change;
5. relation-binding cross-view agreement;
6. primary cross-view agreement / JS;
7. state-role and option-role concentration;
8. mean best pair score.

Role concentration or high pair score alone is never semantic success.

## 5. Forbidden post-result actions

After S12 DEV exposure, do not:
- tune role temperature;
- tune contrastive temperature;
- change the 0.5 direct/relation arithmetic mean;
- change binding coefficient;
- lower gates;
- retry seed/LR/templates;
- reuse S12 DEV rows in a future track;
- reintroduce a token-distance window;
- increase parameter count merely because S12 fails;
- reopen M5/Laya/Jev unless `HIRA_V1_S12_RELATION_BINDING_DEV_READY` is achieved.
