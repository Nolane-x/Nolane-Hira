# HIRA V1 S11 handoff — to S12 Relation-Structured Role/Value Binding

S11 is frozen as:

`HIRA_V1_S11_ROLE_BINDING_DEV_FAIL`

## Canonical evidence

A0:
- run `36440258349`
- artifact `10978171127`
- outcome `HIRA_V1_S11_A0_IDENTITY_READY`

TRAIN/DEV:
- run `36441386657`
- artifact `10979667976`
- digest `sha256:8780134b6d05d46aeeb9c60b77f1bb3ec07f9cfb76bec1eda5c52d708bf6de2b`
- selected epoch **24**
- checkpoint SHA256 `1112c1085633b897aa7de2f373401c1f41d095d2c18a3c5cf866bea31f891f55`

Selected DEV:
- canonical accuracy **0.3489583333**
- paired both-correct **0.125**
- question-swap choice-change **0.6458333333**
- cross-view decision agreement **0.2526041667**
- canonical binding accuracy **0.4192708333**
- canonical signed binding margin **-0.9078732127**
- binding cross-view agreement **0.2057291667**
- role entropy **0.7444132517**
- role max weight **0.2625867178**
- option-order flip **0.0**
- full-K/state-once/relation-delta-zero PASS

Training:
- binding loss **1.37438 -> 0.39169**
- fresh binding accuracy peaked **0.52344** at epoch 7
- fresh signed margin best **-0.11302** at epoch 6, then deteriorated

## Key result

S11 improves role localization but does not solve value association.

The fixed token-distance window is the wrong structural prior: different wording can place the semantic value before, after, or farther from the role phrase. Sharper role localization therefore does not guarantee the correct value is transported to the option comparison.

## Do not do

Do not:
- tune S11 role temperature;
- tune contrastive temperature;
- widen/narrow the value window on exposed DEV;
- retry seed/LR/templates;
- reuse S11 DEV rows;
- increase parameter count merely because S11 failed;
- reopen M5/Laya/Jev.

## S12 target

Keep the exact **49,152-parameter** optimization surface.

Test **Relation-Structured Role/Value Binding**:
- identify role evidence from the question/state interaction;
- identify candidate value evidence from state/option interaction;
- score explicit role-token/span -> value-token/span pairs;
- use parameter-free structural compatibility rather than a fixed token-distance window;
- support value-before-role and value-after-role wording symmetrically;
- preserve full-K and option permutation equivariance;
- add zero learned downstream parameters;
- use wholly fresh TRAIN/DEV rows and preregister all gates before exposure.

The S12 question is:

**Can explicit role-value relation pairing generalize where both pooled grounding (S10) and fixed-locality binding (S11) fail?**
