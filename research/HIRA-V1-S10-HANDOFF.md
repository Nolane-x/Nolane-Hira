# HIRA V1 S10 handoff — to S11 Role-Preserving Evidence Binding

S10 is frozen as:

`HIRA_V1_S10_GROUNDING_DEV_FAIL`

## Canonical evidence

TRAIN/DEV:
- run `36430015114`
- artifact `10974197119`
- artifact digest `sha256:fd5326ffab4c474acfab830df470f9ddc412b212e8d8e3687fa3a34fb354c104`
- selected epoch **4**
- checkpoint SHA256 `de425041ccc3fb9c9c75a6256b94dfde97e135e4667984860bb1c7c451aca9c9`

Selected DEV:
- canonical accuracy **0.2682291667**
- paired both-correct **0.0572916667**
- question-swap choice-change **0.359375**
- cross-view choice agreement **0.359375**
- canonical grounding accuracy **0.4036458333**
- canonical signed grounding margin **-0.8021047898**
- grounding cross-view agreement **0.2317708333**
- option-order flip **0.0**
- full-K/state-once/relation-delta-zero PASS

Training signal:
- grounding loss fell from **1.380856** to **0.373064** over 24 epochs.

## Key result

S10 successfully made the auxiliary grounding task learnable on TRAIN, but the learned behavior does not generalize correctly to fresh DEV.

The parameter-free attention becomes more selective, yet the selected evidence is not reliably the correct role/value evidence. Pooling question-conditioned state evidence into one vector, then comparing it to pooled option vectors, is therefore not sufficient.

## Do not do

Do not:
- retune S10 attention temperature;
- retune S10 grounding coefficient/contrastive temperature;
- retry the S10 seed/LR/templates;
- reuse S10 DEV rows for S11 fitting or selection;
- add parameters merely because S10 failed;
- reopen M5/Laya/Jev.

## S11 target

Keep the exact **49,152-parameter** optimization surface.

Test **Role-Preserving Evidence Binding**:
- preserve token-level state/question/option interactions instead of collapsing state evidence to one pooled vector;
- separate query-role evidence from state-value support;
- require the correct option to be supported by both;
- use a parameter-free late-interaction binding operator over the existing shared projection;
- keep original A13 and HIRACore frozen;
- add zero learned downstream parameters;
- use wholly fresh TRAIN/DEV rows and preregister all gates before exposure.

The S11 question is:

**Can token-level role-preserving binding generalize where S10's pooled grounding collapses?**
