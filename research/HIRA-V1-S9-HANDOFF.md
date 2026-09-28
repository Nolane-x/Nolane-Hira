# HIRA V1 S9 handoff — to S10 Query-Conditioned State-to-Option Grounding

S9 is frozen as:

`HIRA_V1_S9_MARGIN_DEV_FAIL`

## Canonical evidence

A0:
- run `36421792484`
- artifact `10969906709`
- digest `sha256:e0ff0ad9907bf452fe0771822c085962b3fdda05ca95b3cf8855f7054d5dbb29`

TRAIN/DEV:
- run `36422317628`
- artifact `10971005811`
- digest `sha256:9c4a3e03a7964af6552ce8dc87608ca42074ccd8fd66d89b4dcaedca132c3094`
- selected epoch 16
- checkpoint SHA256 `532ee4b0339c8861e4851aaa21de9a051333e1cf21ffe2d1799375df1d5bc3c7`

Selected DEV:
- canonical accuracy 0.3619791667
- paired both-correct 0.203125
- question-swap choice-change 0.8229166667
- cross-view selected-choice agreement 0.703125
- canonical mean signed gold margin -0.0191702538
- canonical margin satisfaction 0.0
- option-order flip 0.0
- full-K/state-once/relation-delta-zero PASS.

## Key result

S9 is the first Hira v1 track to pass the frozen >=0.80 question-change gate.

However correct semantic separation did not generalize:
- signed gold margin remains negative;
- margin satisfaction remains zero;
- accuracy is below S8.

Therefore query routing/sensitivity is no longer the primary bottleneck. The next clean hypothesis is explicit query-conditioned **state fact -> option grounding**.

## Do not do

Do not:
- retune S9 margin/coefficient;
- reinterpret confidence separation as semantic correctness;
- reuse S9 rows for S10 fitting or selection;
- add capacity just because S9 failed;
- reopen M5/Laya/Jev.

## S10 target

Keep the exact 49,152-parameter model surface.

Add a parameter-free grounding objective:
- question selects/reweights state evidence;
- selected state evidence must align to the correct option;
- selected evidence must repel every wrong option;
- apply across canonical and paraphrase views;
- preserve a small cross-view invariance term.

The key question is whether direct state-to-option grounding can turn S9's 82.3% question sensitivity into correct choices.
