# HIRA V1 S6 handoff — to S7 Joint A13-W28 Semantic Co-Adaptation

S6 is frozen as:

`HIRA_V1_S6_A13_LORA_DEV_FAIL`

## Canonical evidence

A0:
- run `36406203754`
- artifact `10962577478`
- digest `sha256:13ad5013cd2e4873c5801e4441602de6997e50305ccac39284886647268c4976`

TRAIN/DEV:
- run `36407493234`
- artifact `10963225985`
- digest `sha256:bfbc2cbc63c5f06cf1fc6dfdbe2f86a4a84cb05ec80f4cb98d3683401c6ae49f`
- selected epoch 20
- checkpoint SHA256 `e4ed8bafc7ef85d1b2ed2ad4fd46be0d7982758a708f1683c7969b2281d99e3b`

Selected DEV:
- accuracy 0.3333333333
- paired both-correct 0.0885416667
- question-swap choice-change 0.390625
- option-order flip 0.0
- full-K/state-once/relation-delta-zero PASS.

## Key result

S6 learned strong TRAIN representation alignment inside A13 but fresh DEV did not generalize.

S5 showed W28-only adaptation fails.
S6 showed A13-only adaptation with frozen W28 fails.

The next clean intervention is joint A13-W28 co-adaptation, not a wider S6 retry.

## Do not do

Do not:
- retune S6;
- change S6 rank/layers/loss weights using exposed DEV;
- reuse S6 rows for S7 fitting or selection;
- reopen M5/Laya/Jev from S6.

## S7 target

Jointly adapt:
- S6 final-layer A13 attention LoRA;
- S5 shared W28-style 256->128 projection.

Freeze everything else.

S7 must use fresh TRAIN/DEV evidence and a new preregistered authority.
