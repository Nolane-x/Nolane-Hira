# HIRA V1 S4 handoff — to S5 Semantic Projection Relearning

S4 is frozen as:

`HIRA_V1_S4_ADAPTER_DEV_FAIL`

## Canonical evidence

A0:
- run `36397966988`
- artifact `10959630634`
- digest `sha256:1943d802440ec759e4d32c97c61fdb93714ce8fb0a1897df25bd3bcdbc071c59`

TRAIN/DEV:
- run `36398414087`
- artifact `10959711099`
- digest `sha256:112aecf9aa1f9c6d5925fd7beddab5167cce7d3eca6d94d15873909ee02309b4`
- selected epoch 15
- checkpoint SHA256 `956bfb4987ce6a5ad7c983be75fafbc16eb178cf3a87ad010743bf24c2327ed8`

Selected DEV:
- accuracy 0.2760416667
- paired both-correct 0.0
- question-swap choice-change 0.09375
- option-order flip 0.0104166667
- full-K/state-once/relation-delta-zero PASS

## Do not do

Do not:
- retune S4 on exposed DEV;
- widen the adapter;
- retry S4 learning rate/epochs/activation;
- use S4 rows for future model selection;
- reopen M5/Laya/Jev from S4.

## S5 target

Test direct relearning of the shared A13 256 -> relation 128 semantic projection before considering A13 unfreezing.

Keep:
- A13 frozen;
- state-once;
- dynamic full-K;
- opaque IDs;
- no domain-specific heads;
- fresh evidence only.

Suggested S5 trainable surface:
- one shared bias-free 256->128 projection = **32,768 params**.

Suggested downstream:
- parameter-free triadic scoring.

Suggested optimization:
- paired decision CE + swap margin;
- explicit representation alignment objective so the learned projection is not driven only by narrow classification labels.

S5 must preregister fresh TRAIN/DEV before exposure.
