# HIRA V1 S5 handoff — to S6 Limited A13 Semantic Encoder Adaptation

S5 is frozen as:

`HIRA_V1_S5_PROJECTION_DEV_FAIL`

## Canonical evidence

A0:
- run `36400104371`
- artifact `10960266188`
- digest `sha256:0ba8a60dd46a8cae47ed7098c558cce9fa7e8b4e137231116bb458943a386607`

TRAIN/DEV:
- run `36400648104`
- artifact `10960895474`
- digest `sha256:b6cb1154ccb5fe0fce38481f82bdb00881e6d4c277fa66a4e2cad0a3a1fd06c6`
- selected epoch 29
- checkpoint SHA256 `1babafd79bc2c97c438856c6943741f5f1bcc9d874664d214bcc050f04141560`

Selected DEV:
- accuracy 0.3125
- paired both-correct 0.0625
- question-swap choice-change 0.21875
- option-order flip 0.0
- full-K/state-once/relation-delta-zero PASS

## Key scientific result

S5 successfully learned strong two-view alignment on TRAIN but did not convert it into transferable decision semantics on fresh DEV.

The external shared projection is therefore insufficient. The next track should adapt A13 itself under a tightly bounded parameter budget.

## Do not do

Do not:
- retune S5 on exposed DEV;
- retry S5 InfoNCE weight/temperature;
- retry projection initialization/width;
- reuse S5 rows for S6 fitting or selection;
- reopen M5/Laya/Jev from S5.

## S6 target

Test limited A13 semantic adaptation.

Preserve:
- state-once;
- dynamic full-K;
- opaque IDs;
- no dataset/domain-specific head;
- fresh evidence only;
- simple downstream scoring.

Preferred first candidate:
- freeze embeddings and early A13 layers;
- adapt only a compact late-layer surface or low-rank adapters in the final encoder block;
- freeze the downstream relation/scoring path so the causal intervention is clearly inside A13.

Do not unfreeze the entire 12.75M A13 frontend as the first S6 experiment.
