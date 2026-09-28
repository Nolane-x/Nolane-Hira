# HIRA V1 S8 handoff — to S9 Invariant Semantic Margin Separation

S8 is frozen as:

`HIRA_V1_S8_INVARIANT_DEV_FAIL`

## Canonical evidence

A0:
- run `36416813621`
- artifact `10968037685`
- digest `sha256:cd7b84cf7de183a6bd263ab8923764d810ff97f44a56b930190893e213b2804a`

TRAIN/DEV:
- run `36417365478`
- artifact `10969085984`
- digest `sha256:40a0f1ae053f03ae59b87f784c5aaeeada3fb36646e7009eb6781592e8c6b49b`
- selected epoch 23
- checkpoint SHA256 `32e3eec6ab3e1c1403284a60b24f13780783222fb237df2ae47eeaf977c4c2a7`

Selected DEV:
- canonical accuracy 0.4192708333
- canonical paired both-correct 0.1822916667
- question-swap choice-change 0.7239583333
- paraphrase accuracy 0.3880208333
- cross-view selected-choice agreement 0.6223958333
- cross-view mean JS 0.0008056818
- option-order flip 0.0078125
- full-K/state-once/relation-delta-zero PASS.

## Key result

S8 is the strongest Hira v1 fresh-DEV result so far, but low JS divergence does not imply stable decisions. The distributions can be almost identical while near-ties flip the selected option.

The next intervention should directly train decision separation rather than add model capacity.

## Do not do

Do not:
- retune S8 against exposed DEV;
- change S8 JS coefficient from DEV;
- invent an S8 early-stop rule from epoch 23/24;
- reuse S8 rows for S9 fitting or selection;
- widen the model merely because S8 missed the gate;
- reopen M5/Laya/Jev from S8.

## S9 target

Keep the exact 49,152-parameter S8 architecture.

Train on fresh two-view semantic cases with an explicit all-negative margin:
- gold versus every wrong option;
- canonical and paraphrase views;
- retain a small preregistered cross-view consistency term;
- measure top1-top2 decision margin and cross-view choice stability.

The key question is whether semantic separation converts S8's 72.4% question sensitivity into robust correct choices.
