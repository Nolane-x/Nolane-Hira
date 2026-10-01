# HIRA V1 S28 handoff — to S29 Query-Explicit Role Binding

S28 is frozen as:

`HIRA_V1_S28_ANCHOR_FACTOR_TRANSPORT_DEV_FAIL`

Authority:
- A0 run `36867793857`
- A0 artifact `11164614334`
- A0 digest `sha256:b3033c40357ba4b9b8cca378377a6d7e87f27f7e8449489e0beef20eb7fdec22`
- TRAIN/DEV run `36872373034`
- artifact `11168765249`
- artifact digest `sha256:5a5e28ea906cd5b86bff11177818382a1487934ab7ed99d53eaacf1e8a67bb5e`
- selected epoch **20**
- checkpoint `b9e0f59b5e624981db79f6e0e2525fe4f350ddfaf727164fdd8aefd2717fd8cd`

Selected DEV:
- fused canonical/paraphrase **0.5260416667 / 0.734375**
- paired **0.203125**
- question-swap **0.453125**
- fused agreement **0.5026041667**
- fused JS **0.0705156003**
- fused margins **-0.1302195030 / +0.3248527013**
- relation canonical/paraphrase **0.4296875 / 0.4583333333**
- relation margins **-0.0796973606 / -0.0394903719**
- relation agreement **0.6015625**
- final-signature cosine **0.6421900640**
- final-signature discrimination margin **-0.0111360058**
- role-anchor cosine **0.9185716013**
- role-anchor discrimination margin **-0.0099708166**
- value-anchor cosine **0.8572883606**
- value-anchor discrimination margin **-0.0366763414**

Best residual evidence:
- role-anchor cosine **0.9266317983**
- value-anchor cosine **0.8881069670**
- role discrimination margin never positive; best **-0.0088916677**
- value discrimination margin never positive; best **-0.0205161761**
- final-signature discrimination margin best only **0.0020386775**

## Stop rule applied

Do not tune S28 transport strength, factor weights, margin, negative mining, seed, LR or epochs.

## S29 target

**Final-Block Attention+FFN LoRA Semantic Adaptation**

### Why this is the next non-duplicative intervention

Do not reopen:
- S11/S12 query-role / role-value binding;
- S13/S26 relation-signature factorization;
- S18-S20/S27-S28 consistency or transport-only losses;
- S22-S23 optimizer-priority variants;
- S24 fusion weighting;
- W7 explicit conjunctive/smooth-AND factor scoring.

Those families already have fresh negative evidence.

One important A13 surface remains untested in v1: the feed-forward transform of the final BERT layer.

S6 introduced LoRA only on:
- `attention.self.query`
- `attention.self.key`
- `attention.self.value`
- `attention.output.dense`

Every S7-S28 descendant inherited that same attention-only A13 adaptation.

### S29 controlled reset

Return to the **S17 semantic frontier** rather than inheriting the degraded S26-S28 inference changes.

Keep S17:
- S13 relation expert;
- S14 equal standardized full-K fusion;
- S17 primary/relationship loss partition;
- S17 relation-priority norm-balanced gradient handling;
- shared 256→128 projection;
- state-once/full-K/opaque IDs;
- all original A13 weights frozen;
- HIRACore frozen.

Change only final-block A13 LoRA coverage.

Existing final attention LoRA:
- four 256→256 linears;
- rank 8 / alpha 8 / dropout 0;
- **16,384 params**.

Add:
- final `intermediate.dense`: 256→1024 rank-8 LoRA = **10,240 params**;
- final FFN `output.dense`: 1024→256 rank-8 LoRA = **10,240 params**.

Frozen S29 physical surface:
- A13 LoRA total **36,864**;
- shared projection **32,768**;
- total trainable **69,632**.

No learned downstream head/router/gate/calibrator.
No additional encoder layer.
No change to LoRA rank/alpha after A0.

### Required A0

Before any fresh DEV:
- exact zero-init token/pooled/logit/choice identity vs S17/S14;
- exact six LoRA module paths and shapes;
- exact **36,864** A13 LoRA + **32,768** projection = **69,632**;
- original A13/HIRACore frozen;
- attention-LoRA gradient nonzero;
- FFN intermediate-LoRA gradient nonzero;
- FFN output-LoRA gradient nonzero;
- shared projection gradient nonzero;
- state-once/full-K/option permutation/mass;
- S17 norm-balanced gradient mechanics unchanged;
- complete six-module checkpoint roundtrip/replay.

Fresh S29 authorities must exclude all exact S0-S28 rows, M5 final/confirmatory rows and W29-W34 sealed rows.

No Laya/Jev benchmark until DEV_READY.
