# HIRA V1 S12 handoff — to S13 Cross-View Relation Canonicalization

S12 is frozen as:

`HIRA_V1_S12_RELATION_BINDING_DEV_FAIL`

## Canonical evidence

A0:
- run `36497191159`
- artifact `11004450349`
- digest `sha256:62340525b36377e08d5ec481d852dd9422264a43579a0347c7d126943180bbf5`

TRAIN/DEV:
- run `36498016327`
- artifact `11004453797`
- digest `sha256:f1939bac3e772a969e7e43d0dc78c163333b7cce1baa16922e929877c04950c9`
- selected epoch **18**
- checkpoint SHA256 `60bddcd73e21220ea4131bc3f1ca5f55ba8f18c5277152902a2ca94f7b661869`

Selected DEV:
- canonical accuracy **0.28125**
- paired both-correct **0.109375**
- question-swap choice-change **0.6302083333**
- cross-view decision agreement **0.3802083333**
- canonical relation-binding accuracy **0.4010416667**
- paraphrase relation-binding accuracy **0.5390625**
- canonical signed relation margin **-0.7074106541**
- paraphrase signed relation margin **-0.1640277983**
- relation-binding cross-view agreement **0.4036458333**
- option-order flip **0.0**
- full-K/state-once/relation-delta-zero PASS

TRAIN relation-binding loss:
- epoch 1: **1.364938**
- epoch 24: **0.367593**

## Key result

Removing S11's fixed token-distance prior was not enough.

S12 can optimize relation binding strongly on TRAIN, but fresh relation identity remains wording/template dependent. The most diagnostic evidence is the selected-checkpoint asymmetry between canonical and paraphrase views despite their semantic equivalence.

Therefore the next bottleneck is not simply:
- query-role localization;
- token distance;
- relation-pair capacity.

It is **relation representation invariance across semantically equivalent wording views**.

## Do not do

Do not:
- retune S12 role temperature;
- retune contrastive temperature;
- change the 0.5 direct/relation mixture from exposed DEV;
- retry S12 seed/LR/templates;
- reuse S12 DEV rows for S13 fitting or selection;
- increase model capacity merely because S12 failed;
- reopen M5/Laya/Jev.

## S13 target

Keep the exact **49,152-parameter** optimization surface.

Test **Cross-View Relation Canonicalization**:
- retain the S12 zero-parameter relation-binding operator as a diagnostic base;
- derive a parameter-free relation signature for each semantic query/view;
- explicitly align canonical and paraphrase relation signatures for the same semantic fact;
- preserve separation between the two different queried roles within each state;
- require relation evidence and selected option evidence to remain stable across wording views;
- add no learned downstream head;
- use wholly fresh S13 TRAIN/DEV rows;
- preregister all losses, selection order and gates before exposure.

The S13 question is:

**Can explicit cross-view canonicalization make role/value relation geometry invariant enough to generalize, without adding model capacity?**
