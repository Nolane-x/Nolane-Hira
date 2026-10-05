# HIRA V1 S68 contract — Context-Modulated Antisymmetric Pairwise Comparator

Status: **FROZEN / PRE-A0**

Issue: #323

Parent:
- S67 merged main `942871bb44bbef3e28e74698f161b02c6796cdcb`
- unique scientific run `37336779415`
- verdict **Case B**.

## Controlled scientific question

Can the shared state/query semantic frame improve pairwise comparison if it modulates the comparison metric itself, rather than being asked to rescue weak pairwise evidence downstream through another reliability target?

## Matched pairwise architecture

Both reference and treatment:
- exact S59 detached representation dim **512**
- exact S59 hidden width **64**
- `A [64,512]`
- `u [64]`
- `G [64,8]`
- total trainable **33,344**
- no bias
- exact same initialization
- same optimizer and gold-pair objective.

S59 base params:
- **32,832**

Added matched modulation params:
- **512**

Treatment parameter advantage:
- **0**.

## Context projection

Fixed buffer:
- `P_ctx [8,512]`
- seed **68068**
- row normalized
- zero trainable params.

Set context uses detached mean across options.

Reference:
- identity half retained;
- joint-context half exactly zeroed.

Treatment:
- full identity + joint state/query/option representation.

## Modulation

`gamma = 1 + 0.5*tanh(G z)`

Therefore gamma is bounded to approximately **[0.5,1.5]**.

For each pair:
- normalize detached option representation;
- `d_ij=r_i-r_j`;
- `h_ij=tanh(A d_ij)`;
- `raw_ij=u·(h_ij*gamma)`;
- explicit antisymmetrization.

G initializes to zero, so gamma=1 exactly and both arms must be functionally identical to S59 at A0.

## Ownership

The pairwise path may update only:
- A
- u
- G.

It may not update:
- native runtime
- cache
- correction
- detached representation.

## A0 authority

Use only already-exposed parent rows plus synthetic mechanics.

Must prove:
- 33,344 vs 33,344 params
- added treatment params 0
- tensors exactly A,G,u
- arm initialization bit-identical
- A/u exact S59 initialization
- G exact zero
- context projection identical/non-trainable
- initial gamma exact 1
- initial pair matrix exact S59
- initial aggregate exact S59
- initial selected choice exact S59
- reference and treatment context features differ when joint context exists
- zeroing joint-context half makes reference/treatment context equal
- option permutation context invariance
- pair/aggregate permutation equivariance
- antisymmetry exact
- diagonal exact zero
- K=3/7/255
- finite probability mass
- A/u/G gradients live
- reference/treatment G gradients differ on contextual probe
- representation/correction/native/cache receive no pairwise gradient
- checkpoint replay exact
- fresh S68 TRAIN/DEV exposed=false.

## Fresh S68 scientific court

Only after A0 + exact pre-DEV CI:
- seed **89001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S68 domains
- exact S67 state/question/option overlap **0**
- K=4
- 24 epochs
- one DEV only.

Primary readout is pairwise semantic transfer:
- gold-pair accuracy
- gold-pair margin
- pairwise aggregate correctness.

Final decision diagnostics must also report:
- fused canonical/paraphrase
- paired both-correct
- question-swap
- agreement
- JS.

## Stop rule

After one S68 DEV:
- no projection seed/dim change
- no modulation scale change
- no additive context bias
- no hidden width/rank change
- no context-source retrofit
- no objective coefficient change
- no optimizer change
- no reliability-target reopening
- no retry
- no second DEV
- no external Laya/Jev evaluation.
