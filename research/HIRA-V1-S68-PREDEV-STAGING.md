# HIRA V1 S68 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #323  
PR: #324

## Parent S67

Merged main:
`942871bb44bbef3e28e74698f161b02c6796cdcb`

Unique S67 scientific authority:
- run `37336779415`
- verdict **Case B**
- no Actions artifact exists because the post-DEV verifier failed after TRAIN/DEV completed
- raw unique receipt is frozen in `research/HIRA-V1-S67-MATCHED-RECEIPT.json`
- no S67 DEV rerun occurred.

## Qualified S68-A0

Run: `37388612141`  
Artifact: `11380079442`  
Digest: `sha256:06cd73759293281780a4ba8e9aa1c375dd95ea1a632061c2351cf21fd5787da5`

Outcome:
`HIRA_V1_S68_A0_CONTEXT_MODULATED_PAIRWISE_READY`

Qualified:
- reference pairwise params **33,344**
- treatment pairwise params **33,344**
- added treatment params **0**
- tensors exactly `A,G,u`
- S59 base params **32,832**
- modulation params **512**
- context dim **8**
- fixed projection seed **68068**
- modulation scale **0.5**
- arm initialization bit-identical
- A/u exact S59 initialization
- G exact zero
- initial pair/aggregate/choice exact S59
- initial gamma exact 1
- reference/treatment context intervention mechanically non-degenerate
- joint-context ablation exact
- option permutation equivariance PASS
- antisymmetry exact
- diagonal exact zero
- K=3/7/255 PASS
- A/u/G gradients live
- reference/treatment G gradients differ
- no representation/correction/native gradient leakage
- checkpoint replay exact
- no fresh S68 TRAIN/DEV exposure.

## Fresh S68 authority

- seed **89001**
- TRAIN **768**
- DEV **192**
- **12 fresh S68 domains**
- exact S67 state/question/option overlap **0**
- K=4
- 24 epochs
- one DEV only.

Authority:
`src/nmd/v1_s68_authority.py`

## Matched fresh scientific trajectory

Shared exactly:
- S51 immutable native authority/cache
- one correction trajectory, **114,688 params**
- same TRAIN rows and order
- same optimizer/LR/weight decay
- same S64 contextual downstream gate architecture, **60 params per arm**
- bit-identical gate initialization
- exact same S66 per-view counterfactual responsibility BCE for both downstream gates
- same bounded residual contracts
- same frozen S17 selector.

Reference pairwise head:
- `ContextModulatedPairwiseHead(use_joint_context=False)`
- **33,344 params**
- identity-only set context.

Treatment pairwise head:
- `ContextModulatedPairwiseHead(use_joint_context=True)`
- **33,344 params**
- full identity + joint state/query/option set context.

Both pairwise heads:
- bit-identical initial parameters
- exact same gold-vs-distractor pairwise objective
- independent optimizer state
- exact same capacity
- treatment parameter advantage **0**.

S67 oracle-alpha supervision is explicitly **not** reopened.

## DEV readouts

For both arms report:
- pairwise gold-pair accuracy/margin
- direct pairwise aggregate canonical/paraphrase accuracy
- pairwise aggregate paired both-correct
- pairwise aggregate question-swap
- pairwise aggregate cross-view agreement/JS
- pairwise aggregate gold margins
- final fused canonical/paraphrase accuracy
- final paired both-correct
- final question-swap
- final agreement/JS
- final gold margins.

Fused shadow baseline remains diagnostic only.

## Stop rule

Once S68 TRAIN begins:
- no context projection seed/dim change
- no modulation-scale change
- no additive context bias
- no hidden-width/rank change
- no context-source retrofit
- no loss coefficient change
- no optimizer/LR/weight-decay change
- no reliability-target reopening
- no native retraining
- no selector change
- no retry for scientific weakness
- no second S68 DEV
- no external Laya/Jev evaluation.

## Authorization rule

Marker:
`research/HIRA-V1-S68-ENABLE-TRAIN-DEV`

It MUST remain absent until the **exact final pre-DEV staging head** passes generic CI on Python 3.10 and 3.12.

Scientific failure is valid.
