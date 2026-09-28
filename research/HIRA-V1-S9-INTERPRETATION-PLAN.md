# HIRA V1 S9 — preregistered interpretation plan

Status: **FROZEN BEFORE S9 TRAIN/DEV RESULT IS READ**

Authority run already executing from exact head:

`84d67ae04a0ae02026f2f5be35843442ecebf9e8`

This note does not modify S9 architecture, data, optimizer, loss, selection or gates.

Its purpose is to preregister how the margin diagnostics will be interpreted so the research narrative is not chosen after observing DEV.

## 1. Primary hypothesis

S8 showed:
- materially better semantic accuracy and question sensitivity;
- extremely low cross-view JS divergence;
- weak cross-view selected-choice agreement.

S9 tests whether the missing ingredient is explicit separation of the correct option from **all** incorrect options.

The relevant intervention is:
- same 49,152-parameter model surface as S8;
- same two-view evidence structure;
- replace pair-specific swap margin with gold-vs-all-negative margin;
- same margin 0.20 and coefficient 0.25.

## 2. Outcome classes

### A. Correct separation

Evidence pattern:
- canonical accuracy rises materially;
- paired both-correct rises;
- signed gold-vs-max-wrong margin becomes more positive;
- margin-satisfaction rate rises;
- cross-view choice agreement does not collapse.

Interpretation:
- explicit all-negative separation is supported as a useful missing supervision signal.

Only the preregistered DEV_READY gate may authorize sealed confirmation.

### B. Confident but wrong

Evidence pattern:
- absolute top1-top2 margin rises;
- but signed gold-vs-max-wrong margin remains weak/negative;
- or margin satisfaction remains low;
- accuracy does not improve.

Interpretation:
- the model has learned confidence/separation without correct semantic grounding.
- stronger margin coefficients or larger margins are **not** authorized inside S9.
- a future track should address semantic grounding/alignment rather than merely sharpen logits.

### C. Correct margin improves, wording invariance degrades

Evidence pattern:
- canonical signed margin/accuracy improve;
- paraphrase margin or cross-view selected-choice agreement degrades materially.

Interpretation:
- S9 separation is wording-specific.
- a future track should align **decision margins/rankings across views**, not simply increase capacity.

### D. Margin remains near zero

Evidence pattern:
- signed gold margins and margin satisfaction remain close to A0/S8 scale;
- TRAIN objective does not create meaningful separation.

Interpretation:
- the existing A13-LoRA + projection surface may not realize the required separation under this objective.
- S9 must close without loss/learning-rate/margin retries.

### E. TRAIN separation but fresh DEV collapse

Evidence pattern:
- TRAIN margin/CE improve;
- DEV signed margin, accuracy or paired correctness stagnate/regress.

Interpretation:
- the bottleneck remains generalization across lexical/template variation rather than raw optimization capacity.
- no DEV-driven early stopping or margin retuning is permitted.

## 3. Diagnostic priority

When reading S9 DEV, distinguish these metrics:

1. **signed gold-vs-max-wrong margin**
   - positive only when gold beats the strongest distractor;
   - directly measures correct separation.

2. **margin satisfaction >= 0.20**
   - measures how often the preregistered decision margin is actually achieved.

3. **top1-top2 absolute margin**
   - measures confidence/separation even if the winner is wrong;
   - must never be treated as semantic success by itself.

4. **accuracy / paired both-correct**
   - primary semantic correctness.

5. **question-swap selected-choice change**
   - checks whether the model responds to which fact is queried.

6. **cross-view selected-choice agreement / JS**
   - checks wording invariance;
   - low JS alone is insufficient, as S8 demonstrated.

## 4. Forbidden post-result reinterpretations

After S9 DEV exposure, do not:
- redefine top1-top2 margin as success if signed gold margin is negative;
- lower the 0.20 margin target;
- lower quality gates;
- select an epoch by a metric outside the preregistered selection order;
- retry coefficient, margin, seed, LR or data templates;
- reopen sealed/M5/Laya/Jev unless DEV_READY is achieved.

A negative result is valid research evidence.
