# HIRA V1 S10 — preregistered interpretation plan

Status: **FROZEN BEFORE S10 TRAIN/DEV RESULT IS READ**

Qualified A0:
- run `36429235962`
- artifact `10973275125`
- outcome `HIRA_V1_S10_A0_IDENTITY_READY`

This note does not modify S10 architecture, evidence, optimizer, losses, selection order or gates.

## 1. Primary hypothesis

S9 established that query sensitivity can generalize:
- question-swap choice-change 0.8229166667.

But S9 still had:
- low semantic accuracy;
- negative signed gold margin;
- zero decision-margin satisfaction.

S10 tests whether the missing link is direct:
**question -> state evidence -> correct dynamic option grounding**.

The controlled change is replacing the state-blind question-option auxiliary target with a parameter-free state-grounded target, at the same 0.10 coefficient/temperature scale.

## 2. A0 baseline that must not be tuned against

Fresh A0 grounding:
- canonical accuracy 0.34375;
- canonical signed grounding margin -0.1946920;
- paraphrase accuracy 0.34375;
- paraphrase signed grounding margin -0.2491169;
- grounding cross-view choice agreement 0.40625;
- canonical normalized attention entropy ~0.8098;
- canonical max attention weight ~0.2219.

These values are diagnostic only. They cannot justify changing S10 settings.

## 3. Outcome classes

### A. Grounding + decisions improve together

Pattern:
- canonical grounding accuracy/margin rise materially;
- primary canonical accuracy and paired correctness rise;
- question sensitivity stays strong;
- cross-view stability does not collapse.

Interpretation:
- direct state-conditioned grounding is supported as a useful missing supervision signal.

Only the frozen DEV_READY gate may authorize sealed confirmation.

### B. Grounding improves, decisions do not

Pattern:
- grounding accuracy and signed grounding margin rise strongly;
- attention becomes more selective in a semantically correct way;
- primary triadic accuracy/paired correctness stagnate.

Interpretation:
- the representation can learn correct state->option grounding, but the **primary parameter-free triadic decision operator does not exploit it sufficiently**.

A future track may integrate the already parameter-free grounding evidence into the decision rule. Do not widen model capacity by default.

### C. Attention sharpens but grounding stays wrong

Pattern:
- max attention weight rises / entropy falls;
- grounding signed gold margin stays negative or grounding accuracy stagnates;
- primary accuracy does not improve.

Interpretation:
- the model is becoming confidently selective over the wrong state evidence.
- stronger attention temperature or grounding coefficient is not authorized inside S10.
- future work should address evidence identity/role alignment rather than sharpen attention.

### D. Grounding remains diffuse and weak

Pattern:
- attention entropy/max weight remain near A0 scale;
- grounding accuracy/margin do not improve;
- TRAIN grounding loss also changes little.

Interpretation:
- the current A13-LoRA + shared projection surface may not realize query-conditioned grounding under this parameter-free operator.
- S10 closes without temperature/coefficient/capacity retries.

### E. TRAIN grounding improves but fresh DEV grounding collapses

Pattern:
- TRAIN grounding loss/accuracy improve;
- fresh DEV grounding accuracy/margin stagnate or regress;
- wording-view generalization remains weak.

Interpretation:
- the bottleneck remains generalization of evidence grounding across lexical/template changes.
- no DEV-driven early stopping or grounding retuning is permitted.

### F. Primary decisions improve without grounding improving

Pattern:
- canonical decision accuracy/paired correctness rise;
- grounding metrics remain near baseline.

Interpretation:
- improvement cannot be attributed to the intended grounding mechanism with confidence; the shared representation may have improved through the other retained objectives.
- do not claim grounding hypothesis success from decision accuracy alone.

## 4. Diagnostic priority

Read metrics in this order:
1. primary canonical paired both-correct / accuracy;
2. canonical grounding accuracy;
3. canonical signed grounding gold-vs-max-wrong margin;
4. question-swap choice-change;
5. grounding cross-view agreement;
6. primary cross-view agreement / JS;
7. attention entropy and max-weight concentration.

Attention concentration alone is never semantic success.

## 5. Forbidden post-result reinterpretations

After S10 DEV exposure, do not:
- tune attention temperature from DEV;
- tune grounding coefficient/temperature from DEV;
- invent a grounding-accuracy gate after seeing the result;
- lower existing gates;
- retry seed/LR/templates;
- reuse S10 DEV rows in S11;
- call lower entropy a success if grounding correctness is weak;
- reopen sealed/M5/Laya/Jev unless `HIRA_V1_S10_GROUNDING_DEV_READY` is achieved.
