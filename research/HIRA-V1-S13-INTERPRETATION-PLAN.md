# HIRA V1 S13 — preregistered interpretation plan

Status: **FROZEN AFTER QUALIFIED A0, BEFORE TRAIN/DEV RESULT**

Qualified A0:
- run `36524047158`
- artifact `11013757416`
- digest `sha256:d2aeeec672dd7787aab3f3fe68799e83d303a99b74b4ebf8926ed75e44fbf8ba`
- outcome `HIRA_V1_S13_A0_IDENTITY_READY`

This note does not modify S13 architecture, data, optimizer, losses, selection order, or DEV gates.

## Frozen A0 baseline

- same-option signature cosine: **0.7203133**
- same-vs-strongest-wrong signature margin: **-0.00246885**
- canonical relation-binding accuracy: **0.28125**
- paraphrase relation-binding accuracy: **0.4375**
- canonical relation margin: **-0.3119711**
- primary cross-view selected-choice agreement: **0.6875**

## Primary hypothesis

Equivalent surface forms already produce moderately similar relation signatures, but those signatures are not sufficiently discriminative against wrong options.

S13 tests whether explicit same-option cross-view alignment plus wrong-option separation can turn this latent invariance into correct, wording-stable semantic choice without adding capacity.

## Outcome classes

### A. Signature identity and decisions improve together

Pattern:
- same-option cosine rises strongly;
- signature margin becomes clearly positive;
- relation-binding accuracy/margin improve;
- primary paired both-correct and accuracy improve;
- cross-view decision agreement remains high.

Interpretation:
- relation canonicalization is supported as a missing semantic mechanism.

Only the frozen DEV_READY gate may open sealed confirmation.

### B. Signature identity improves but primary decision remains weak

Pattern:
- cosine >=0.90 and positive signature margin;
- relation binding improves;
- primary triadic decision remains below gate.

Interpretation:
- the representation is canonicalized, but the unchanged primary decision path does not exploit it sufficiently.

A later track may integrate already parameter-free canonical relation evidence into the primary rule. Do not increase capacity by default.

### C. Cosine rises but wrong-option margin stays weak

Pattern:
- same-option cosine increases;
- strongest wrong option also becomes similarly aligned;
- signature margin remains near zero or negative.

Interpretation:
- the objective creates invariance without semantic discrimination.

Do not raise canonicalization coefficient or retune the separation margin on exposed DEV.

### D. Signature invariance does not improve

Pattern:
- same-option cosine remains near A0;
- TRAIN canonicalization loss changes little;
- fresh DEV relation metrics remain weak.

Interpretation:
- the existing A13-LoRA + shared projection does not expose a canonicalizable relation coordinate under this operator.

Close S13 without retrying temperature/coefficient/seed.

### E. TRAIN canonicalization improves but fresh DEV collapses

Pattern:
- TRAIN signature alignment/separation improves strongly;
- fresh DEV cosine or margin stagnates/regresses;
- canonical/paraphrase relation asymmetry remains large.

Interpretation:
- the bottleneck remains lexical/template generalization, not optimization.

No DEV-driven retry is permitted.

### F. Primary decisions improve without signature discrimination

Pattern:
- primary metrics rise;
- signature margin remains weak.

Interpretation:
- retained decision objectives/shared adaptation, not the S13 mechanism, may explain gains.

Do not claim S13 hypothesis success from primary accuracy alone.

## Diagnostic priority

Read metrics in this order:
1. canonical paired both-correct / accuracy;
2. canonical relation-binding accuracy and margin;
3. same-option signature cosine;
4. same-vs-strongest-wrong signature margin;
5. primary cross-view agreement / JS;
6. question-swap choice-change;
7. role/pair concentration diagnostics.

High cosine alone is never semantic success.

## Forbidden post-result actions

After S13 DEV exposure, do not:
- tune canonicalization coefficient;
- tune role/pair/contrastive temperature;
- tune signature margin;
- lower any gate;
- retry seed/LR/templates on exposed DEV;
- reuse S13 DEV rows in the next track;
- reopen Laya/Jev unless `HIRA_V1_S13_CANONICAL_RELATION_DEV_READY` is achieved.
