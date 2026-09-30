# HIRA V1 S21 interpretation plan — frozen before TRAIN/DEV

Status: **FROZEN BEFORE FRESH DEV EXPOSURE**

Issue: #225  
PR: #226

## Fixed hypothesis

S21 tests one structural claim only:

> Explicitly factorizing queried semantic role from bound content inside the **primary** scorer can generalize better than the coordinate-mixed S17 primary triad, while preserving S17's strong relation expert and norm-balanced optimization.

No new learned capacity is added.

## Fixed scientific comparison

Primary reference is **S17**, the strongest fresh DEV frontier.

S17 selected:
- fused canonical: **0.7213541667**
- fused paraphrase: **0.5546875**
- paired both-correct: **0.5104166667**
- question-swap: **0.984375**
- fused cross-view agreement: **0.5286458333**
- fused canonical margin: **+0.3138313380**
- raw primary canonical: **0.5911458333**
- raw primary paraphrase: **0.4427083333**
- relation canonical: **0.640625**
- relation paraphrase: **0.6380208333**
- relation canonical margin: **+0.2073315941**

S18-S20 are failed side tracks and are not the performance baseline.

## Frozen S21 configuration

- seed: **36001**
- 24 epochs
- batch size: **16 semantic cases**
- AdamW lr: **2e-4**
- weight decay: **0.01**
- grad clip: **1.0**
- exact physical trainable params: **49,152**
- role temperature: **0.10**
- role/content weights: **0.50 / 0.50**
- S17 fused-output symmetric JS coefficient: **0.25**
- swap coefficient/margin: **0.25 / 0.20**
- option alignment: **0.05**, temperature **0.10**
- relation CE: **0.10**
- signature canonicalization: **0.15**
- signature separation margin: **0.20**
- S17 norm-balanced shared-gradient rule unchanged

No S18/S19/S20 intervention is active.

## Preregistered interpretations

### Outcome A — structural factorization succeeds

Evidence:
- S21 materially improves fresh fused/paired performance relative to S17, or reaches DEV_READY;
- primary canonical/paraphrase improve without destroying relation quality;
- question-swap remains >=0.80;
- semantic margins become positive and stable.

Interpretation:
- coordinate-mixed primary triad was a material bottleneck;
- role/content separation is useful architecture evidence.

Only DEV_READY can open sealed confirmation.

### Outcome B — primary improves, fusion/relation regresses

Evidence:
- S21 primary metrics improve versus S17 raw primary;
- relation canonical/margin or fused metrics regress.

Interpretation:
- factorized primary geometry is useful but incompatible with the shared projection / relation expert under current joint optimization.

Next work should isolate expert geometry or shared-surface coupling, not tune role weights after DEV.

### Outcome C — role gating sharpens but fresh semantics do not improve

Evidence:
- TRAIN objective learns and role diagnostics become concentrated;
- fresh primary/fused accuracy remains below S17;
- role/content synthetic mechanism remains valid.

Interpretation:
- role selection itself is learnable but lexical/domain transfer remains weak;
- explicit role gating may overfit surface field cues.

Do not respond by changing role temperature after exposure.

### Outcome D — relation expert remains S17-like, primary factorization fails

Evidence:
- relation canonical/paraphrase stay near S17;
- S21 primary/fused metrics fall materially.

Interpretation:
- new primary operator discards useful semantic interaction contained in the coordinate-mixed triad.

Next track should restore S17 primary and investigate representation/capacity, not further modify this factorization formula.

### Outcome E — global regression

Evidence:
- primary, relation and fused metrics all regress;
- gradient balance may show strong conflict or destructive coupling.

Interpretation:
- changing primary inference altered the shared projection learning geometry and damaged both experts.

Return to S17 frontier. Do not tune 0.5/0.5 weights.

## DEV_READY

Use only the gates frozen in `HIRA-V1-S21-CONTRACT.md`.

Scientific FAIL is valid.

No post-DEV:
- role-temperature tuning
- role/content-weight tuning
- operator-formula changes
- seed/LR/template retry
- gate weakening
- reuse of S21 DEV rows

No Laya/Jev, sealed English confirmation or multilingual transfer unless DEV_READY.
