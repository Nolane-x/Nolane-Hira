# R8-W32 handoff — dual residual adapter + shared interaction metric

Status: **CLOSED — W32_INTERACTION_ADAPTER_FAIL**

Issue: #153  
PR: #154  
Branch: `feat/r8-w32-interaction-semantic-adapter`  
Base main: `762289911da2b7cde860383268c5d35d6ebd2f98`

## Objective

W32 tested whether W31's successful state/schema asymmetric transfer geometry could be strengthened by a shared low-rank cross-interaction metric and a structural monotone-vector regularizer.

The frozen production candidate remained compact and state-once:

- A13 encoder frozen;
- exact W28 T0 projection frozen;
- state residual rank-8 adapter: 2,048 params;
- schema residual rank-8 adapter: 2,048 params;
- state interaction 128→8: 1,024 params;
- schema interaction 128→8: 1,024 params;
- total trainable candidate parameters: **6,144**;
- no factor-specific learned head;
- no primitive-specific learned head;
- relation refinement OFF;
- full-K typed runtime.

Runtime mode:

`interaction_symmetric_semantic`

## Candidate geometry

```
t_s = T0(state)
t_o = T0(schema)

z_s = normalize(t_s + U_s(D_s(t_s)))
z_o = normalize(t_o + U_o(D_o(t_o)))

r_s = M_s(z_s)
r_o = M_o(z_o)

interaction(o,s) = <r_o, r_s> / sqrt(8)
similarity(o,s)  = <z_o, z_s> + interaction(o,s)
```

Identity initialization:

- both residual up projections start at exact zero;
- state interaction map starts at exact zero;
- initial interaction contribution is exactly zero;
- initial W32 logits exactly equal frozen unbridged T0.

## Training contract

Frozen before exposure:

- seed: 3217
- epochs: 18
- logical batch: 32
- AdamW lr: 2e-4
- weight decay: 0.01
- grad clip: 1.0
- semantic temperature: 0.07
- anchor coefficient: 0.35
- positive frozen-T0 anchor margin threshold: 0.08
- invalid-vector probability-mass coefficient: 0.15

Primary loss:

- factor-balanced F0/F1/F2 CE.

Anchor loss:

- preserves frozen-T0 correct decisions whose signed margin is >= 0.08.

Structural validity loss:

- valid factor vectors: 000, 100, 110, 111;
- invalid vectors: 001, 010, 011, 101;
- penalizes probability mass assigned to invalid vectors;
- introduces no trainable factor head.

DEV selection order:

1. worst-factor balanced accuracy;
2. worst-factor top-1;
3. composed severity;
4. lower invalid-vector rate;
5. earlier epoch.

## Fresh evidence partitions

Reference qualification:

- RA
- RB

TRAIN:

- RC
- RD
- RE
- RF

DEV:

- RG

SEALED CONFIRM:

- RH
- RI

Every partition uses distinct contexts/styles and exact W32 text is disjoint from W28/W29/W30/W31 authority text.

## Reference qualification — frozen

Run:

`36292606367`

Outcome:

`W32_REFERENCE_QUALIFIED`

- RA: PASS
- RB: PASS
- case count: 192
- HIRA candidate evaluated: false
- A13 loaded: false
- prior-wave authority rows used: false
- exact-text overlap: none

Artifact:

- `r8-w32-reference-qualification`
- ID: `10922738458`
- digest: `sha256:02cafde04b10c075df848d3c0d45b9f9c139b3ae1516a82c2a428e2163131751`

## TRAIN/DEV — frozen

Run:

`36293558503`

- TRAIN RC/RD/RE/RF: 384 cases
- DEV RG: 96 cases
- selected epoch: 17
- checkpoint SHA256:
  `212caf6d5cdb06743979da5d7b64fff5887fc5c2b11e1be91006bc6463e63a5f`
- anchor rate: 0.7100694444444444
- confirm rows used during training/selection: 0

Training artifact:

- `r8-w32-interaction-training`
- ID: `10923630412`
- digest: `sha256:4b8289345f92f7fb45b5af1f4c2e6d2765ca56b4e1e52dffc62968f26b868f92`

### RG baseline

- F0 top-1: 0.7916666666666666
- F1 top-1: 0.875
- F2 top-1: 0.7604166666666666
- vector/severity: 0.5104166666666666
- invalid-vector rate: 0.08333333333333333

### RG W32

- F0 top-1: 0.9166666666666666
- F1 top-1: 0.96875
- F2 top-1: 0.8020833333333334
- vector/severity: 0.6875
- invalid-vector rate: 0.0
- probability mass error: 1.1920928955078125e-07

## SEALED CONFIRM — frozen

First and only RH/RI sealed run:

`36294714414`

Outcome:

`W32_INTERACTION_ADAPTER_FAIL`

- RH runtime: PASS
- RI runtime: PASS
- RH absolute quality: FAIL
- RI absolute quality: FAIL
- candidate truncation: false
- relation refinement: false
- state-once: preserved
- typed primitive agreement: preserved
- option-order invariance: preserved
- confirm used for fitting/selection: false

Audit artifact:

- `r8-w32-authoritative-audit`
- ID: `10923651937`
- digest: `sha256:b7b9f078a36af924e2a3a409756df5a2b66a7d80e038890b792434244c50340a`

### Pooled RH+RI baseline

- F0 top-1: 0.796875
- F1 top-1: 0.8697916666666666
- F2 top-1: 0.765625
- F0 BA: 0.8645833333333333
- F1 BA: 0.8697916666666667
- F2 BA: 0.5868055555555556
- vector/severity: 0.515625
- invalid-vector rate: 0.08333333333333333

### Pooled RH+RI W32

- F0 top-1: 0.9427083333333334
- F1 top-1: 0.953125
- F2 top-1: 0.8125
- F0 BA: 0.9409722222222222
- F1 BA: 0.953125
- F2 BA: 0.7916666666666667
- vector/severity: 0.7135416666666666
- invalid-vector rate: 0.0
- probability mass error: 1.1920928955078125e-07

## Frozen promotion verdict

Absolute production gate failed on both sealed domains.

Pooled transfer observations:

- composed severity delta: +0.19791666666666663
- worst-factor top-1 delta: +0.046875
- F0 improvement: +0.14583333333333337
- F1 improvement: +0.08333333333333337
- F2 improvement: +0.046875
- no factor regressed

The severity transfer requirement passed.

The worst-factor improvement requirement failed:

- required >= +0.08
- observed +0.046875

The remaining dominant bottleneck is F2.

`HIRA_V0_TRANSFER_CORE_READY` is **not authorized**.

## Scientific conclusion

W32 establishes that:

- compact shared interaction helps;
- F0/F1 semantic discrimination is now strong on fresh sealed evidence;
- structural invalid-vector loss can drive observed invalid vectors to zero;
- runtime integrity is no longer a meaningful blocker;
- generic interaction plus vector legality is still insufficient for F2 compositional semantics.

The next wave must target the composition mechanism itself, not simply add another generic transfer bridge.

## Evidence firewall after closure

RH/RI are permanently exposed.

They are forbidden for all future:

- training;
- DEV selection;
- hyperparameter tuning;
- architecture selection;
- candidate ranking.

The W32 checkpoint must never be retuned against RH/RI.

Prior aggregate results may only motivate a fresh hypothesis.

## Recommended next wave

W33 should preserve the successful W32 base but test a shared compact compositional latent operator that can represent conjunction-like evidence interactions before final semantic scoring.

Constraints remain:

- fresh qualification/TRAIN/DEV/CONFIRM evidence;
- no W32 checkpoint tuning on RH/RI;
- no factor-specific classifier;
- no primitive-specific learned head;
- exact T0 identity initialization;
- compact state-once/full-K runtime.

See `research/R8-W32-CLOSURE.md` for the canonical closure.
