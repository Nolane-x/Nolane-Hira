# HIRA V1 S25 A0 receipt — Decoupled Expert Projection Surfaces

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #233  
PR: #234

## Canonical authority

- run `36841998584`
- artifact `11151613971`
- artifact digest `sha256:75390a682778eafafbf9e16dd3271ee004770c06a8bd49fb9dd0d3e12c6a19ee`
- authority head `6f6a99ec5a230981a53b6a311446202c22b5965c`
- outcome `HIRA_V1_S25_A0_DECOUPLED_PROJECTIONS_READY`

Earlier runs:
- `36824997994`: non-result harness abort before receipt/artifact;
- `36840933785`: diagnostic non-result abort exposing frozen-S14 FP32 permutation reduction error.

Neither earlier run is scientific authority.

## Frozen architecture proved

Physical surface:
- shared A13 LoRA: **16,384**
- primary-private 256→128 projection: **32,768**
- relation-private 256→128 projection: **32,768**
- total: **81,920**
- fusion-added parameters: **0**
- frozen A0 runtime trainable params: **0**
- original A13 trainable params: **0**
- HIRACore trainable params: **0**

Initialization:
- shared encoder identity: PASS
- primary initialization identity: PASS
- relation initialization identity: PASS
- both private projections bit-identical to frozen W28 T0: PASS
- private projection storage distinct: PASS
- S14 equal-fusion identity: PASS

Checkpoint:
- dual-projection state ownership roundtrip: PASS
- independent frozen replay: PASS

## Gradient ownership court

Cross-private leakage:
- primary block → relation-private max abs: **0**
- relation block → primary-private max abs: **0**

Own-private gradients:
- primary-private L1: **300.2012329102**
- relation-private L1: **50.0523834229**

Shared-LoRA gradients:
- primary block L1: **9.9481992722**
- relation block L1: **1.7553433180**

Shared neutral-bisector:
- primary norm: **0.2631338835**
- relation norm: **0.0453022644**
- normalized pre-dot: **-0.0190551449**
- normalized post-dot: **-0.0190551449**
- projection coefficient: **0**
- combined norm: **0.1542180777**
- special case: `neutral_bisector`

Thus S25 establishes the intended ownership contract: private surfaces are isolated while both objectives still reach the shared LoRA.

## Full-K / permutation / numerical court

Expert permutation:
- primary logit max abs: **0**
- relation logit max abs: **0**

Frozen S14 fusion:
- mapped option-order choice flip rate: **0**
- FP32 fused logit permutation diagnostic: **4.2915344238e-6**

The nonzero fused logit diagnostic is retained transparently. It arises inside the frozen S14 center/RMS reduction and is not used as a new S25-specific numerical identity claim. Historical S14 authority gates mapped selected-choice permutation, which is exactly preserved.

Other:
- full-K: PASS
- state-once views: **32**
- max probability-mass error: **1.1920929e-7**

## A0 semantic diagnostics

Not model-selection evidence:
- primary canonical: **31.25%**
- primary paraphrase: **43.75%**
- relation canonical: **28.125%**
- relation paraphrase: **46.875%**
- fused canonical: **28.125%**
- fused paraphrase: **46.875%**

These rows exist only to exercise the architecture. No fusion, optimizer, capacity, seed, LR, selector or DEV gate may be tuned from these A0 semantic percentages.

## Consequence

S25-A0 is QUALIFIED.

Fresh S25 TRAIN/DEV may open only after:
1. this receipt is frozen;
2. interpretation plan is frozen;
3. wholly fresh S25 TRAIN/DEV authority generator is frozen;
4. TRAIN/DEV workflow is staged;
5. exact-head generic CI passes;
6. one explicit TRAIN/DEV authorization is committed.

No S25 A0 rerun is authorized.
