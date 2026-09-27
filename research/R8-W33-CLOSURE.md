# R8-W33 closure — shared co-evidence compositional metric

Status: **CLOSED — W33_REFERENCE_QUALIFICATION_FAIL**

Issue: #155  
PR: #156  
Branch: `feat/r8-w33-coevidence-composition`  
Base main: `8be5d068225d364211987d0b73071030e39401c4`

## 1. Purpose

W33 was opened after W32 localized the dominant transfer bottleneck to F2 compositional discrimination.

W32 had already established:
- strong fresh sealed F0/F1 quality;
- zero observed invalid factor vectors;
- healthy state-once/full-K/typed runtime behavior;
- persistent F2 underperformance.

W33 therefore preregistered a compact shared co-evidence operator intended to reward multiple distinct semantic evidence pieces without using factor-specific heads.

## 2. Frozen candidate hypothesis

Candidate runtime mode:

`coevidence_symmetric_semantic`

Candidate architecture:
- state residual rank-8 adapter: 2,048 params;
- schema residual rank-8 adapter: 2,048 params;
- state interaction 128→8: 1,024 params;
- schema interaction 128→8: 1,024 params;
- state composition 128→8: 1,024 params;
- schema composition 128→8: 1,024 params;
- total trainable candidate parameters: **8,192**.

The added composition operator uses the second strongest distinct state-token support for each semantic view.

Identity contract:
- residual up projections start at zero;
- state interaction starts at zero;
- state composition starts at zero;
- initial interaction contribution = 0;
- initial co-evidence contribution = 0;
- initial W33 logits exactly equal frozen unbridged T0.

## 3. Fresh authority

W33 authority partitions were preregistered as:

Reference qualification:
- SA
- SB

TRAIN:
- SC
- SD
- SE
- SF

DEV:
- SG

SEALED CONFIRM:
- SH
- SI

All contexts/style banks/text were fresh against W28/W29/W30/W31/W32.

## 4. Reference qualification

Authoritative run:

`36295893504`

Frozen outcome:

`W33_REFERENCE_QUALIFICATION_FAIL`

Per-domain:
- SA: FAIL
- SB: PASS

Case count:
- 192 reference-only cases

Qualification isolation:
- HIRA candidate evaluated: false
- A13 loaded: false
- W32 rows used: false
- W31 rows used: false
- W30 rows used: false
- W29 rows used: false
- older authority rows used: false
- exact-text overlap with exposed authorities: none

Frozen qualification artifact:
- name: `r8-w33-reference-qualification`
- artifact ID: `10924395669`
- digest: `sha256:ce511450337120be77f4d864ee48d0d5fa2029dab785bf20c0b616b6b2cf0ad1`

## 5. Why W33 stopped

The preregistered qualification rule required both SA and SB to pass before any candidate TRAIN/DEV exposure.

SA failed.

Therefore:
- SC/SD/SE/SF were never authorized for candidate training;
- SG was never authorized for candidate selection;
- SH/SI remained sealed;
- A13 candidate evaluation never began;
- the 8,192-parameter co-evidence hypothesis was **not empirically tested**.

This is an authority failure, not a candidate-quality failure.

## 6. Scientific interpretation

W33 does **not** establish that the co-evidence hypothesis is weak.

It establishes only that the first fresh SA/SB reference authority set was not strong enough to serve as a clean scientific judge under the frozen NLI reference panel.

The correct response is not:
- tune against SA;
- weaken the qualification threshold;
- train anyway;
- reuse SB plus exposed SA.

The correct response is to reject this authority instance and construct a wholly fresh next-wave authority before testing the candidate hypothesis.

## 7. Evidence boundary

SA/SB are permanently exposed.

They are forbidden for all future:
- training;
- DEV selection;
- hyperparameter tuning;
- candidate choice;
- authority repair/tuning;
- architecture ranking.

SC–SI were not candidate-exposed in W33, but W34 must still use wholly fresh authority identifiers/text rather than inheriting W33 partitions.

## 8. Next-wave direction

W34 should preserve the W33 co-evidence architectural hypothesis as an **untested** candidate direction, while rebuilding reference authority from fresh domains with clearer semantic entailment structure.

W34 should:
- keep exact-T0 identity;
- keep 8,192-parameter shared candidate capacity unless preregistered otherwise;
- keep no factor-specific/primitive-specific heads;
- use fresh qualification/TRAIN/DEV/CONFIRM domains;
- keep the same hard stop if either reference qualification domain fails.

No W33 result authorizes HIRA-v0 transfer-core promotion.

`HIRA_V0_TRANSFER_CORE_READY` remains blocked.
