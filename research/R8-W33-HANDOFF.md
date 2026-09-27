# R8-W33 handoff — shared co-evidence compositional metric

Status: **CLOSED — W33_REFERENCE_QUALIFICATION_FAIL**

Issue: #155  
PR: #156  
Branch: `feat/r8-w33-coevidence-composition`  
Base main: `8be5d068225d364211987d0b73071030e39401c4`

## Why W33 existed

W32 closed with:
- F0 top-1 0.9427083333333334
- F1 top-1 0.953125
- F2 top-1 0.8125
- F2 balanced accuracy 0.7916666666666667
- composed severity 0.7135416666666666
- invalid-vector rate 0.0
- runtime PASS on both sealed domains

That localized the dominant remaining semantic bottleneck to F2 compositional discrimination.

W33 therefore preregistered a **shared co-evidence compositional metric**.

## Candidate hypothesis

Runtime mode:

`coevidence_symmetric_semantic`

Architecture:
- state residual rank-8: 2,048 params
- schema residual rank-8: 2,048 params
- state interaction 128→8: 1,024 params
- schema interaction 128→8: 1,024 params
- state composition 128→8: 1,024 params
- schema composition 128→8: 1,024 params
- total candidate parameters: **8,192**

The composition term uses the second strongest distinct state-token support for each semantic view.

Identity boundary:
- residual up projections start at zero
- state interaction starts at zero
- state composition starts at zero
- initial interaction = 0
- initial co-evidence = 0
- initial candidate logits exactly equal frozen T0

## Frozen optimizer contract

Prepared before qualification result:
- seed 3317
- epochs 20
- batch 32
- AdamW lr 2e-4
- weight decay 0.01
- grad clip 1.0
- temperature 0.07
- anchor coefficient 0.35
- anchor margin threshold 0.08
- invalid-vector mass coefficient 0.15

No TRAIN/DEV marker was ever created.

## Fresh evidence plan

Qualification:
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

All W33 exact text was fresh against exposed W28–W32 evidence.

## Qualification result — frozen

Authoritative run:

`36295893504`

Outcome:

`W33_REFERENCE_QUALIFICATION_FAIL`

Per domain:
- SA: FAIL
- SB: PASS

Isolation:
- case count: 192
- HIRA candidate evaluated: false
- A13 loaded: false
- W32 rows used: false
- W31 rows used: false
- W30 rows used: false
- W29 rows used: false
- older authority rows used: false
- exact-text overlap: none

Artifact:
- `r8-w33-reference-qualification`
- ID: `10924395669`
- digest: `sha256:ce511450337120be77f4d864ee48d0d5fa2029dab785bf20c0b616b6b2cf0ad1`

## Hard-stop consequence

The preregistered rule required **both SA and SB to pass**.

Because SA failed:
- SC–SF TRAIN were not exposed;
- SG DEV was not exposed;
- SH/SI remained sealed;
- A13 candidate execution did not begin;
- the 8,192-param co-evidence candidate was never empirically tested.

Do not report W33 as a candidate failure.

It is an **authority qualification failure**.

## Evidence firewall after closure

SA/SB are permanently exposed and forbidden for:
- training
- DEV selection
- hyperparameter tuning
- candidate choice
- authority tuning
- future candidate comparison

Do not repair W33 by modifying SA wording or lowering thresholds after the fact.

## Next wave

W34 should keep the co-evidence hypothesis as **untested**, but create wholly fresh authority with clearer reference entailment structure.

W34 must retain:
- fresh qualification/TRAIN/DEV/CONFIRM partitions;
- reference-only qualification;
- no A13/candidate load during qualification;
- exact T0 identity;
- no factor-specific or primitive-specific learned head;
- hard stop if either qualification domain fails.

Canonical closure:

`research/R8-W33-CLOSURE.md`
