# HIRA V1 S39 pre-DEV harness repair receipt

Status: **MECHANICAL REPAIR / NO S39 DEV EXPOSURE**

Issue: #261
PR: #262

## Why this repair is protocol-safe

The first S39 TRAIN/DEV authorization marker was committed before the matched workflow branch trigger was corrected.

A later mechanical re-arm created workflow run `37091887607` on head `67ceca865ac67f59bdf4ca5e3cc395d5e064e4f8`.

That run:
- completed setup/download/install;
- failed at **step 7 — Compile and run S39 contracts**;
- failed specifically in `py_compile` with an unterminated string literal in `scripts/hira_v1_s39_train_dev.py`;
- skipped canonical A0 verification;
- skipped M4 integrity;
- skipped **step 10 — Run one fresh matched S39 TRAIN DEV court**;
- produced no S39 TRAIN/DEV artifact.

Therefore no S39 DEV row was evaluated, no selected epoch/checkpoint/metric was exposed, and the repair remains **pre-scientific-exposure**. This is a mechanical abort, not a DEV retry or post-DEV tuning.

## Mechanical defects found

1. Trainer imported a nonexistent A0 module:
   - wrong: `hira_v1_s39_a0_full_bilinear_readout`
   - canonical: `hira_v1_s39_a0_gradient_isolated_bilinear`

2. Matched workflow verifier asserted the S38 seed:
   - wrong: **59001**
   - frozen S39 seed: **60001**

3. The matched workflow branch trigger required correction to:
   - `feat/hira-v1-s39-gradient-isolated-bilinear`

4. Trainer output newline literals are normalized so the script is syntactically valid when directly compiled.

## Methodological clarification added before DEV

The frozen contract keeps the existing selector independently per arm.

Because independent selection may choose different epochs even when the native runtime trajectory is identical, S39 now also reports a **same-epoch counterfactual**:

- take the treatment-selected epoch;
- use the control native DEV metrics from that exact epoch;
- require the control/treatment runtime fingerprint at that epoch to match;
- report treatment minus same-epoch-control as the pure W evaluation effect.

This does not alter:
- data
- seed
- optimizer
- losses
- gradient isolation
- selector
- gates
- model capacity
- training epochs
- batch order

It only prevents interpretation from conflating W effects with a different selected runtime epoch.

## Re-authorization rule

A replacement marker update may be used only after:
- repaired trainer/workflow are frozen;
- regression tests pass;
- exact repaired-head generic CI passes;
- GitHub still shows no prior S39 matched TRAIN/DEV scientific run or artifact.

This is a **pre-exposure mechanical re-arm**, not a second DEV.

The one-DEV scientific stop rule remains unchanged.
