# HIRA V1 S40 pre-A0 mechanical repair receipt

Status: **MECHANICAL ABORT / NO QUALIFIED A0 SEMANTIC EXPOSURE**

Issue: #263
PR: #264

Attempted A0:
- run `37095497741`
- head `0ef70c08fadb2c16842d4f5f8388ae266e4987a6`

Verified:
- install PASS
- compile/contracts PASS
- M4 integrity PASS
- A0 step entered
- no `HIRA_V1_S40_A0_RECEIPT` emitted
- receipt verification skipped
- integrity freeze skipped
- artifact upload skipped

Exact failure:
`RuntimeError: S40-A0 synthetic signature drift failed`

The failure occurred inside the synthetic mechanical anchor-gradient probe before A0 semantic accuracy was calculated and before any qualified result was emitted.

## Mechanical defect

The synthetic drift probe perturbed only the first LoRA-B tensor by a very small deterministic magnitude.

That perturbation did not produce a measurable native-signature change at the court's numerical precision.

This does not falsify or modify:
- the reference signature anchor
- the projection equation
- W capacity
- joint S38 treatment objective
- future TRAIN/DEV authority
- optimizer/gates

## Frozen repair

The repair:
- perturbs all LoRA-B tensors deterministically;
- uses fixed amplitude **0.02** only for the synthetic A0 drift discriminator;
- requires native signature max-abs drift **> 1e-7**;
- requires signature anchor **> 1e-9**;
- does not use this perturbation in training or DEV.

A replacement A0 is permitted only after exact repaired-head generic CI passes.

This remains pre-qualified-A0 mechanical repair, not post-result tuning.
