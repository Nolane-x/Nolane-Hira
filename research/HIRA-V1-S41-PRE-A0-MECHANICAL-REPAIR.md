# HIRA V1 S41 pre-qualified A0 mechanical repair receipt

Status: **MECHANICAL REPAIR / NO QUALIFIED S41-A0 RECEIPT**

Issue: #265
PR: #266

## Failed attempted A0

Run:
- `37098846970`
- head `65f5a9ff1bc43d989ed028df22776cff8f50231f`

Passed:
- checkout
- M4 artifact download
- exact stack install
- compile/contracts
- frozen M4 integrity

Failed:
- step 8 — `Run S41 A0 optimizer-step court`
- guard: `S41-A0 AdamW candidate/state parity failed`

Skipped:
- receipt verification
- integrity freeze
- artifact upload

Therefore:
- no `HIRA_V1_S41_A0_RECEIPT` was emitted;
- no S41-A0 artifact exists;
- no qualified A0 semantic result exists;
- no TRAIN/DEV authority exists.

## Mechanical cause

S41 requires the candidate update to match **standard PyTorch AdamW exactly**.

The first engine manually reproduced AdamW arithmetic. Its small synthetic unit tests passed, but the live A13 trainable tensors exposed a bitwise parity mismatch.

The scientific mechanism is not "approximately AdamW"; exact optimizer-faithful movement is part of the contract.

## Repair

The repaired `adamw_candidate_deltas` now delegates the stateful candidate transition to **`torch.optim.AdamW` itself on shadow copies of only the trainable tensors**.

Properties:
- real treatment parameters are not mutated while deriving the candidate;
- candidate parameter delta comes from standard PyTorch AdamW;
- next first/second moments and step counter come from that same optimizer transition;
- configured LR/betas/eps/weight decay/foreach/fused semantics remain frozen;
- the downstream anchor projection is unchanged;
- W candidate delta remains outside the runtime anchor projection.

No semantic threshold, data, seed, capacity, anchor target, projection equation, or selector changed.

## Re-arm rule

A replacement S41-A0 run is allowed only after:
- repaired engine tests pass;
- exact repaired-head generic CI passes;
- the A0 marker is updated with this mechanical-abort provenance.

A qualified replacement A0 will again be one diagnostic court, not a post-result retry.
