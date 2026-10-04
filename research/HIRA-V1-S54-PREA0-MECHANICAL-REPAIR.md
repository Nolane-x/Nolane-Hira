# HIRA V1 S54 pre-A0 mechanical repair receipt

Status: **MECHANICAL REPAIR / A0 STILL UNAUTHORIZED**

Issue: #291  
PR: #292

No S54-A0 or S54 DEV exposure has occurred.

## Repair 1 — fresh authority seed

S53 already used seed **74001**.

The initial S54 preregistration draft accidentally repeated 74001.

Before any S54 scientific exposure, S54 was corrected to seed **75001**.

No data, result, metric, or model output existed before this correction.

## Repair 2 — masked-input rejection test ownership

Early core CI:
- run `37207206749`
- failed on both Python 3.10 and 3.12.

The S54 test expected the all-masked state path to raise an S54-specific message containing:
`active state token`.

The inherited S49 query-free identity validator correctly rejects the same invalid input earlier with:
`S49 requires state content`.

The scientific invariant is **rejection of all-masked state/query/option inputs**, not ownership of the error string.

Repair:
- tests now require `ValueError` rejection;
- no S54 operator equation changed;
- no mask semantics changed;
- no capacity/temperature/data/seed changed.

## Governance

A0 remains marker-gated.

A new exact final repaired head MUST pass generic CI on Python 3.10 and 3.12 before:
`research/HIRA-V1-S54-ENABLE-A0`
may exist.
