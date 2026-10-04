# HIRA V1 S52 pre-DEV mechanical repair receipt

Status: **MECHANICAL REPAIR / DEV STILL UNAUTHORIZED**

Issue: #287  
PR: #288

## Failure caught before DEV

Intermediate CI runs:
- `37199468020`
- `37199542775`

Failure:
the S52 private-state roundtrip test expected checkpoint keys
`adapter_a / adapter_b / bilinear_weight`.

The inherited frozen S44 correction checkpoint contract actually uses:
- `adapter.a`
- `adapter.b`
- `bilinear.weight`.

This mismatch affected only S52 checkpoint persistence/loading mechanics.

## Repair

S52 private state now extends the exact inherited S44 key contract:

- `adapter.a`
- `adapter.b`
- `bilinear.weight`
- `query_canonicalizer.adapter_a`
- `query_canonicalizer.adapter_b`

No changes to:
- query canonicalizer equations
- canonicalizer parameter count
- correction parameter count
- auxiliary coefficient
- separation ceiling
- data
- seed
- optimizer
- loss
- selector
- native authority
- cache
- DEV gates.

DEV exposure remains **0**.

## Governance

The old pre-DEV staging head `3623c138df2ce8d733ab0ad681f8a59b4a49e7ec` is superseded for authorization purposes.

A new exact final staging head after this repair MUST pass generic CI on Python 3.10 and 3.12 before `HIRA-V1-S52-ENABLE-TRAIN-DEV` may exist.
