# R8-W5h handoff — balanced anti-collapse token binding

Status: **implementation complete enough for a clean empirical authority run under issue #73. No W5h empirical verdict is valid until the fresh push authority completes.**

## Frozen premise

W5g reached 50.52% fresh untouched accuracy, MRR 0.6402 and K255 accuracy 41.67%, but missed:
- overall accuracy >= 0.60;
- K255 top-5 >= 0.70.

The remaining hypothesis is many-to-one MaxSim collapse: several option tokens can reuse one context token as their strongest evidence.

## Controlled candidates

All candidates have exactly 32,769 trainable parameters:
- bias-free 256->128 projection;
- scalar logit scale.

Only binding differs:
1. `idf-maxsim-proj128` — fresh W5g-style control;
2. `idf-competitive-proj128` — subtract within-option sibling-token common mode per context token before MaxSim;
3. `idf-greedy-unique-proj128` — deterministic unique context assignment by salience before fallback.

IDF salience and final weighted-mean + minimum-coverage aggregation are frozen.

## Fresh authority

TRAIN 512 / DEV 176 / post-selection CONFIRM 192.

Repaired fresh-authority seeds:
- train 141109;
- dev 142211;
- confirm 143313;
- initialization/training 503 (unchanged).

## Protocol repair boundary

Two independent pre-authority audits were completed before accepting any W5h verdict.

1. **Rendered-template freshness repair**
   - head `8ebdcac...` still reused W5g textual scaffold families despite new IDs/domain vocabulary;
   - all empirical runs at or before that boundary are non-authoritative;
   - fresh authority seeds/templates were replaced without changing matcher mechanisms, optimizer budget, selector order or gates.

2. **Untouched-CONFIRM sealing repair**
   - unit tests previously materialized seeded CONFIRM rows only to count them;
   - `generate_binding_authority("confirm")` is now sealed by default;
   - only the post-selection CONFIRM evaluator may open it with explicit capability after selector/hash verification;
   - all empirical runs before `8f5dfc559cb905cc8991fc72e709ec73e9a5d79d` are non-authoritative.

The final pre-authority guard was strengthened at
`e0d2798c28048ee12903e1e8d30aaf9b38f78981`
to reject known rendered scaffold phrases from **W5a through W5g**, not only W5g.

This handoff-only descendant changes no matcher, data generator, optimizer, selector, gate or CONFIRM capability. It is therefore code-identical for the frozen W5h mechanism and is authorized to trigger the clean push authority.

## Authority contract

The only closing sequence is:

`unit -> fresh A13 cache -> 3 frozen candidates -> DEV-only selection freeze -> untouched CONFIRM -> frozen verdict`

PR validation is intentionally unit-only. The full empirical chain is push-only.

CONFIRM requirements:
- generated exactly once after selection freeze;
- seed 143313;
- 192 cases at K=32/64/128/255;
- no post-CONFIRM tuning;
- forbidden benchmark data used = false;
- campaign cells populated = 0;
- state encoded once per case;
- probability mass error <= 1e-6.

## Selection

DEV-only:

accuracy -> K255 top-5 -> K255 accuracy -> K128 accuracy -> overall top-5 -> MRR -> Brier -> earlier epoch.

Selector freezes one candidate and carries the fresh `idf-maxsim-proj128` checkpoint as the same-authority mechanism control.

## Frozen rescue gates

Competence:
- overall >= 0.60;
- K128 >= 0.40;
- K255 >= 0.30;
- K255 top-5 >= 0.70;
- probability error <= 1e-6.

Mechanism:
- overall gain vs fresh IDF-MaxSim control >= +0.05;
- K255 top-5 gain vs fresh IDF-MaxSim control >= +0.05.

Valid verdicts only:
- `BALANCED_BINDING_RESCUE`;
- `BALANCED_BINDING_CONTROL_ALREADY_RESCUES`;
- `BALANCED_BINDING_PARTIAL`;
- `BALANCED_BINDING_FAIL`.

No public campaign cells may be populated by W5h.

## Closure rule

Do not merge PR #74 or close issue #73 from unit/CI success alone.

Close W5h only after:
1. the clean push authority succeeds end-to-end;
2. the selected/control checkpoint hashes and artifacts are preserved;
3. the untouched CONFIRM receipt is recorded;
4. the frozen verdict is written into this handoff and PR;
5. the result is merged without reinterpretation or posthoc threshold changes.
