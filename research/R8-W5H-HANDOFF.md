# R8-W5h handoff — balanced anti-collapse token binding

Status: **CLOSED. Authoritative W5h verdict: `BALANCED_BINDING_PARTIAL`. PR #74 merged into `main` as `95eee0c40c41995d658d7b43e5c9c7d2b437716d`.**

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

The final template-freshness guard was strengthened at
`e0d2798c28048ee12903e1e8d30aaf9b38f78981`
to reject known rendered scaffold phrases from **W5a through W5g**, not only W5g.

3. **Verdict-attribution repair**
   - pre-repair logic could emit `BALANCED_BINDING_CONTROL_ALREADY_RESCUES` whenever the selected candidate crossed absolute competence but mechanism deltas failed, even if the fresh control itself remained below the absolute competence boundary;
   - this is a labeling/interpretation bug, not a matcher or data change;
   - the repaired rule emits `CONTROL_ALREADY_RESCUES` only when the control independently satisfies all absolute competence gates;
   - if selected crosses competence but control does not and mechanism attribution is insufficient, the verdict is conservatively `BALANCED_BINDING_PARTIAL`;
   - authority run #31 from `a986dbb8...` is therefore non-authoritative and must not supply a verdict. The next code descendant supersedes it before untouched CONFIRM exposure.

The repaired descendant changes no matcher, data generator, optimizer, selector ordering, threshold, or CONFIRM capability. It repairs only verdict attribution and adds regression coverage.

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


## Authoritative closure

Sole authoritative push run:
- run `35875847461`;
- exact pre-merge head `ddf6b9213b8c7fa00f087ad448ce8f8c69c9e330`;
- unit -> cache -> 3 candidates -> DEV freeze -> untouched CONFIRM: all PASS;
- CONFIRM generated after selection freeze: true;
- forbidden benchmark data used: false;
- campaign cells populated: 0.

Selected candidate:
- `idf-competitive-proj128`;
- epoch 5;
- 32,769 trainable parameters;
- matcher SHA-256 `13f06a272db8d1a0ab1decf863d6e9b43fabaed92db1b036accd063c4e7c917d`.

Fresh same-authority control:
- `idf-maxsim-proj128`;
- epoch 6;
- matcher SHA-256 `a221bdba00f20a4716b804225a83dc2b1dd18a4e1e9634791642e524a9d9f182`.

Untouched CONFIRM, 192 cases:
- selected accuracy 0.473958 = **47.40%**;
- selected MRR **0.649996**;
- selected top-5 **86.98%**;
- K32 accuracy **62.50%**;
- K64 accuracy **56.25%**;
- K128 accuracy **43.75%**;
- K255 accuracy **27.08%**;
- K255 top-5 **70.83%**;
- probability-mass max error **2.384e-7**;
- state text encodes/case **1.0**.

Fresh control:
- accuracy **7.29%**;
- MRR **0.213872**;
- top-5 **32.29%**;
- K128 accuracy **4.17%**;
- K255 accuracy **2.08%**;
- K255 top-5 **14.58%**.

Same-authority causal deltas:
- accuracy **+40.10 percentage points**;
- MRR **+0.4361**;
- top-5 **+54.69 points**;
- K128 accuracy **+39.58 points**;
- K255 accuracy **+25.00 points**;
- K255 top-5 **+56.25 points**.

Frozen gates passed:
- K128 accuracy >= 0.40;
- K255 top-5 >= 0.70;
- probability integrity;
- overall gain vs control >= +0.05;
- K255 top-5 gain vs control >= +0.05.

Frozen gates missed:
- overall accuracy >= 0.60;
- K255 accuracy >= 0.30.

Therefore the frozen verdict is **`BALANCED_BINDING_PARTIAL`**, not RESCUE.

Scientific interpretation:
- soft within-option competition is a real mechanism;
- hard greedy uniqueness is not the answer under this authority;
- the remaining failure is dominated by top-1 ordering among already-near-top candidates, not broad semantic collapse;
- W5g and W5h absolute percentages come from different fresh authorities and are not a direct regression comparison.

Preserved artifacts:
- cache `10759175733` / `sha256:c0ee87ad8e046beabe413762d6e7a19370d803182572b68d0194ec3ad4a9e541`;
- maxsim candidate `10757629158` / `sha256:2d248058c8814f6dd3ccd27b967a3126a3bcfe0369bf38558e8c64e13154be06`;
- competitive candidate `10758896154` / `sha256:7935028093d487a46fc1e57e4f5043c24bc4af393b2b44518cb951ef0b0dfb7a`;
- greedy-unique candidate `10759196799` / `sha256:fb297c78c436dc76e8527de20a42dff5d04d2d062a6a63a39c3011ba8ffb138c`;
- selected `10758957261` / `sha256:7b330b51ad43544ef21e2b88c3abeb8af1961923909b7c567c6647f74337af6a`;
- confirm `10757949984` / `sha256:dcc978256af7b690435d85e22ce9f4e8e20458bb004f5423fb85817a188c8936`.

## Next research boundary

Do not retune W5h after CONFIRM exposure.

The next fresh lane must preserve:
- exact frozen A13;
- state-once execution;
- candidate-relative IDF salience;
- soft competitive binding as the new mechanism control;
- no public campaign cells.

The next hypothesis should target **cross-candidate top-1 evidence intersection / rival-margin ranking**, because W5h already places the gold in top-5 at high rate while failing to rank it first often enough.
