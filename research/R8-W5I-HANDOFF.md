# R8-W5i handoff — cross-candidate evidence intersection

Status: **CLOSED. Authoritative verdict: `CROSS_CANDIDATE_CONTROL_ALREADY_RESCUES`. PR #76 merged into `main` as `eee76db1cda347b0c615ef44679a504d50ab2a62`.**

## Frozen premise

W5h closed as `BALANCED_BINDING_PARTIAL`:
- accuracy 47.40%;
- MRR 0.6500;
- top-5 86.98%;
- K128 accuracy 43.75%;
- K255 accuracy 27.08%;
- K255 top-5 70.83%.

The W5h mechanism control for W5i is soft within-option competitive binding.

W5i asks whether remaining top-1 ambiguity is caused by scoring each option independently after within-option competition.

## Frozen candidates

All candidates have exactly 32,769 trainable parameters and share:
- frozen A13;
- IDF salience;
- W5h competitive adjusted similarity;
- W5h forward aggregate;
- CE+Brier objective;
- six-epoch budget.

Candidates:
1. `idf-competitive-forward-proj128` — fresh W5h control.
2. `idf-competitive-bidir-mean-proj128` — add context->candidate residual against arithmetic candidate mean.
3. `idf-competitive-bidir-logmeanexp-proj128` — add context->candidate residual against log-mean-exp candidate baseline.

No extra learned coefficient or temperature.

## Fresh authority

TRAIN 512 / DEV 176 / post-selection CONFIRM 192.

Seeds:
- train 151117;
- dev 152219;
- confirm 153321;
- init/training 607.

Vocabulary, rendered templates, IDs and seeds must be disjoint from W5a-W5h.

CONFIRM must remain sealed until DEV selection/checkpoint hashes freeze.

## Frozen gates

Absolute competence:
- overall >= 0.60;
- K128 >= 0.40;
- K255 >= 0.30;
- K255 top-5 >= 0.70;
- probability error <= 1e-6.

Mechanism:
- overall gain vs fresh forward control >= +0.05;
- K255 accuracy gain vs fresh forward control >= +0.03.

Valid verdicts:
- `CROSS_CANDIDATE_BINDING_RESCUE`;
- `CROSS_CANDIDATE_CONTROL_ALREADY_RESCUES`;
- `CROSS_CANDIDATE_BINDING_PARTIAL`;
- `CROSS_CANDIDATE_BINDING_FAIL`.

Campaign cells populated = 0.

## Current implementation boundary

Before adding a push-only empirical workflow:
1. compile the W5i module/scripts;
2. pass W5i unit contracts;
3. prove vocabulary/template freshness against W5a-W5h;
4. audit CONFIRM sealing;
5. then create one clean push-authority lineage.

No empirical authority is accepted from a branch state before these checks pass.


## Authoritative closure

Sole authoritative push:
- run `35884130186`;
- exact pre-merge head `acd66abcd60d287058c24ad84460dc0ba9770a6a`;
- unit -> fresh cache -> 3 candidates -> DEV freeze -> untouched CONFIRM: all PASS;
- CONFIRM generated after selection freeze: true;
- forbidden benchmark data used: false;
- campaign cells populated: 0.

DEV selected:
- `idf-competitive-bidir-logmeanexp-proj128`;
- epoch 6;
- 32,769 trainable parameters;
- matcher SHA-256 `a8e57662aeae877c29da26e99485939b28c0d9d527f8e7400920f35367334db8`;
- DEV accuracy 96.59%;
- DEV K255 accuracy 100%.

Fresh same-authority forward control:
- `idf-competitive-forward-proj128`;
- epoch 4;
- matcher SHA-256 `5ce9cdceb5a87c8b18b8d50e68396f23184e1dfb9acad338f25f24277d438a0d`;
- DEV accuracy 92.61%;
- DEV K255 accuracy 95.83%.

Untouched CONFIRM, 192 cases:

Selected listwise:
- overall accuracy **84.90%**;
- MRR **0.9177**;
- top-5 **100%**;
- K32 accuracy **87.50%**;
- K64 accuracy **83.33%**;
- K128 accuracy **91.67%**;
- K255 accuracy **77.08%**;
- K255 top-5 **100%**;
- probability max error **2.384e-7**;
- state encodes/case **1.0**.

Fresh forward competitive control:
- overall accuracy **82.29%**;
- MRR **0.8844**;
- top-5 **96.35%**;
- K32 accuracy **85.42%**;
- K64 accuracy **79.17%**;
- K128 accuracy **85.42%**;
- K255 accuracy **79.17%**;
- K255 top-5 **91.67%**;
- probability max error **2.384e-7**.

Selected minus control:
- overall accuracy **+2.60 pp**;
- MRR **+0.0333**;
- top-5 **+3.65 pp**;
- K128 accuracy **+6.25 pp**;
- K255 accuracy **-2.08 pp**;
- K255 top-5 **+8.33 pp**.

Pooled frozen A13:
- overall accuracy **0%**;
- top-5 **1.56%**.

Frozen absolute competence gates pass for both selected and control.

The new W5i listwise mechanism misses both attribution gates:
- overall gain >= +0.05: FAIL;
- K255 accuracy gain >= +0.03: FAIL.

Therefore the authoritative verdict is:
**`CROSS_CANDIDATE_CONTROL_ALREADY_RESCUES`**.

Scientific interpretation:
- W5i reverse/listwise evidence is not promoted;
- forward competitive binding has now crossed the frozen rescue boundary on a second fresh authority;
- the result is stronger evidence for promoting the W5h mechanism into HIRA's actual state-once path than for continuing synthetic semantic scorer search;
- W5h and W5i absolute percentages come from different authorities and should not be treated as a direct paired comparison.

Preserved artifacts:
- cache `10762975228` / `sha256:d8918c1c044bf191b5165411acb5f7568eaa76a610089c8d07b82910dfd5f3c5`;
- forward candidate `10764240783` / `sha256:f3c0ff451c976caff10b8098b010abc7b8758d4c4797110c4cce18943edb4f74`;
- bidir mean `10763881488` / `sha256:88f9a76a52a1689251532f72e5f4701c88391c1666e41108b6c13920478feaa5`;
- bidir logmeanexp `10763956381` / `sha256:6eb806872f87f3bf02d3251fd531da8314ea54fe26bfc3e93ff717dde0675582`;
- selected `10763826483` / `sha256:3b7f3bb6d70156f6ba3ea8c6acd7edc1c0462aa1947354371e5b0a371da801c6`;
- confirm `10763947194` / `sha256:4a1c8c64a52255b75d552a696697d43e00a72bcff776a40b6c64c26e38c0845d`.

## Next research boundary

Do not create W5j as another synthetic semantic scorer.

Promote only the fresh-control mechanism:
`idf-competitive-forward-proj128`.

The next lane must:
1. locate HIRA's actual state-once/coarse decision path;
2. integrate competitive token binding without re-encoding state per option;
3. preserve deterministic fallback and current safety/reliability contracts;
4. test fresh typed/reliability authorities before any public campaign cells;
5. keep the W5i reverse/listwise additions as negative/diagnostic evidence only.
