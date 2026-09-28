# HIRA V1 S0 -> S1 handoff

Status: **S0 CLOSED — S1 AUTHORIZED AS A FRESH TRACK**

Canonical S0 closure:

`research/HIRA-V1-S0-CLOSURE.md`

## Frozen S0 authorities

S0-A question-blindness:
- run `36382301792`
- artifact `10952694461`
- digest `sha256:f96fa93dfd2f640008a1ac3cc9ce40129cda9d57bbf062c6616a7b640c87a812`
- outcome `HIRA_V1_S0_QUESTION_BLINDNESS_CONFIRMED`

S0-B0 parameter-free:
- run `36382895869`
- artifact `10952918278`
- digest `sha256:e98d93b1db860a4e7f8a867c0789ced676f74f357fff90c964b0fb5b4daa0af9`
- outcome `HIRA_V1_S0_PARAMETER_FREE_QUERY_BASELINE_READY`

S0-C TRAIN/DEV:
- run `36383487080`
- artifact `10953344707`
- digest `sha256:bf3c2e14dd6f08927bd5f48396c99e897cd85b6bb8b19ea75d6f2f39803972b7`
- outcome `HIRA_V1_S0_QUERY_REPAIR_DEV_FAIL`
- selected epoch 8
- selected checkpoint SHA256 `1c00d903d8fda91fd3f905eaed2dac21dbd9e49d90422e89fb1a9b01b70575d5`

## S0 aggregate result

The v0 question-blindness diagnosis is confirmed.

However:
- fixed query relevance: changes logits, no choice movement;
- learned multiplicative query gate: insufficient fresh DEV generalization;
- selected DEV accuracy: 0.3828125;
- paired both-correct: 0.09375;
- VI accuracy: 0.265625;
- order invariance/runtime mechanics remain healthy.

## S1 architectural direction

Do not tune QCCE on S0 DEV.

S1 should test a fresh hypothesis:

# Query-Keyed Evidence Extraction (QKEE)

Core idea:

Instead of:
`score(state, option) * independent_question_relevance`

use:
1. question tokens as keys/queries over state tokens;
2. extract a query-specific evidence representation from the state;
3. preserve multiple evidence pieces where needed;
4. score option semantic views against that query-specific evidence;
5. optionally retain W34 co-evidence as a residual path.

The question must therefore change **which state evidence exists for scoring**, not merely rescale an already-built state-option similarity matrix.

## S1 constraints

Preserve:
- frozen Hira v0 evidence;
- exact W28/W34 provenance as a baseline;
- state encoded once;
- question schemas cacheable;
- dynamic K/full-K;
- opaque IDs;
- permutation equivariance;
- no dataset-specific heads;
- compact parameter budget.

Fresh authority required:
- no S0 TRAIN or DEV rows;
- no S0-A localization rows;
- no M5 final/confirmatory rows;
- no W29-W34 sealed rows.

S1 must preregister TRAIN/DEV and gates before any candidate score exposure.

Because S0 shows VI remains weak, S1 should separately distinguish:
- query-routing failure;
- semantic-front-end multilingual failure.

Do not treat the two as the same mechanism.
