# HIRA V1 S30 interpretation plan — frozen before S30-A0/DEV exposure

Status: **FROZEN**

Issue: #243
PR: #244

## Controlled question

S30 compares two final-block encoder adaptation surfaces on the same fresh authority:

- Arm A: attention-only LoRA + shared projection
- Arm B: FFN-only LoRA + shared projection

Everything else remains the S17 semantic shell.

## Frozen arm capacities

Arm A:
- attention LoRA **16,384**
- projection **32,768**
- total **49,152**

Arm B:
- FFN LoRA **20,480**
- projection **32,768**
- total **53,248**

Both use rank 8 / alpha 8 / dropout 0.

## Frozen matched TRAIN/DEV

- seed **51001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S30 domains
- K=4
- two state views
- two question views per semantic query
- two semantic option views
- 24 epochs
- batch 16
- AdamW lr 2e-4
- weight decay 0.01
- grad clip 1.0
- identical row order per epoch across arms
- independent model and optimizer state
- independent projection copies from the same W28 T0 projection

## Frozen loss / inference shell

Both arms keep:
- S13 relation canonicalizer
- S14 equal standardized full-K fusion
- S15 relation-logit detach
- S17 primary/relation loss partition
- S17 relation-priority norm-balanced gradient update
- S17 coefficients and temperatures
- state-once/full-K
- no learned downstream scorer/router/gate/calibrator

## Independent selectors

Each arm selects its own epoch using the exact S17 lexicographic selector:
1. paired
2. fused canonical
3. relation canonical
4. relation canonical margin
5. fused canonical margin
6. question-swap
7. fused agreement
8. signature cosine
9. signature discrimination
10. lower canonical decision loss
11. earlier epoch

No cross-arm epoch mixing.

## Semantic gates

Each arm independently uses the existing v1 frontier gates:
- fused canonical >= .85
- paired >= .75
- question-swap >= .80
- fused agreement >= .95
- fused JS <= .05
- fused canonical margin >= .15
- relation canonical >= .80
- relation margin >= .15
- signature cosine >= .90
- signature discrimination >= .15
- option flip <= .02
- mass <= 1e-6
- full-K/state-once/capacity/freeze mechanics

## Matched comparison

Report exact metric deltas:

`FFN-only - attention-only`

for:
- fused canonical/paraphrase
- paired
- question-swap
- fused agreement/JS
- fused canonical/paraphrase margins
- raw primary canonical/paraphrase
- relation canonical/paraphrase
- relation margins/agreement
- signature cosine/discrimination

Do not collapse the comparison into one scalar score.

## Preregistered interpretation

### A — FFN-only coherent advantage
If FFN-only improves the primary semantic endpoints coherently on the matched rows, the FFN-only family remains viable for a later fresh confirmation. S30 itself is not confirmation.

### B — attention-only matches/exceeds FFN-only
Close final-block FFN adaptation as a useful v1 direction. Do not widen the encoder further merely because more layers are available.

### C — split evidence
Close S30 as unresolved. Do not tune either arm on exposed DEV to force a winner.

### D — one arm independently reaches DEV_READY
Freeze that arm immediately. No second S30 DEV. Any production/external claim still requires the separately defined next-stage confirmation protocol.

## Stop rule

After S30 DEV exposure:
- no rank/alpha/dropout changes
- no arm-specific LR/schedule
- no extra epochs
- no seed retry
- no loss/fusion/selector changes
- no gate weakening
- no third arm
- no earlier-layer adaptation

No Laya/Jev benchmark unless a later confirmed candidate authorizes it.
