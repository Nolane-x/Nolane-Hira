# R8-W5g handoff — contrastive-salience late interaction

Status: **implementation active under issue #71; no empirical W5g result yet.**

## Why W5g exists

W5f established direct token late interaction as a material semantic mechanism:
- pooled CONFIRM accuracy 0.52%;
- W5f proj128 MaxSim 29.69%;
- K128 25%;
- K255 16.67%.

W5f remains PARTIAL because uniform token aggregation misses the frozen rescue gates.

## New hypothesis

Dynamic candidate sets contain token content that is common across nearly every option and therefore weakly discriminative.

W5g tests whether candidate-relative token salience improves binding:
- use tokenizer document frequency across current options;
- downweight common option tokens;
- upweight rare/discriminative option tokens;
- optional common-mode subtraction for repeated token IDs.

No W5f CONFIRM case is reused.

## Controlled candidates

All three have exactly the same trainable surface:
- bias-free 256->128 projection;
- one scalar logit scale.

Candidates:
1. uniform-proj128;
2. idf-proj128;
3. idf-centered-proj128.

Same optimizer, epochs, seed and fresh data. Only aggregation differs.

## Fresh authority

TRAIN 512, DEV 176, untouched post-selection CONFIRM 192.
K reaches 255.
Seeds: 121001 / 122002 / 123003.

W5g vocab/templates/case IDs are disjoint from W5a-W5f.

## Selection and confirm

Freeze one candidate on DEV using:
accuracy -> K128 -> K255 -> top5 -> MRR -> Brier -> earlier epoch.

Selector must always carry the fresh uniform-proj128 checkpoint into the selected artifact.

After freeze, generate CONFIRM once and evaluate:
- selected candidate;
- fresh uniform baseline;
- frozen pooled A13 baseline.

## Scientific scope

Zero public campaign cells.
No Banking77, typed final, MASSIVE/XNLI or Laya/Jev final examples.
No posthoc threshold/weight changes after CONFIRM exposure.
