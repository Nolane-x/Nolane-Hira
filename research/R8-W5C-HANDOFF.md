# R8-W5c handoff — semantic alignment + representation probes

Status: **COMPLETE / NEGATIVE; merged to main as `a691ef7b0fad2ce55cb13d09e0047faa9bd5b0e8`; authoritative verdict `A13_PROBE_FAIL`.**

## Why W5c exists

W5a:
- fresh CONFIRM 0/256 top-1.

W5b:
- fresh token-aware CONFIRM 0/192 top-1;
- pooled remained best on DEV.

In both:
- full-K mechanics PASS;
- probability mass PASS;
- state-once PASS.

W5c therefore stops modifying HIRA itself and diagnoses whether frozen A13 carries recoverable semantic-key information.

## Frozen encoder

A13:
- microsoft/xtremedistil-l6-h256-uncased;
- revision 4226d9e4d2c08703e5cb0491b479bfc6a1607181;
- model.safetensors SHA-256 5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880;
- max_length 256;
- fully frozen.

## Fresh authorities

TRAIN:
- 768 cases;
- seed 71001;
- K 8/16/32/64/128/255.

DEV:
- 192 cases;
- seed 72002;
- 32 cases at each K;
- only checkpoint-selection authority.

CONFIRM:
- 256 cases;
- seed 73003;
- K 32/64/128/255, 64 each;
- generated only after both learned probe checkpoints are frozen.

Exact vocabulary overlap with W5a/W5b is forbidden and unit-tested.

No public benchmark data is allowed.

## Four probes

P0 pooled_cosine:
- no trainable parameters;
- context normalize(state_global + question);
- cosine against pooled option embeddings.

P1 token_max:
- no trainable parameters;
- option-token to state-token cosine;
- max over state tokens per option token, then mean.

P2 bilinear:
- context [state_global; question] 512 -> 128;
- option 256 -> 128;
- normalized dot product with learned scale.

P3 pair_mlp:
- context 512 -> 128;
- option 256 -> 128;
- [c,o,c*o,abs(c-o)] -> 128 -> scalar.

P2/P3:
- hard CE + 0.1 hard Brier;
- AdamW lr 3e-4, wd 0.01;
- 8 epochs;
- Python/NumPy/Torch seed 131;
- deterministic shuffle seed 131 + epoch.

## DEV checkpoint key

For each learned probe:
1. higher overall accuracy;
2. higher K128 accuracy;
3. higher K255 accuracy;
4. higher top5 recall;
5. higher MRR;
6. lower hard Brier;
7. earlier epoch.

P0/P1 are fixed probes and have no checkpoint selection.

## CONFIRM classification

Each probe passes competence only if:
- overall accuracy >= .60;
- K128 accuracy >= .40;
- K255 accuracy >= .30;
- K255 top5 >= .70;
- probability mass max error <= 1e-6.

Frozen precedence:
1. DIRECT_POOLED_SIGNAL
2. DIRECT_TOKEN_SIGNAL
3. LINEAR_RECOVERABLE_SIGNAL
4. NONLINEAR_RECOVERABLE_SIGNAL
5. A13_PROBE_FAIL

This is diagnostic classification, not model selection.

If all four fail, stop adding scorer variants around frozen A13 and move to encoder representation/capacity research.


## Authoritative result

Run: `35834712173`.

Artifacts:
- frozen inputs `10739205899`;
- bilinear `10738787670`;
- pair-MLP `10739275838`;
- confirm `10738718234`.

Fresh DEV:
- pooled cosine accuracy 1.0417%;
- token-max 3.125%;
- bilinear 4.1667%, K128/K255 0%;
- pair-MLP 2.0833%, K128/K255 0%.

Untouched CONFIRM:
- pooled 0/256;
- token-max 1/256;
- bilinear 0/256;
- pair-MLP 0/256;
- every K128/K255 top-1 gate fails;
- probability mass PASS;
- state-once 256/256 PASS.

Verdict:
`A13_PROBE_FAIL`.

This result closes scorer-only exploration around frozen A13. It does not prove all possible decoders impossible, but it is strong negative evidence across raw pooled, raw token, learned linear/bilinear and learned nonlinear comparison.

## Next scientific step

Run controlled A13 top-layer semantic adaptation on entirely fresh authorities.

Do not run A22 yet unless the existing capacity trigger remains satisfied after this top-layer-unfreezing experiment.
