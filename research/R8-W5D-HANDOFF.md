# R8-W5d handoff — controlled A13 top-layer semantic adaptation

Status: **implementation active under issue #65; no empirical result yet.**

## Why W5d exists

W5c verdict was `A13_PROBE_FAIL`.

Frozen A13 did not expose sufficient fresh high-cardinality semantic signal to:
- pooled cosine;
- token-max;
- learned bilinear;
- learned pair-MLP comparison.

W5d therefore changes representation rather than adding another scorer.

## Frozen mechanism

A13 exact base:
- microsoft/xtremedistil-l6-h256-uncased;
- revision 4226d9e4d2c08703e5cb0491b479bfc6a1607181;
- model.safetensors SHA-256 5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880.

Scorer:
- no trainable scorer;
- context = normalize(state pooled + question pooled);
- option = normalized pooled option embedding;
- logits = 10 * cosine;
- CE + 0.1 hard Brier.

Only the final one or two transformer blocks are trainable.

## Fresh authority

TRAIN:
- 416 cases;
- seed 81001;
- K=8/16/32/64/128/255.

DEV:
- 144 cases;
- seed 82002;
- 24 cases per K;
- only selection authority.

CONFIRM:
- 192 cases;
- seed 83003;
- K=32/64/128/255, 48 each;
- generated only after selection freeze.

W5d vocabulary is unit-tested exactly disjoint from W5a/W5b/W5c.

## Candidates

Exactly four:
- top1-lr1e5;
- top1-lr3e5;
- top2-lr1e5;
- top2-lr3e5.

Shared:
- AdamW;
- wd 0.01;
- 3 epochs;
- seed 171;
- no scheduler/warmup/clipping/AMP/accumulation;
- no HIRA parameter training.

## Outcome

If untouched CONFIRM satisfies all competence + representation-gain gates:
`A13_ADAPTATION_PASS`.

Otherwise:
`A13_ADAPTATION_FAIL_CAPACITY_TRIGGER`,
and A22 capacity-control becomes the next authorized lane.

W5d populates zero campaign cells.
