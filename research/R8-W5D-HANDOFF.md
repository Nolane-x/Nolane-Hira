# R8-W5d handoff — controlled A13 top-layer semantic adaptation

Status: **COMPLETE / NEGATIVE; merged to `main`; authoritative verdict `A13_ADAPTATION_FAIL_CAPACITY_TRIGGER`; A22 capacity control authorized.**

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


## Authoritative W5d result

Merge commit:
`c44a16452b9a013fdccdd9486d5fdbca9dab0433`.

Authoritative run:
`35844159119`.

Artifacts:
- selected adaptation `10742754150`, digest `sha256:b52239222910bc12f81d3ac99a6019ca56959d9896c70a8371960c2dfbe771f5`;
- untouched confirm `10743381013`, digest `sha256:9327c703f7d548eaa315598b27884f98dff7ce40ad2ff7c6187200f433731ecf`.

DEV selected:
- candidate `top2-lr1e5`;
- top 2 / 6 A13 layers trainable;
- lr 1e-5;
- epoch 2;
- adaptation SHA-256 `a4f2b72a84896178959ee5000028f71c9220ade95f472c8e002eebf4baac6187`;
- trainable params 1,579,520;
- DEV accuracy 0.0416667;
- K=128 accuracy 0;
- K=255 accuracy 0.

Untouched CONFIRM:
- n = 192;
- frozen A13 top-1 = 0/192;
- adapted A13 top-1 = 0/192;
- frozen MRR = 0.0402164;
- adapted MRR = 0.0447221;
- frozen top-5 = 0.015625;
- adapted top-5 = 0.0208333;
- K=128 adapted top-1 = 0;
- K=255 adapted top-1 = 0;
- probability-mass gate PASS;
- every competence/gain gate FAIL.

Verdict:
`A13_ADAPTATION_FAIL_CAPACITY_TRIGGER`.

The confirm receipt explicitly sets:
`a22_capacity_control_authorized=true`.

This closes A13 scorer/representation/top-layer adaptation sweeps. Do not reopen another A13 tuning lane before the A22 capacity diagnostic.
