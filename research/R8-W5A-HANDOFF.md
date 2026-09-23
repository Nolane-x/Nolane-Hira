# R8-W5a handoff — fresh semantic routing competence rebuild

Status: **COMPLETE empirical negative result; merged to `main` in `ebbba0d29574386624f53b9b0f1538aa75189295`.**

## Frozen rationale

W4a:
- Banking77 semantic generalization: 0/400.

W4b:
- MECHANICS_PASS at K=128 and K=255;
- semantic-key routing near random.

Therefore W5a trains semantic relation/routing using only fresh generated data.

## Forbidden

No Banking77, typed final, W4b cases/vocab/templates, MASSIVE/XNLI eval or Laya/Jev final examples.

## Authorities

TRAIN:
- 640 cases;
- K 8/16/32/64/128/255;
- seed 51001;
- four training templates.

DEV:
- 160 cases;
- seed 52002;
- held-out templates;
- only authority for candidate/epoch selection.

CONFIRM:
- 256 cases;
- K=32/64/128/255, 64 each;
- seed 53003;
- disjoint vocabulary and templates;
- generated only after selected head/config is frozen.

## Model

Frozen A13 encoder.

Train only 422,159-parameter HIRA head.

Candidate grid:
- init R15 / W3;
- lr 3e-4 / 1e-3;
- four candidates;
- six epochs;
- AdamW wd 0.01;
- hard CE 1.0 + hard Brier 0.1;
- full-K forced budget 255;
- no adaptive budget.

## Confirm gate

SEMANTIC_COMPETENCE_PASS requires all:
- overall accuracy >= 0.50;
- K128 accuracy >= 0.30;
- K255 accuracy >= 0.20;
- K255 top5 >= 0.50;
- probability mass max error <= 1e-6;
- candidate budget exactly K;
- state-once 1 encode/case.

W5a populates zero R8 campaign cells.


## Authoritative result

Run `35826397280` completed the full frozen pipeline.

DEV-selected head:
- candidate `w3-lr3e4`;
- epoch 1;
- SHA-256 `c38da77a6b71b249ab02925b2f0581f06f2ef0fb9035bac6c0d4135203b725ec`;
- DEV accuracy 0.03125;
- DEV top-5 recall 0.18125.

Untouched CONFIRM:
- 256 cases;
- accuracy **0/256 = 0.0**;
- top-5 recall 0.01171875;
- MRR 0.0370922766;
- K=32/64/128/255 accuracy all 0.0;
- K=255 top-5 recall 0.0;
- probability mass PASS;
- exact full-K budget PASS;
- state-once PASS.

Scientific verdict:
**SEMANTIC_COMPETENCE_FAIL**.

Artifacts:
- selected head `10735990905`, digest `sha256:b96ad96eeb5f5dde0e670b4e1623e65741ec0c815ca423bd9fd9b7200b8e0cf1`;
- confirm `10736475259`, digest `sha256:cbe7fda9a1482e5a54d70416057dd721d031608659ed9152e200a5c51cf1460f`.

Do not change W5a gates, LR, epochs, vocab, templates or CONFIRM after exposure.

## Failure anatomy

W5a confirms that mechanics are healthy but the pooled semantic comparison mechanism is not.

A concrete architectural gap is now explicit:
- A13 emits token embeddings;
- HIRACore already supports `option_tokens` and token-level summarization;
- W5a cache/training supplied only pooled option embeddings;
- the existing token-aware path therefore received no option tokens;
- short state descriptions were represented to the relation path through coarse segment pooling.

The next experiment must test a fresh token-aware comparison mechanism on new authorities, not rescue W5a.
