# HIRA V1 S29 handoff — to S30 Matched Attention-vs-FFN Adaptation Surface Court

S29 is frozen as:

`HIRA_V1_S29_FULL_BLOCK_LORA_DEV_FAIL`

## Canonical authority

A0:
- run `36883766054`
- artifact `11172806806`
- digest `sha256:1f05cab39d88b481968d27c7260b9a7192f184bf650863c5fc7874c2d2f53f9f`
- outcome `HIRA_V1_S29_A0_FULL_BLOCK_LORA_READY`

TRAIN/DEV:
- run `36884997207`
- artifact `11175525933`
- digest `sha256:a0047da70d0708c51aa9a23f5fe1338cfb1f434f88e5d80e2e322bfd9eaa51ec`
- scientific head `6a479d20c727031011c734d7f4ba34498f526cf9`
- selected epoch **12**
- checkpoint `7abfe0209d1c0a8166a365c58c0e1193dafe5ceb33a0ad8a4e961d34b3eba426`
- outcome `HIRA_V1_S29_FULL_BLOCK_LORA_DEV_FAIL`

## Selected DEV

- fused canonical **0.5807291667**
- fused paraphrase **0.3385416667**
- paired **0.2916666667**
- question-swap **0.5104166667**
- fused agreement **0.4661458333**
- fused JS **0.0421490973**
- fused margins **+0.0490893635 / -0.2587409501**
- relation canonical/paraphrase **0.4348958333 / 0.3385416667**
- relation margins **-0.0679718368 / -0.6121830083**
- relation agreement **0.5963541667**
- signature cosine **0.8419482509**
- signature discrimination **0.1245272141**

Gate result:
- **13/22 PASS**
- **9/22 FAIL**

## Key residual

The new FFN capacity is active:
- FFN intermediate A0 B-gradient L1 **2.3468730450**
- FFN output A0 B-gradient L1 **0.9416253567**

TRAIN improves continuously, but DEV cross-view structure does not:
- TRAIN total loss **1.7388 → 0.8126**
- signature cosine peaks **0.95194** at epoch 5
- signature cosine ends **0.54644** at epoch 24
- relation canonical/paraphrase margins remain negative at all epochs

Therefore do not widen S29.

## S30 target

**Matched Attention-vs-FFN Adaptation Surface Court**

Question:

> Under the exact same fresh data and S17 optimization shell, is final-block FFN-only adaptation stronger or weaker than the already-established attention-only adaptation surface?

Arm A:
- attention-only LoRA **16,384**
- shared projection **32,768**
- total **49,152**

Arm B:
- FFN-only LoRA **20,480**
- shared projection **32,768**
- total **53,248**

Both:
- rank 8 / alpha 8 / dropout 0
- original A13 frozen
- HIRACore frozen
- same S13 relation expert
- same S14 fusion
- same S15 detach
- same S17 loss partition
- same S17 norm-balanced gradient rule
- identical fresh S30 TRAIN/DEV rows
- identical optimizer and batch order
- independent zero-init LoRA states
- independent projection copies initialized from the same W28 T0 projection

S30 is a paired mechanism-localization court.

Do not:
- reuse S29 DEV rows
- tune either arm after DEV
- add a full-block third arm
- adapt earlier layers
- vary rank / alpha / LR by arm
- weaken gates
- declare Laya/Jev parity from this court

If FFN-only materially dominates the matched attention-only arm, the next candidate may build from FFN-only with a fresh confirmatory protocol.

If attention-only matches or exceeds FFN-only, close final-block FFN adaptation as a useful v1 direction and move away from encoder-surface widening.
