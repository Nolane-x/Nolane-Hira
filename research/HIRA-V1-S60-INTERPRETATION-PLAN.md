# HIRA V1 S60 interpretation plan — frozen before S60-A0/DEV exposure

Status: **FROZEN**

Issue: #305

## Controlled variable

Shared TRAIN trajectory:
- correction 114,688 params
- pairwise head 32,832 params
- composer scalar 1 param
- same immutable cache
- same gold labels
- same optimizers frozen by contract
- no teacher
- same S17 selector.

Reference:
- exact fused decision.

Treatment:
- bounded TRAIN-calibrated fused + pairwise residual.

## Evidence hierarchy

1. exact S51 native authority;
2. immutable cache ownership;
3. fresh S60 partition authority;
4. S60-A0 exact identity + bounded influence;
5. pairwise and composer gradient isolation;
6. parameter accounting 114,688 + 32,832 + 1;
7. shared epoch trajectory receipts;
8. correctness/discrimination;
9. selected-choice stability.

If 1–7 fail, A–E classification is forbidden.

## Primary DEV metrics

Reference/treatment:
- selected epoch
- canonical/paraphrase accuracy
- paired both-correct
- question-swap
- selected-choice agreement
- cross-view JS
- canonical/paraphrase gold margins
- decision CE
- full-K mass
- state-view encodes.

Treatment composer:
- selected alpha
- alpha trajectory
- residual max / fused RMS
- bound violations = 0
- pairwise gold-pair accuracy/margin
- anti-symmetry error.

## Material interpretation

**A** — stability improves while correctness/discrimination is retained or improved.

**B** — correctness/discrimination retained but stability gain is not material.

**C** — stability improves materially but correctness/discrimination materially falls.

**D** — both regress materially.

**E** — full DEV_READY.

No scalar winner score.

## Stop rule

One S60 DEV only. No post-DEV:
- blend/calibration sweep
- objective change
- adaptive gate
- residual transformation change
- gradient-path change
- selector change
- retry.
