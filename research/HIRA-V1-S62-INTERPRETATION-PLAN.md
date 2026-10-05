# HIRA V1 S62 interpretation plan — frozen before A0/DEV exposure

Status: **FROZEN**

Issue: #309

## Controlled variable

Shared:
- exact native/cache/correction/pairwise trajectory;
- exact S61 4 features;
- exact 5-param gate architecture;
- exact alpha bounds/init;
- exact optimizer constants;
- exact bounded decision composition;
- exact frozen S17 selector.

Reference:
- S61 gold-CE gate supervision.

Treatment:
- counterfactual reliability BCE supervision.

No other scientific variable may differ.

## Reliability target contract

Fixed alpha probe: **0.35**.  
Comparison tolerance: **1e-8**.

Positive iff probe is Pareto-safe:
- paired gold CE non-worse within tolerance;
- paired cross-view JS strictly lower beyond tolerance.

Any correctness harm => negative.
Any stability non-improvement => negative.

## Primary evidence

For each branch report:
- selected epoch
- canonical/paraphrase accuracy
- paired both-correct
- question-swap
- agreement
- JS
- gold margins
- relation diagnostics
- gate mean/min/max/std alpha
- TRAIN reliability positive fraction
- TRAIN reliability BCE
- full-K/mass/state-once.

Matched proof:
- correction/head state hashes identical across arms/epoch;
- reference/treatment gates identical at init;
- no treatment added capacity;
- targets TRAIN-only.

## Frozen cases

**A**: materially stronger stability with correctness/discrimination retained or improved.

**B**: target affects gate but stability remains weak.

**C**: stability improves materially but correctness falls.

**D**: both regress.

**E**: full DEV_READY; confirmation only before external comparison.

No scalar winner score and no post-DEV reinterpretation.
