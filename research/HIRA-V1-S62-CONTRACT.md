# HIRA V1 S62 contract — TRAIN-Only Reliability-Supervised Adaptive Gate

Status: **PREREGISTERED / NO S62-A0 OR DEV EXPOSURE**

Issue: #309

## Parent evidence

S61 closes as **Case B**.

Fresh S61:
- run `37302829277`
- artifact `11343045582`
- digest `sha256:6e99ba1a5fab92cb6031c41eea54636721b78e690236dfb39788066803785f74`
- merged main `e6c7053a8d24c75b5fcf1db75c8aa9f30cea35e2`.

Treatment minus fused:
- canonical **+0.26 pp**
- paraphrase **+2.60 pp**
- paired both-correct **-0.52 pp**
- question-swap **+1.56 pp**
- agreement **0.00 pp**
- JS **+0.003272 worse**.

Conclusion: adaptive alpha is mechanically real and correctness-safe, but gold CE does not supervise cross-view reliability.

## Scientific question

> Does explicit TRAIN-only counterfactual reliability supervision make the exact S61 gate learn when bounded pairwise evidence is actually safe and stability-improving?

## Frozen matched architecture

Shared:
- S51 persisted native authority;
- immutable cache;
- one encoder/state-once;
- correction params **114,688**;
- pairwise head params **32,832**;
- exact S61 feature function, dimension **4**;
- exact S61 gate architecture, **5 params / arm**;
- no hidden layer;
- alpha max **0.35**;
- initial alpha **0.10**;
- bounded residual direction unchanged;
- pairwise-only final path forbidden;
- no teacher / DEV pseudo-target / self-anchor.

Reference gate:
- exact S61 gold-CE calibration objective.

Treatment gate:
- TRAIN-only reliability BCE.

Both gates initialize bit-identically:
- w = 0
- b = -0.916290731874155.

Added treatment params over reference: **0**.

## Counterfactual reliability target

Frozen:
- probe alpha **0.35**
- tolerance **1e-8**.

For each aligned canonical/paraphrase TRAIN query pair:
- baseline = exact fused logits;
- probe = exact bounded hybrid using fixed alpha_override=0.35.

Per query pair:
`CE_base = 0.5*(CE(f_c,g)+CE(f_p,g))`
`CE_probe = 0.5*(CE(q_c,g)+CE(q_p,g))`
`JS_base = JS(f_c,f_p)`
`JS_probe = JS(q_c,q_p)`.

Binary target:
`y=1` iff:
- `CE_probe <= CE_base + 1e-8`
- AND `JS_probe < JS_base - 1e-8`.

Otherwise `y=0`.

This is a Pareto target with an explicit correctness veto. There is no CE/JS scalar tradeoff coefficient.

## Treatment loss

For canonical and paraphrase S61 features:
- `z_c=b+w*x_c`
- `z_p=b+w*x_p`

Use the same paired target y on both:
`L_rel = 0.5*(BCEWithLogits(z_c,y)+BCEWithLogits(z_p,y))`.

Targets/features are detached. No treatment gate gradient enters correction, pairwise head, native runtime or cache.

## Reference loss

Exact S61 `adaptive_gate_gold_loss`.

## Required A0

Reliability target:
- beneficial synthetic court => y=1;
- JS-help but CE-harm => y=0;
- CE-help but JS-harm => y=0;
- both-harm => y=0;
- target detached;
- alpha_probe exact 0.35;
- tolerance exact 1e-8;
- no DEV API.

Matched court:
- reference/treatment gate params 5 each;
- trainable tensors w,b each;
- bit-identical initialization;
- treatment minus reference added params 0;
- reference CE gradients reach w,b;
- treatment BCE gradients reach w,b;
- both have zero upstream gradients;
- mixed target court contains both classes.

Mechanics:
- feature dimension 4;
- K=3/7/255;
- option permutation;
- exact alpha=0 fused identity;
- residual bound;
- probability mass <=1e-6;
- one encoder/state-once;
- checkpoint roundtrip exact.

## Fresh S62 authority

- seed **83001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh domains
- exact S61 state/question/option overlap **0**
- K=4
- 24 epochs
- one DEV only.

## Frozen interpretation

A — reliability treatment improves stability materially over gold-CE reference while correctness/discrimination is retained/improved.

B — gate behavior changes but stability remains weak: S61 four-feature representation is insufficient.

C — stability improves but correctness materially falls despite TRAIN correctness veto: probe-target transfer is unsafe; next family needs an inference-time correctness-preserving veto.

D — both correctness and stability regress: reject reliability-supervised gate family.

E — full DEV_READY: freeze and open one separate fresh confirmation court before external Laya/Jev evaluation.

## Stop rule

After one S62 DEV:
- no probe-alpha/tolerance/target change;
- no BCE weighting;
- no feature/hidden-layer change;
- no alpha max/init sweep;
- no optimizer change;
- no regularizer;
- no gradient coupling;
- no architecture/capacity change;
- no native retraining;
- no selector change;
- no retry;
- no second S62 DEV;
- no external Laya/Jev evaluation.
