# R8-W7b handoff — free-form retune attribution and replication

Status: **PRE-DATA. No W7b empirical cache, training result, DEV selection or CONFIRM result exists yet.**

Issue: #99

Branch:
`feat/r8-w7b-freeform-attribution`

Base main at branch creation:
`e7e015ccdc441e02f9d9e3edef36afa4d138182e`

Read first when restoring:
1. this file;
2. `research/R8-W7-HANDOFF.md`;
3. `research/R8-W6J-HANDOFF.md`;
4. issue #99.

## 0. Project identity

Nolane HIRA is a compact non-autoregressive typed decision system, not a tiny next-token LLM.

Long-term goals:
- state encoded once/case;
- dynamic semantic schemas;
- large changing candidate sets;
- typed primitives: choice / score / noul;
- small local-friendly parameter budget;
- semantic binding and structured decision making;
- calibrated/reliable probabilities;
- fresh sealed authorities and strict negative-result preservation.

## 1. Frozen A13

- `microsoft/xtremedistil-l6-h256-uncased`;
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`;
- weight SHA `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`;
- max length 256;
- fully frozen.

## 2. Production path entering W7b

- competitive free-form semantic scorer: 32,769 trainable params;
- HIRACore: frozen in W7b;
- state-once runtime;
- relation stage remains frozen;
- all W7b trainable candidates initialize from exact W6e joint-primary.

Frozen W6e init:
- HIRA SHA `d1d3359b01f0ef863de226bf51144c295eebafdeae6245fdd6d68d1db22b2588`;
- scorer SHA `6d5a7f2d3ed63ecd756181b1cb54e4704f68e5f74983a897f0a68fb4d1d63d2e`;
- artifact `10805567861`.

## 3. Decisive history

W5 established that simple pooled matching and raw encoder scaling were insufficient; late interaction + competitive salience produced the production scorer.

W6/W6b integrated it into typed HIRA and achieved strong overall competence but incomplete K64/reliability.

W6c showed calibration is separable from semantic generalization.

W6d/e showed high overall typed accuracy can coexist with unstable K64 fresh-domain generalization.

W6f/g/h/i localized and then falsified several local explanations:
- removing salience is worse;
- relation reranking is net helpful;
- isolated field retrieval can be weak;
- small residual adapter does not reproduce;
- naive canonical/factorized interfaces do not rescue.

W6j established a stable architectural diagnosis:
**`COARSE_CONJUNCTION_LIMIT`**.

W7 then tested an explicit factor-evidence smooth-AND branch.

Exact W7 empirical head:
`e4563996b1a905c777460f723bfbec1a41ac92ea`

Authority run:
`36098102118`

Frozen verdict:
**`CONJUNCTIVE_COARSE_FAIL`**.

W7 merged to main:
`e7e015ccdc441e02f9d9e3edef36afa4d138182e`.

## 4. W7 failure and positive control signal

CONFIRM-AL:
- frozen K64 18.75%;
- freeform-retune 81.25%;
- conjunctive-primary 43.75%;
- conjunctive-replica 52.08%.

CONFIRM-AM:
- frozen K64 10.42%;
- freeform-retune 62.50%;
- conjunctive-primary 64.58%;
- conjunctive-replica 62.50%.

The explicit conjunction architecture fails its causal and replication gates.

But freeform retuning dramatically improves the frozen scorer:
- AL one-field K2 coarse 72.72% -> 95.33%;
- AL pair losses 12.40 -> 1.04;
- AM one-field K2 coarse 71.96% -> 90.82%;
- AM pair losses 12.38 -> 2.04.

This signal is not yet attributed and not production-approved.

## 5. W7b purpose

W7b asks:

**Which part of the W7 free-form retune recipe caused the gain, and does it reproduce on wholly fresh domains?**

W7b adds no new architecture.

## 6. Forbidden evidence

Never reuse for W7b training/selection/tuning:
- W6b CONFIRM;
- W6c CONFIRM;
- W6d F;
- W6e L/M;
- W6f N/O/P;
- W6g Q/R/S;
- W6h Y/Z;
- W6i AA/AB/AC;
- W6j AD/AE/AF;
- W7 AL/AM;
- W7 AG-AK as W7b rows;
- typed final/test;
- public campaign cells.

AL/AM are exposed permanently.

## 7. Fresh W7b domains

TRAIN:
- AN orbital coolant routing;
- AO sterile packaging logistics;
- AP autonomous bridge inspection;
- AQ subsea communications repair.

DEV:
- AR desert atmospheric sensing.

CONFIRM:
- AS distributed battery fire safety;
- AT robotic cold-chain handling.

Seeds:
- AN 261401;
- AO 261407;
- AP 261419;
- AQ 261427;
- AR 262531;
- AS 263641;
- AT 264749.

Optimization:
- typed-only 2003;
- pair-only 2011;
- typed-plus-pair primary 2017;
- typed-plus-pair replica 2027.

## 8. Candidate factorial

0. frozen-w6e-control
- 0 trainable.

1. typed-only-retune
- scorer 32,769 trainable;
- typed W6 objective only.

2. pair-only-retune
- scorer 32,769 trainable;
- diagnosis full-K one-field coarse hinge only;
- diagnostic attribution path, not production candidate.

3. typed-plus-pair-primary
- scorer 32,769 trainable;
- exact W7 freeform recipe;
- typed objective + pair margin.

4. typed-plus-pair-replica
- same as primary;
- independent seed.

No candidate may be added after exposure.

## 9. Frozen objectives

Typed:
- hard CE 1.0;
- teacher KL .5;
- hard Brier .1;
- soft Brier .5;
- ordinal MAE .2.

Pair margin:
- full-K coarse;
- gold vs known one-field negatives;
- margin .20;
- weight .25.

Pair-only optimizes only this weighted pair loss.

Training:
- exactly 6 epochs;
- AdamW lr 3e-4;
- weight decay .01;
- no scheduler/warmup/AMP/clipping;
- deterministic shuffle;
- full-K;
- same cases/steps.

## 10. Authority shape

Same five typed decisions:
diagnosis / response / needs_review / risk / urgency.

Diagnosis K:
8 / 16 / 32 / 64.

48 strata.

TRAIN:
384 states / 1,920 decisions.

DEV-AR:
192 / 960.

CONFIRM-AS:
192 / 960 sealed.

CONFIRM-AT:
192 / 960 independently sealed.

AS/AT only after all DEV checkpoints are frozen.

K64 anatomy:
- gold;
- >=12 one-field;
- >=20 two-field;
- >=15 three-field;
- remaining far/cross-combination negatives.

## 11. DEV freeze

Independent per candidate, lexicographic:
1. K64 final top1;
2. K32 final top1;
3. one-field K2 coarse;
4. lower K64 K2-coarse pair-loss count;
5. diagnosis choice;
6. overall;
7. earlier epoch.

No cross-candidate winner selection.

## 12. Frozen replication gates

typed-plus-pair primary on both AS/AT:
- overall >=.85;
- diagnosis choice >=.80;
- K32 >=.75;
- K64 >=.60;
- one-field K2 coarse >=.90;
- K64 pair-loss count <=3.0;
- probability error <=1e-6;
- state once.

Versus frozen:
- K64 gain >=+.30;
- one-field gain >=+.12;
- pair-loss reduction >=50%.

Replica:
- K64 >=.55;
- one-field >=.87;
- overall >=.82.

## 13. Frozen attribution classes

`TYPED_RETUNE_DOMINANT`:
- typed-only within .05 K64 of typed+pair on both;
- within .03 one-field K2 on both;
- pair margin gives <15% pair-loss reduction vs typed-only.

`PAIR_MARGIN_CAUSAL_CONTRIBUTOR`:
- typed+pair beats typed-only by >=.05 K64 OR >=.04 one-field on both;
- >=20% pair-loss reduction on both;
- pair-only independently improves frozen by >=.10 K64 OR >=.08 one-field on >=1 domain;
- no >.03 overall regression vs typed-only.

`PAIR_MARGIN_PRIMARY`:
- pair-only within .05 K64 and .03 one-field of typed+pair on both;
- typed-only trails typed+pair by >=.08 K64 on both.

`MIXED_FREEFORM_ATTRIBUTION`:
replication passes but attribution classes disagree.

`FREEFORM_RETUNE_NONREPLICATING`:
primary or replica replication gates fail.

Overall:
- `FREEFORM_RETUNE_REPLICATION_ATTRIBUTED`;
- `FREEFORM_RETUNE_REPLICATION_UNATTRIBUTED`;
- `FREEFORM_RETUNE_NONREPLICATING`.

No new verdict after exposure.

## 14. Scientific boundary

No new architecture in W7b.

Do not:
- reuse AL/AM;
- change gates/seeds/losses after exposure;
- add conjunction branch;
- scale A13;
- unfreeze HIRACore;
- choose seeds after CONFIRM;
- promote pair-only;
- claim broad superiority.

If typed+pair reproduces strongly:
next phase may test a production free-form retune recipe on external/public tasks.

If it fails:
preserve failure and stop treating W7 control as a general solution.

## 15. Current state

Completed:
- W7 authoritative failure frozen into `research/R8-W7-HANDOFF.md`;
- PR #98 merged;
- issue #97 closed;
- W7b issue #99 preregistered;
- W7b branch created from exact post-W7 main;
- this handoff created before any W7b empirical exposure.

Not yet implemented:
- fresh AN-AT authority generator;
- state-once cache;
- factorial trainer;
- DEV-AR freezer;
- sealed AS/AT evaluator;
- W7b unit workflow;
- W7b authority workflow.

No W7b empirical data exists.

## 16. Immediate continuation

1. Implement fresh AN-AT generator and freshness tests.
2. Implement factorial candidates without new architecture.
3. Reuse production scorer/HIRACore contracts.
4. Build state-once cache with diagnosis hard-negative identity metadata.
5. Implement all loss modes and optimizer-step parity.
6. Implement independent DEV freeze.
7. Seal AS/AT capability.
8. Freeze classifier/verdict code before data.
9. Run pre-data unit/CI.
10. Enable authority only after all contracts are green.
11. After first eligible exposure, mutate nothing except technical infrastructure before result exposure.
12. Freeze exact result into this handoff before merge.

## 17. Handoff discipline

At every meaningful session update:
- exact branch head;
- exact runs;
- artifacts/digests;
- pre-data/exposed status;
- exact metrics if exposed;
- frozen verdict;
- completed/remaining work;
- forbidden next moves.

A future AI should be able to continue without chat memory.
