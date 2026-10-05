# HIRA V1 S57 contract — Discrete Pairwise Ranking Consistency

Status: **PREREGISTERED / NO S57-A0 EXPOSURE**

Issue: #297

Parent:
- S56 run `37265241846`
- artifact `11326048054`
- Case **B**
- whole-distribution consistency improves stability but collapses correctness.

## Scientific question

> Can cross-view ordinal option ordering be stabilized without forcing whole probability distributions to match?

## Frozen backbone

S57 inherits:
- exact persisted S51 native authority;
- one immutable shared native cache;
- `JointStateQueryOptionPrivateCorrectionFork`;
- correction params **114,688 / arm**;
- identity params **0**;
- native trainability **0**;
- one encoder/state-once;
- full-K.

No new trainable parameters.

## Standardized pairwise margins

Before ranking comparison, each view's full-K logits are standardized with the exact S56 operator:

`z = (logits - mean) / sqrt(mean(centered^2) + 1e-6)`

This freezes:
- offset invariance;
- positive-scale invariance;
- comparable pairwise thresholds.

For every unordered option pair `i<j`:

`m_a = z_a[i]-z_a[j]`
`m_b = z_b[i]-z_b[j]`

## Eligibility and directional anchors

Frozen activation threshold:
**0.25**

Overall pair eligibility:
`max(abs(m_a),abs(m_b)) >= 0.25`

A directional anchor contributes only when its own absolute standardized margin is >= 0.25.

Anchor signs are detached:
`s_a = sign(stop_gradient(m_a))`
`s_b = sign(stop_gradient(m_b))`

Frozen preserved margin floor:
**0.05**

Directional penalty:
`softplus(0.05 - s_anchor * m_other)`

The symmetric ordinal loss averages all valid canonical→paraphrase and paraphrase→canonical directional penalties.

## Gold-order protection

For an option pair that contains the gold option:

An anchor direction is allowed only if that anchor view ranks the gold above the paired distractor.

If `i==gold`, anchor is correct only when `m_anchor>0`.
If `j==gold`, anchor is correct only when `m_anchor<0`.

Therefore:
- a wrong gold-vs-distractor ranking may not teach the other view;
- a correct gold-vs-distractor ranking remains eligible;
- non-gold pairs remain eligible under the ordinary threshold rule.

Gold labels are used only for this anchor filter. They do not add parameters.

## Matched arms

Reference:
- ordinal coefficient **0.0**

Treatment:
- ordinal coefficient **0.05**

Both:
- same architecture
- same correction params 114,688
- added params 0
- bit-identical initialization
- same native/cache bytes
- same rows/order
- same optimizer/LR/weight decay/grad clip
- same base correctness/relation objective
- same checkpoint selector.

No S56 decision-JS term is used.

## Required S57-A0

### Matched ownership
- correction params 114,688 each
- added params 0
- identity params 0
- initialization bit-identical
- native optimizer absent
- native/cache gradients 0
- one encoder/state-once.

### Ordinal mechanics
- matching strong ranking has low/minimal loss
- controlled sign flip has materially larger loss
- shared logit offset invariance exact
- positive logit scale invariance within floating tolerance
- anchor signs detached
- wrong gold-involving anchor is filtered
- correct gold-involving anchor is retained
- non-gold active anchor remains eligible
- flat logits => zero active anchor fraction and exact zero weighted ordinal loss
- option permutation equivariance with remapped gold
- K=3/7/255 finite
- full-K probability mass <=1e-6
- treatment auxiliary gradient reaches correction params
- reference auxiliary exact zero.

### Anti-collapse
Uniform logits have:
- active anchor fraction 0
- top1-top2 gap 0
- logit RMS 0.

Uniform predictions cannot count as successful ordinal stability.

## Fresh authority

Intended:
- seed **78001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S57 domains
- K=4
- private epochs **24**
- one DEV only.

Exact overlap with S56 state/question/option text must be zero.

## Frozen interpretation

**A** — selected-choice stability materially improves while correctness/discrimination remains:
pairwise ordinal consistency is useful.

**B** — stability improves but correctness still materially collapses:
bad/unstable anchors remain; move to consensus/teacher anchoring.

**C** — correctness remains but stability does not materially improve:
ordinal auxiliary is too weak; move to an explicit pairwise decision model.

**D** — correctness and stability both regress:
reject pairwise ordinal consistency.

**E** — full DEV_READY:
freeze immediately and open confirmation before external Laya/Jev evaluation.

## Stop rule

After one S57 DEV:
- no coefficient sweep
- no threshold/margin sweep
- no anchor-filter variant
- no teacher/consensus anchor added post-hoc
- no architecture/capacity change
- no native retraining
- no selector change
- no retry
- no gate weakening
- no second S57 DEV
- no external Laya/Jev evaluation.
