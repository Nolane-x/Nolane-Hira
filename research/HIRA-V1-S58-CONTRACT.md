# HIRA V1 S58 contract — Consensus-Teacher Pairwise Ranking

Status: **PREREGISTERED / NO S58-A0 OR DEV EXPOSURE**

Issue: #299

## Parent evidence

S57 closes as **Case B**.

Fresh S57 court:
- run `37271509208`
- artifact `11327849211`
- artifact digest `sha256:868faf45b1994286e52cbac4c35adfbbb420e2ba8a6c561545067da013c05147`
- reference selected epoch **19**
- reference checkpoint `reference-private-candidate.pt`
- reference checkpoint SHA-256 `804dc0b31ca75a77400c4a658ae893058d39916ed287d9fe2d45643798b739aa`

S57 treatment improved fused agreement **+8.07 pp** but reduced correctness and relation agreement.
Conclusion: self-generated ordinal anchors are unsafe.

## Scientific question

> Can decision-level pairwise consistency retain S57's stability benefit without correctness collapse when targets come only from a frozen cross-view teacher consensus rather than the student's own current ordering?

## Frozen teacher authority

Teacher is fixed before S58 data exposure:

- source run: `37271509208`
- source artifact: `11327849211`
- artifact name: `hira-v1-s57-discrete-pairwise-ranking-consistency`
- artifact digest: `sha256:868faf45b1994286e52cbac4c35adfbbb420e2ba8a6c561545067da013c05147`
- checkpoint: `reference-private-candidate.pt`
- checkpoint SHA-256:
  `804dc0b31ca75a77400c4a658ae893058d39916ed287d9fe2d45643798b739aa`
- branch: reference
- selected epoch: **19**
- private params: **114,688**

Teacher is:
- non-trainable;
- detached from S58 optimizer;
- never selected or modified using S58 TRAIN/DEV;
- used only to define pairwise target eligibility/sign.

## Matched student court

Reference student:
- normal S58 private correctness objective;
- teacher-consensus coefficient **0.0**.

Treatment student:
- same architecture/capacity/init/data/optimizer/selector;
- teacher-consensus coefficient **0.05**.

Both:
- `JointStateQueryOptionPrivateCorrectionFork`
- correction params **114,688**
- added trainable params **0**
- identity params **0**
- exact same S51 persisted native artifact
- same immutable cache
- one encoder/state-once
- full-K.

## Teacher consensus target

For every option pair i,j:

Teacher canonical margin:
`t_c = z_c[i]-z_c[j]`

Teacher paraphrase margin:
`t_p = z_p[i]-z_p[j]`

A pair is eligible only when:
1. `abs(t_c) >= 0.25`;
2. `abs(t_p) >= 0.25`;
3. `sign(t_c) == sign(t_p)`;
4. if gold is involved, the consensus sign ranks gold above the other option.

Consensus sign is detached.

Disputed, weak, or wrong-gold teacher pairs are inactive.

## Treatment student loss

For each eligible pair and both student views:

`L_pair = softplus(0.05 - s_teacher * m_student)`

where:
- `s_teacher` is detached consensus sign;
- `m_student` is student pairwise margin.

Aggregate over eligible pairs/views only.

Frozen:
- coefficient **0.05**
- teacher active threshold **0.25**
- student target margin **0.05**
- no full-distribution JS auxiliary
- no student-self anchor
- no teacher update.

Reference weighted auxiliary is exactly zero.

## Anti-collapse

Uniform/flat teacher:
- zero eligible pairs.

Uniform/flat student with eligible teacher targets:
- positive loss because target margin is not satisfied.

Thus flattening cannot be rewarded as stability.

## Required S58-A0

Teacher authority:
- exact artifact/checkpoint hash pinning
- checkpoint schema/reference branch/epoch verified
- teacher correction loaded and frozen
- teacher gradients absent
- deterministic replay.

Consensus mechanics:
- strong same-sign teacher pair active
- low-confidence teacher pair inactive
- teacher sign-disagreement inactive
- wrong gold-order teacher pair inactive
- correct gold-order teacher pair active
- non-gold consensus pair remains eligible
- consensus sign detached.

Student mechanics:
- satisfied target margin gives lower loss than violated target
- option permutation equivariance
- shared-offset invariance
- positive scaling preserves teacher eligibility/sign
- reference auxiliary exact zero
- treatment auxiliary gradient reaches student correction
- no teacher/native/cache gradients
- flat teacher anti-collapse
- flat student with active teacher target nonzero
- K=3/7/255
- probability mass <=1e-6
- one encoder/state-once.

## Fresh S58 authority

Intended:
- seed **79001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S58 domains
- exact S57 state/question/option overlap **0**
- K=4
- private epochs **24**
- one DEV only.

## Frozen interpretation

**A** — selected-choice stability improves while correctness/discrimination is retained:
teacher-consensus ordinal targets solve the unsafe self-anchor failure.

**B** — stability improves but correctness still materially collapses:
teacher authority itself is not safe enough; move to explicit learned pairwise decision model.

**C** — correctness remains but stability does not materially improve:
consensus mask is too conservative; move to explicit learned pairwise decision model.

**D** — correctness and stability both materially regress:
reject teacher-consensus ranking.

**E** — full DEV_READY:
freeze and open separate confirmation before external Laya/Jev evaluation.

## Stop rule

After one S58 DEV:
- no teacher swap
- no threshold/margin/coefficient sweep
- no consensus-mask variant
- no architecture/capacity change
- no native retraining
- no selector change
- no retry
- no gate weakening
- no second S58 DEV
- no external Laya/Jev evaluation.

Scientific failure is valid.
