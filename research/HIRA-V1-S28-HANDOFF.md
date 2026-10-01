# HIRA V1 S28 handoff — to S29 Query-Explicit Role Binding

S28 is frozen as:

`HIRA_V1_S28_ANCHOR_FACTOR_TRANSPORT_DEV_FAIL`

Authority:
- A0 run `36867793857`
- A0 artifact `11164614334`
- A0 digest `sha256:b3033c40357ba4b9b8cca378377a6d7e87f27f7e8449489e0beef20eb7fdec22`
- TRAIN/DEV run `36872373034`
- artifact `11168765249`
- artifact digest `sha256:5a5e28ea906cd5b86bff11177818382a1487934ab7ed99d53eaacf1e8a67bb5e`
- selected epoch **20**
- checkpoint `b9e0f59b5e624981db79f6e0e2525fe4f350ddfaf727164fdd8aefd2717fd8cd`

Selected DEV:
- fused canonical/paraphrase **0.5260416667 / 0.734375**
- paired **0.203125**
- question-swap **0.453125**
- fused agreement **0.5026041667**
- fused JS **0.0705156003**
- fused margins **-0.1302195030 / +0.3248527013**
- relation canonical/paraphrase **0.4296875 / 0.4583333333**
- relation margins **-0.0796973606 / -0.0394903719**
- relation agreement **0.6015625**
- final-signature cosine **0.6421900640**
- final-signature discrimination margin **-0.0111360058**
- role-anchor cosine **0.9185716013**
- role-anchor discrimination margin **-0.0099708166**
- value-anchor cosine **0.8572883606**
- value-anchor discrimination margin **-0.0366763414**

Best residual evidence:
- role-anchor cosine **0.9266317983**
- value-anchor cosine **0.8881069670**
- role discrimination margin never positive; best **-0.0088916677**
- value discrimination margin never positive; best **-0.0205161761**
- final-signature discrimination margin best only **0.0020386775**

## Stop rule applied

Do not tune S28 transport strength, factor weights, margin, negative mining, seed, LR or epochs.

## S29 target

**Query-Explicit Role Binding**

Why:
S28 proves wording-equivalent anchors can align strongly while different semantic queries remain insufficiently separated. The missing signal is explicit query identity in the relation score.

Freeze before A0:
- S21 primary unchanged
- shared A13 LoRA + private projection ownership unchanged
- exact physical trainable surface **81,920** unless separately preregistered otherwise
- value/content relation component unchanged
- S14 equal standardized full-K fusion unchanged
- no learned router/gate/calibrator

Candidate structural intervention:
1. derive a normalized query-role anchor from relation-projected question tokens;
2. retain S26 state-role anchor and option-role anchor;
3. role evidence becomes a fixed symmetric query-conditioned score, e.g. equal contribution from query↔state and query↔option compatibility;
4. value/content evidence remains the S26 factorized pair evidence;
5. final role/value combination is fixed before A0;
6. relation signature must expose the query-role relation without option-specific parameters.

A0 must include:
- correct role + correct value
- correct role + wrong value
- wrong role + same value
- wrong role + wrong value
- direct question-swap response
- paraphrase stability
- option permutation
- zero added learned state
- full-K/state-once
- S25 gradient ownership
- checkpoint replay

Use wholly fresh S29 authority.
No S28 DEV rows may enter S29 TRAIN/DEV.
No Laya/Jev benchmark until DEV_READY.
