# HIRA V1 S51 matched scientific receipt — Artifact-Pinned Private Court

Status: **FROZEN / CASE C**

Scientific run: `37197889785`  
Artifact: `11301721375`  
Artifact digest: `sha256:07cb321dd6f390bea5685f9c0786069ca77e4be757960c55bd3cbad8f21f7d43`  
Scientific head: `bf2f1ebd08e2361d7cc2cc58d78affceebb8b8dc`

Outcome:
`HIRA_V1_S51_ARTIFACT_PINNED_PRIVATE_COURT_DEV_COMPLETE`

## Authority chain

S51-A0:
- run `37192065019`
- artifact `11299312438`

Persisted native authority:
- run `37192490832`
- artifact `11299783210`
- epoch **24**
- runtime/native digest:
  `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA:
  `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`

Phase B loaded the exact same runtime/native digest.
Native optimizer constructed: **false**.
Native retraining: **false**.
Native trainable params: **0**.

Shared cache:
- TRAIN digest `6f295cf263e5c806b7b02f44877212456d1df75aadaed77c0ecc3b965c66cf20`
- DEV digest `660d1624d2a3c85ace86f81656750a6b7ebf4c7f45e0300644de1425e93b734a`
- reference/treatment same cache bytes: **true**
- private state-view encodes: **0**
- cache regeneration after DEV: **false**.

## Selected checkpoints

Reference selected epoch: **23**

Treatment selected epoch: **22**

## Reference — question-conditioned native relation signature

- fused canonical accuracy **0.4973958333333333**
- fused paraphrase accuracy **0.5572916666666666**
- paired both-correct **0.2552083333333333**
- fused selected-choice agreement **0.6770833333333334**
- fused JS **0.031627295150732**
- canonical margin **-0.06660363419602315**
- paraphrase margin **0.09597101404021184**
- relation canonical accuracy **0.4557291666666667**
- relation paraphrase accuracy **0.5052083333333334**
- relation agreement **0.6380208333333334**
- relation JS **0.01961116698415329**
- same-option signature cosine **0.9434975932041804**
- same-vs-wrong signature margin **0.17891620906690756**

## Treatment — query-free state↔option identity

- fused canonical accuracy **0.5182291666666666**
- fused paraphrase accuracy **0.578125**
- paired both-correct **0.3020833333333333**
- fused selected-choice agreement **0.671875**
- fused JS **0.03629013880466422**
- canonical margin **0.0725939900924762**
- paraphrase margin **0.040400502271950245**
- relation canonical accuracy **0.4973958333333333**
- relation paraphrase accuracy **0.46875**
- relation agreement **0.5286458333333334**
- relation JS **0.022787605101863544**
- identity same-option cosine **0.9438342799743017**
- identity same-vs-wrong margin **0.17771565603713194**

## Treatment minus reference

Correctness:
- fused canonical **2.08 pp**
- fused paraphrase **2.08 pp**
- paired both-correct **4.69 pp**
- canonical relation accuracy **4.17 pp**
- paraphrase relation accuracy **-3.65 pp**

Stability:
- fused selected-choice agreement **-0.52 pp**
- fused JS **+0.004663** worse
- relation selected-choice agreement **-10.94 pp**
- relation JS **+0.003176** worse

Margins:
- fused canonical **+0.139198**
- fused paraphrase **-0.055571**
- relation canonical **+0.087629**
- relation paraphrase **-0.073577**

## Treatment identity diagnostics

- cross-state-view same-option cosine **0.9438342799743017**
- same-vs-strongest-wrong margin **0.17771565603713194**
- fixed-identity raw-query swap mean abs logit delta **0.37688538742562133**

## Frozen interpretation

**Case C — useful correctness remains/improves, but selected-choice stability does not materially improve.**

The query-free identity treatment improves fused canonical/paraphrase accuracy by about **+2.08 pp** each and paired both-correct by **+4.69 pp**, while fused selected-choice agreement changes by only **-0.52 pp** and fused JS worsens.

More importantly, corrected relation selected-choice agreement falls by **10.94 pp**.

The identity geometry itself is stable (same-option cosine ~0.944, same-vs-wrong margin ~0.178), yet changing the raw query still moves the final readout strongly. Therefore simply removing query dependence from option identity is **not the dominant solution** to cross-view instability.

S51 closes the S49–S51 query-free identity family as insufficient by itself.

No second S51 DEV is authorized.
