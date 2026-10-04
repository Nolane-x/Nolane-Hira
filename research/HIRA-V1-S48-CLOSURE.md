# HIRA V1 S48 closure — Query-Quotient Option Evidence

Status: **SCIENTIFICALLY CLOSED — DOMINATED NEGATIVE / POST-DEV DIAGNOSTIC FAILURE RECOVERED**

Issue: #279  
PR: #280

## Qualified A0

- run `37171579714`
- artifact `11291393129`
- digest `sha256:de5ae3d7c936539c19d4b1a9a695fbd27a44fbeac7d1f5d46f26d5ab959090e7`
- outcome `HIRA_V1_S48_A0_QUERY_QUOTIENT_OPTION_EVIDENCE_READY`

A0 established:
- reference/treatment correction capacity both **114,688**
- total treatment **163,840**
- quotient learned params **0**
- K=3/7/255
- option permutation equivariance
- orthogonal query nuisance removal
- no raw-query bypass
- one encoder pass
- exact native/correction ownership
- W -> B -> A warm-start.

## Fresh scientific court

Run: `37172187841`  
Scientific head: `dbfea4876d9592e69777eebb906f5f18b14272fe`

Both arms completed all **24 epochs** and exposed DEV metrics.

The workflow then failed only in the post-training quotient diagnostic due to a count assertion. This is a **post-DEV mechanical failure**, so the scientific budget is consumed and no fresh rerun is allowed.

Recovered exact selection:
- raw-query reference: **epoch 13**
- query-quotient treatment: **epoch 23**

Native runtime trajectory:
- all **24/24** epoch hashes identical.

## Treatment minus matched reference

Correctness:
- fused canonical **-10.94 pp**
- fused paraphrase **-17.19 pp**
- paired both-correct **-6.25 pp**
- relation canonical **-5.73 pp**
- relation paraphrase **-20.05 pp**

Stability:
- fused selected-choice agreement **-10.16 pp**
- fused JS **+0.00270** worse
- relation selected-choice agreement **-13.02 pp**
- relation JS **-0.01542** lower
- signature cosine **-0.03270**
- signature discrimination margin **+0.04143**

## Frozen interpretation

The preregistered A/B/C/D table did not explicitly name a treatment that loses both correctness and selected-choice stability.

Therefore S48 is frozen as:

**DOMINATED NEGATIVE — outside A/B/C/D.**

The key scientific result is important:

> Lower relation-distribution JS is not sufficient for semantic stability.

The quotient treatment makes relation probability distributions numerically closer across views, yet the identity of the selected option becomes **less** stable and correctness collapses.

This means:
- score-level consistency is not the core target;
- raw-query nuisance removal is not sufficient;
- the correction branch is consuming a representation whose **option identity itself is already query-conditioned**.

## Mechanical repair

The post-DEV diagnostic expected `2 * semantic_cases` quotient rows, but canonical and paraphrase tensors each contain two relation queries per semantic case. The correct count is `4 * semantic_cases`.

The assertion has been repaired and regression-guarded **for code correctness only**.

It must not be used to rerun S48 DEV.

## Stop-rule compliance

No:
- quotient blend
- covariance-power sweep
- raw-query bypass
- learned projector
- quotient rank/dimension/epsilon sweep
- alternate checkpoint rule
- loss-weight/capacity/optimizer change
- second encoder
- scientific retry
- gate weakening
- second S48 DEV
- external Laya/Jev evaluation.

Frozen evidence:
- `research/HIRA-V1-S48-RECOVERED-RECEIPT.json`
- `research/HIRA-V1-S48-MATCHED-RECEIPT.md`
- `research/HIRA-V1-S48-POSTDEV-MECHANICAL-FAILURE-37172187841.md`

Next family:
**S49 — Private Query-Free State–Option Identity Signature**.
