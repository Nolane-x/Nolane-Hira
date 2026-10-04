# HIRA V1 S48 recovered matched receipt

Status: **SCIENTIFIC DEV CONSUMED / RECOVERED FROM SEALED RUN LOG**

Scientific run: `37172187841`  
Scientific head: `dbfea4876d9592e69777eebb906f5f18b14272fe`

The workflow failed **after both matched arms completed all 24 epochs and exposed DEV metrics**. The failure occurred only in the post-training quotient diagnostic:

`RuntimeError: S48 quotient diagnostic count changed`

Therefore this is **not** a pre-TRAIN mechanical abort and no second S48 DEV is allowed.

## Recovery authority

The run log contains exactly **48** `HIRA_V1_S45_ARM_EPOCH` records:
- first 24 = raw-query private-correction reference;
- second 24 = S48 query-quotient treatment.

Checkpoint recovery uses the exact preregistered S17/S35 selection key:
1. fused paired-both-correct;
2. fused canonical accuracy;
3. canonical relation accuracy;
4. canonical relation margin;
5. fused canonical margin;
6. question-swap change;
7. fused cross-view agreement;
8. same-option signature cosine;
9. signature same-vs-wrong margin;
10. negative canonical decision loss;
11. earlier epoch tie-break.

Recovered selected checkpoints:
- reference raw-query: **epoch 13**
- treatment quotient-query: **epoch 23**

Native runtime hashes are identical at **all 24 epochs**.

## Reference raw-query selected DEV

- fused canonical **0.5208333**
- fused paraphrase **0.5677083**
- paired both-correct **0.2552083**
- question-swap **0.6302083**
- fused agreement **0.5885417**
- fused JS **0.0463425**
- canonical/paraphrase margins **-0.006132 / +0.157610**
- relation canonical **0.3593750**
- relation paraphrase **0.4895833**
- relation agreement **0.5156250**
- relation JS **0.0247554**
- relation canonical/paraphrase margins **-0.306333 / -0.009445**
- signature cosine **0.9286424**
- signature same-vs-wrong margin **0.1040284**

## Treatment query-quotient selected DEV

- fused canonical **0.4114583**
- fused paraphrase **0.3958333**
- paired both-correct **0.1927083**
- question-swap **0.5781250**
- fused agreement **0.4869792**
- fused JS **0.0490409**
- canonical/paraphrase margins **-0.108505 / -0.213580**
- relation canonical **0.3020833**
- relation paraphrase **0.2890625**
- relation agreement **0.3854167**
- relation JS **0.0093396**
- relation canonical/paraphrase margins **-0.195329 / -0.201118**
- signature cosine **0.8959390**
- signature same-vs-wrong margin **0.1454556**

## Treatment minus reference

Correctness:
- fused canonical **-10.94 pp**
- fused paraphrase **-17.19 pp**
- paired both-correct **-6.25 pp**
- relation canonical **-5.73 pp**
- relation paraphrase **-20.05 pp**

Stability:
- fused agreement **-10.16 pp**
- fused JS **+0.00270** worse
- relation agreement **-13.02 pp**
- relation JS **-0.01542** numerically lower
- signature cosine **-0.03270**
- signature same-vs-wrong margin **+0.04143**

## Frozen interpretation

The preregistered A/B/C/D table did not explicitly name the case where **both correctness and selected-choice stability degrade**.

Therefore the honest classification is:

**DOMINATED NEGATIVE — outside A/B/C/D, scientifically closed.**

S48 does reduce relation-logit JS, but this does **not** translate into stable option ordering:
- relation selected-choice agreement falls by 13.02 pp;
- fused selected-choice agreement falls by 10.16 pp;
- correctness collapses strongly.

This is evidence that the option-difference query quotient removes information needed for correct ranking while failing to stabilize the ordering that matters.

No second S48 DEV is authorized.
