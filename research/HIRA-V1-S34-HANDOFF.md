# HIRA V1 S34 handoff — to S35 Native A13 Relation Geometry Court

S34 is frozen as:

`HIRA_V1_S34_MATCHED_TRANSPORT_DEV_COMPLETE`

## Authority

A0:
- run `36998495974`
- artifact `11222861913`
- digest `sha256:239c2c5964a30d4ed8ca67a04e6b9ed5f4aea90efb80609368d126487fe044d9`

Fresh matched TRAIN/DEV:
- run `37000789652`
- artifact `11225542519`
- digest `sha256:18228340feeee036e354d26eb183840b59525968aa6d312e0e8d429985cfba0e`
- scientific head `10753c1cbd1ee91e89a357284779300ea53e6ee7`

Control selected:
- epoch **22**
- fused canonical/paraphrase **0.4973958 / 0.5390625**
- relation canonical/paraphrase **0.4557292 / 0.4010417**
- signature cosine/margin **0.9228848 / 0.1886656**
- gates **15/22 PASS**

Transport selected:
- epoch **8**
- fused canonical/paraphrase **0.3463542 / 0.3098958**
- relation canonical/paraphrase **0.25 / 0.2552083**
- signature cosine/margin **0.8784866 / 0.1775761**
- gates **14/22 PASS**

## Frozen conclusion

S34 is preregistered **Case D**.

Many-to-many query-conditioned transport is mechanically valid but substantially worsens fresh relation and fused semantics inside the inherited 128D shared-projection geometry.

Do not tune S34.

## S35 hypothesis

**Native A13 Relation Geometry Court**

Matched arms share exact S17:
- A13 final-attention LoRA **16,384**
- shared primary projection **32,768**
- total trainable **49,152**
- primary scorer/fusion/optimizer unchanged

Control:
- exact S13 projected 128D relation operator.

Treatment:
- S13-equivalent role/pair relation logic directly on adapted A13 **256D** state/question/option tokens;
- no projection in relation path;
- signature width **256**;
- relation logits produced from native cosine geometry;
- zero added learned params;
- local relation CE/canonicalization unchanged.

The shared 128D projection stays present and trainable for the primary S17 path only.

Key A0 discriminator:
- perturb the shared projection strongly;
- control relation logits/signatures must change;
- native treatment relation logits/signatures must remain **exactly unchanged**;
- primary logits must still depend on the projection.

Gradient ownership:
- treatment relation block -> shared projection gradient exactly **0**;
- treatment relation block -> LoRA nonzero;
- primary block -> projection nonzero.

This cleanly isolates whether relation information is lost by compact projection geometry.

Use fresh S35 A0/TRAIN/DEV only.
No S34 rows.
No width/normalization/temperature retry after exposure.
No second DEV.
No Laya/Jev reopening before DEV_READY.
