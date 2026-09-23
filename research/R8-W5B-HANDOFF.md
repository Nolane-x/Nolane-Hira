# R8-W5b handoff — token-aware semantic comparison ablation

Status: **implementation active under issue #61; no W5b empirical result yet.**

## Scientific hypothesis

W5a failed with pooled option/state semantics despite healthy high-K mechanics.

W5b tests whether HIRA's already-existing token-aware relation path fixes this semantic bottleneck.

No new trainable parameters are added.

## Backward-compatible runtime change

CompiledSchema now supports optional:
- option_token_embeddings;
- option_token_mask.

Runtime relation modes:
- pooled;
- option_tokens;
- state_tokens;
- dual_tokens.

Default remains pooled.

Therefore all previous heads and runtime calls retain their old behavior unless a token-aware mode is explicitly requested.

## Frozen W5b authorities

TRAIN:
- 512 fresh generated cases;
- seed 61001;
- K=8/16/32/64/128/255;
- new vocabulary/templates disjoint from W5a.

DEV:
- 160 fresh cases;
- seed 62002;
- same vocabulary pools but held-out combinations/templates;
- only selection authority.

CONFIRM:
- 192 cases;
- seed 63003;
- K=32/64/128/255;
- completely disjoint vocabulary/templates;
- generated only after one mode/head is frozen.

Forbidden:
- Banking77;
- typed final;
- W4b cases/vocab/templates;
- W5a cases/vocab/templates/seeds;
- MASSIVE/XNLI eval;
- Laya/Jev final examples.

## Candidate ablation

Exactly four candidates:
- pooled;
- option_tokens;
- state_tokens;
- dual_tokens.

Shared:
- W3 selected init SHA 2505e2cf99d741e590ff26ff7c70a587065e5713a4c6a3e53b6783a03b20446c;
- lr 3e-4;
- 8 epochs;
- AdamW wd 0.01;
- seed 101;
- CE 1.0 + hard Brier 0.1;
- frozen A13;
- 422,159 HIRA params;
- full-K forced budget 255.

## DEV selection

Lexicographic:
1. accuracy;
2. K128 accuracy;
3. K255 accuracy;
4. top5;
5. MRR;
6. lower Brier;
7. earlier epoch.

Mode tie order after every metric ties:
dual_tokens > state_tokens > option_tokens > pooled.

## CONFIRM PASS

TOKEN_SEMANTIC_PASS requires:
- overall >= 0.60;
- K128 >= 0.40;
- K255 >= 0.30;
- K255 top5 >= 0.70;
- probability mass <= 1e-6;
- budget exactly K;
- state-once;
- mechanism-evidence gate.

Mechanism gate:
- selected mode must be non-pooled;
- selected DEV accuracy - pooled DEV accuracy >= +0.10.

W5b populates zero campaign cells.
