from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class HIRAOutput:
    logits: Tensor
    probabilities: Tensor
    coarse_logits: Tensor
    selected_indices: Tensor
    selected_mask: Tensor
    relation_delta: Tensor
    candidate_budget: Tensor
    budget_logits: Tensor
    tail_mass: Tensor
    state_attention: Tensor


class HIRACore(nn.Module):
    """State-once relation core.

    Semantic encoding happens outside this module. Coarse scores are computed for every
    logical option. Relation deltas are computed only for selected candidates and scattered
    back into full-K logit space. Unselected logical-option logits are preserved exactly.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_rel: int = 128,
        n_heads: int = 4,
        budget_buckets=(4, 8, 16, 32, 64, 128, 255),
        dropout: float = 0.05,
    ):
        super().__init__()
        self.d_rel = d_rel
        self.budget_buckets = tuple(map(int, budget_buckets))
        self.q_proj = nn.Linear(d_model, d_rel, bias=False)
        self.seg_proj = nn.Linear(d_model, d_rel, bias=False)
        self.opt_proj = nn.Linear(d_model, d_rel, bias=False)
        self.token_proj = nn.Linear(d_model, d_rel, bias=False)
        self.type_emb = nn.Embedding(3, d_rel)
        self.state_ln = nn.LayerNorm(d_rel)
        self.option_ln = nn.LayerNorm(d_rel)

        self.coarse_scale = nn.Parameter(torch.tensor(10.0))
        self.coarse_bias = nn.Sequential(
            nn.Linear(d_rel * 4, d_rel),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_rel, 1),
        )

        self.option_token_weight = nn.Sequential(
            nn.Linear(d_rel, d_rel // 2),
            nn.GELU(),
            nn.Linear(d_rel // 2, 1),
        )
        self.late_scale = nn.Parameter(torch.tensor(3.0))

        self.cross_attn = nn.MultiheadAttention(
            d_rel, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_ln = nn.LayerNorm(d_rel)
        self.cross_ff = nn.Sequential(
            nn.Linear(d_rel, d_rel * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_rel * 2, d_rel),
        )
        self.cross_score = nn.Sequential(
            nn.Linear(d_rel * 4, d_rel),
            nn.GELU(),
            nn.Linear(d_rel, 1),
        )

        self.budget_gate = nn.Sequential(
            nn.Linear(5 + d_rel, d_rel),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_rel, len(self.budget_buckets)),
        )
        self.delta_scale = nn.Embedding(3, 1)
        nn.init.ones_(self.delta_scale.weight)

    @staticmethod
    def _masked_softmax(logits: Tensor, mask: Tensor | None) -> Tensor:
        if mask is not None:
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _normalized_entropy(p: Tensor, mask: Tensor | None) -> Tensor:
        eps = torch.finfo(p.dtype).eps
        if mask is None:
            k = torch.full_like(p[:, 0], float(p.shape[-1]))
            q = p
        else:
            q = p * mask.to(p.dtype)
            k = mask.sum(-1).clamp_min(2).to(p.dtype)
        h = -(q.clamp_min(eps) * q.clamp_min(eps).log()).sum(-1)
        return h / k.log().clamp_min(eps)

    def _budget_features(
        self, p0: Tensor, coarse: Tensor, state: Tensor, option_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        valid = torch.ones_like(coarse, dtype=torch.bool) if option_mask is None else option_mask
        masked = coarse.masked_fill(~valid, torch.finfo(coarse.dtype).min)
        top2 = masked.topk(k=min(2, masked.shape[-1]), dim=-1).values
        margin = torch.zeros_like(top2[:, 0]) if top2.shape[-1] == 1 else top2[:, 0] - top2[:, 1]
        max_p = p0.max(-1).values
        entropy = self._normalized_entropy(p0, valid)
        count = valid.sum(-1).to(coarse.dtype).clamp_min(1)
        mean = coarse.masked_fill(~valid, 0).sum(-1) / count
        var = (((coarse - mean[:, None]) ** 2).masked_fill(~valid, 0).sum(-1) / count)
        std = var.sqrt()
        k_ratio = count / float(max(self.budget_buckets))
        scalars = torch.stack([max_p, entropy, margin, std, k_ratio], dim=-1)
        return self.budget_gate(torch.cat([scalars, state], dim=-1)), count.long()

    def _select_candidates(
        self,
        coarse: Tensor,
        option_mask: Tensor | None,
        budget_logits: Tensor,
        valid_count: Tensor,
        forced_budget: int | None,
        adaptive_budget: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, k = coarse.shape
        if forced_budget is not None:
            budgets = torch.full(
                (batch,), min(int(forced_budget), k), device=coarse.device, dtype=torch.long
            )
        elif adaptive_budget:
            bucket_values = torch.tensor(self.budget_buckets, device=coarse.device, dtype=torch.long)
            budgets = bucket_values[budget_logits.argmax(-1)].clamp_max(k)
        else:
            # Fail-safe before the budget gate is calibrated/trained.
            budgets = torch.full((batch,), k, device=coarse.device, dtype=torch.long)
        budgets = torch.minimum(budgets, valid_count.clamp_min(1))
        max_budget = int(budgets.max().item())
        masked = coarse if option_mask is None else coarse.masked_fill(~option_mask, -1e4)
        indices = masked.topk(max_budget, dim=-1).indices
        ranks = torch.arange(max_budget, device=coarse.device)[None, :]
        return indices, ranks < budgets[:, None], budgets

    @staticmethod
    def _gather_options(x: Tensor, indices: Tensor) -> Tensor:
        return x.gather(1, indices[..., None].expand(-1, -1, x.shape[-1]))

    @staticmethod
    def _gather_option_tokens(x: Tensor, indices: Tensor) -> Tensor:
        _, _, t, d = x.shape
        return x.gather(1, indices[..., None, None].expand(-1, -1, t, d))

    def _relation_delta(
        self,
        q: Tensor,
        state: Tensor,
        seg: Tensor,
        selected_options: Tensor,
        qtype: Tensor,
        selected_mask: Tensor,
        *,
        segment_mask: Tensor | None = None,
        selected_option_tokens: Tensor | None = None,
        selected_option_token_mask: Tensor | None = None,
    ) -> Tensor:
        query = selected_options + (q + state)[:, None, :]
        key_padding_mask = None if segment_mask is None else ~segment_mask
        attn_out, _ = self.cross_attn(
            query, seg, seg, key_padding_mask=key_padding_mask, need_weights=False
        )
        h = self.cross_ln(query + attn_out)
        h = self.cross_ln(h + self.cross_ff(h))

        if selected_option_tokens is None:
            token_summary = selected_options
        else:
            tok = self.token_proj(selected_option_tokens)
            weights = self.option_token_weight(tok).squeeze(-1)
            if selected_option_token_mask is not None:
                weights = weights.masked_fill(~selected_option_token_mask, -1e4)
            weights = torch.softmax(weights, dim=-1)
            token_summary = torch.einsum("brt,brtd->brd", weights, tok)

        state_expand = state[:, None, :].expand_as(selected_options)
        features = torch.cat([h, selected_options, token_summary, state_expand], dim=-1)
        delta = self.cross_score(features).squeeze(-1)
        delta = delta * self.late_scale.clamp(0.0, 20.0)
        delta = delta * self.delta_scale(qtype).squeeze(-1)[:, None]
        return delta * selected_mask.to(delta.dtype)

    def forward(
        self,
        question: Tensor,
        segments: Tensor,
        options: Tensor,
        qtype: Tensor,
        option_mask: Tensor | None = None,
        *,
        segment_mask: Tensor | None = None,
        option_tokens: Tensor | None = None,
        option_token_mask: Tensor | None = None,
        forced_budget: int | None = None,
        adaptive_budget: bool = False,
    ) -> HIRAOutput:
        if question.ndim != 2 or segments.ndim != 3 or options.ndim != 3:
            raise ValueError("expected question [B,D], segments [B,S,D], options [B,K,D]")
        if options.shape[1] < 2:
            raise ValueError("HIRA requires at least two logical options")
        if qtype.min().item() < 0 or qtype.max().item() > 2:
            raise ValueError("qtype must be 0(choice), 1(score), or 2(noul)")

        q = self.q_proj(question) + self.type_emb(qtype)
        seg = self.seg_proj(segments)
        attn_scores = torch.einsum("bd,bsd->bs", q, seg) / (self.d_rel ** 0.5)
        if segment_mask is not None:
            attn_scores = attn_scores.masked_fill(~segment_mask, -1e4)
        state_attention = torch.softmax(attn_scores, dim=-1)
        state = self.state_ln(torch.einsum("bs,bsd->bd", state_attention, seg) + q)

        o = self.option_ln(self.opt_proj(options))
        ctx = F.normalize(q + state, dim=-1)
        on = F.normalize(o, dim=-1)
        c = ctx[:, None, :].expand_as(o)
        coarse = torch.einsum("bd,bkd->bk", ctx, on)
        coarse = coarse * self.coarse_scale.clamp(0.1, 100.0)
        coarse = coarse + self.coarse_bias(
            torch.cat([o, c, o * c, (o - c).abs()], dim=-1)
        ).squeeze(-1)
        if option_mask is not None:
            coarse = coarse.masked_fill(~option_mask, -1e4)

        p0 = self._masked_softmax(coarse, option_mask)
        budget_logits, valid_count = self._budget_features(p0, coarse, state, option_mask)
        indices, selected_mask, budgets = self._select_candidates(
            coarse, option_mask, budget_logits, valid_count, forced_budget, adaptive_budget
        )

        selected_options = self._gather_options(o, indices)
        selected_tokens = None
        selected_token_mask = None
        if option_tokens is not None:
            selected_tokens = self._gather_option_tokens(option_tokens, indices)
            if option_token_mask is not None:
                selected_token_mask = option_token_mask.gather(
                    1,
                    indices[..., None].expand(-1, -1, option_token_mask.shape[-1]),
                )

        selected_delta = self._relation_delta(
            q,
            state,
            seg,
            selected_options,
            qtype,
            selected_mask,
            segment_mask=segment_mask,
            selected_option_tokens=selected_tokens,
            selected_option_token_mask=selected_token_mask,
        )

        current = coarse.gather(1, indices)
        logits = coarse.scatter(1, indices, current + selected_delta)
        if option_mask is not None:
            logits = logits.masked_fill(~option_mask, -1e4)

        p = self._masked_softmax(logits, option_mask)
        relation_delta = torch.zeros_like(coarse).scatter(1, indices, selected_delta)

        reranked = torch.zeros_like(p, dtype=torch.bool)
        reranked.scatter_(1, indices, selected_mask)
        if option_mask is not None:
            reranked |= ~option_mask
        tail = p.masked_fill(reranked, 0).sum(-1)

        return HIRAOutput(
            logits=logits,
            probabilities=p,
            coarse_logits=coarse,
            selected_indices=indices,
            selected_mask=selected_mask,
            relation_delta=relation_delta,
            candidate_budget=budgets,
            budget_logits=budget_logits,
            tail_mass=tail,
            state_attention=state_attention,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
