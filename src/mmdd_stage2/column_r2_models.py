"""Frozen-prior residual column verification with bundle or set evidence."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .column_data import digest
from .column_metrics import column_order


def mix_condition(seed: int, epoch: int, query_id: str, target_id: str) -> str:
    return 'O-O' if int(digest([seed, epoch, query_id, target_id]), 16) % 2 == 0 else 'O-R'


def shortlist(prior_logits: torch.Tensor, columns: list[int]) -> list[int]:
    return column_order(prior_logits.detach().cpu().tolist(), columns)[:3]


def complete_scores(prior: torch.Tensor, corrected: torch.Tensor, positions: list[int], columns: list[int]) -> torch.Tensor:
    """Rerank the fixed shortlist, retaining every other column in prior order."""
    result = prior.clone()
    result[positions] = corrected
    outside = [i for i in range(len(columns)) if i not in positions]
    if outside:
        # Outside scores are ranking-only sentinels, not verifier logits/probabilities.
        ceiling = min(float(corrected.min()), float(prior[outside].min())) - 1.
        order = column_order(prior.detach().cpu().tolist(), columns)
        for rank, i in enumerate(j for j in order if j in outside):
            result[i] = ceiling - rank
    return result


class EvidenceCorrection(nn.Module):
    def __init__(self, hidden_dim: int, separate: bool) -> None:
        super().__init__()
        self.separate = separate
        self.projection = nn.Sequential(nn.Linear(2*hidden_dim, 256), nn.LayerNorm(256), nn.GELU())
        if separate:
            self.modality = nn.Embedding(2, 16)
            self.phi = nn.Sequential(nn.Linear(4*256+16, 256), nn.GELU())
            self.attention = nn.Linear(256, 1, bias=False)
        self.correction = nn.Sequential(nn.Linear(512 if separate else 1024, 256), nn.GELU(),
                                        nn.Dropout(.1), nn.Linear(256, 1))
        self.beta = nn.Parameter(torch.tensor(0.))

    def item_states(self, no_e: torch.Tensor, evidence: torch.Tensor,
                    modalities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p0 = self.projection(no_e)
        pj = self.projection(evidence)
        base = p0.unsqueeze(0).expand_as(pj)
        mod = self.modality(modalities).unsqueeze(1).expand(-1, p0.shape[0], -1)
        z = self.phi(torch.cat([base, pj, pj-base, pj*base, mod], dim=-1))
        return p0, z, self.attention(z).squeeze(-1)

    def forward(self, prior_logits: torch.Tensor, no_e: torch.Tensor, evidence: torch.Tensor,
                modalities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if evidence.shape[0] == 0:
            return prior_logits, None
        if self.separate:
            p0, z, attention = self.item_states(no_e, evidence, modalities)
            weights = attention.softmax(dim=0)
            value = (weights.unsqueeze(-1)*z).sum(dim=0)
            x = torch.cat([p0, value], dim=-1)
        else:
            p0, pb = self.projection(no_e), self.projection(evidence[0])
            x = torch.cat([p0, pb, pb-p0, pb*p0], dim=-1)
            attention = None
        delta = self.correction(x).squeeze(-1)
        return prior_logits + F.softplus(self.beta)*delta, attention
