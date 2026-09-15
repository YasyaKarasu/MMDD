"""Correct per-query LSE and graph-bound Uniform for the conditional R26 extension."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .scoring import ListScores,TargetScores


def list_terms(scores: ListScores, teacher: ListScores | None = None) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    mask = scores.candidate_mask
    positive = scores.positive_mask
    if positive is None:
        positive = torch.zeros_like(mask)
        valid = scores.positive_indices >= 0
        positive[torch.arange(len(mask),device=mask.device)[valid],scores.positive_indices[valid]] = True
    positive = positive & mask
    active = positive.any(-1) & (mask & ~positive).any(-1)
    sup = scores.logits.sum(-1)*0
    kd = scores.logits.sum(-1)*0
    if active.any():
        logits = scores.logits[active].masked_fill(~mask[active],-torch.inf)
        sup[active] = torch.logsumexp(logits,-1)-torch.logsumexp(logits.masked_fill(~positive[active],-torch.inf),-1)
        if teacher is not None:
            if not torch.equal(mask[active],teacher.candidate_mask[active]):
                raise ValueError("QT Teacher mask must be exactly aligned with the Student branch")
            logp = F.log_softmax(logits,-1)
            logt = F.log_softmax(teacher.logits[active].masked_fill(~mask[active],-torch.inf),-1)
            kd[active] = (logt.exp()*(logt.masked_fill(~mask[active],0)-logp.masked_fill(~mask[active],0))).sum(-1)
    return sup,kd,active


def query_uniform(scores: ListScores, relations: list[str], owners: list[int], query_count: int) -> torch.Tensor:
    mask = scores.candidate_mask
    active = torch.tensor([r != "table->table" for r in relations],device=mask.device) & mask.any(-1)
    losses = scores.logits.sum(-1)*0
    if active.any():
        selected = mask[active]
        n = selected.sum(-1)
        logp = F.log_softmax(scores.logits[active].masked_fill(~selected,-torch.inf),-1)
        losses[active] = -logp.masked_fill(~selected,0).sum(-1)/n-n.float().log()
    owner = torch.tensor(owners,device=mask.device)
    sums = losses.new_zeros(query_count).scatter_add(0,owner,losses)
    # QT lists and inactive lists remain in each query's frozen denominator.
    counts = losses.new_zeros(query_count).scatter_add(0,owner,torch.ones_like(losses))
    return (sums/counts.clamp_min(1)).mean()


def objective(scores: TargetScores, *, family: str, teacher: TargetScores | None = None,
              kd_weight: float = 0., uniform_weight: float = 0., uniform_scores: ListScores | None = None,
              uniform_relations: list[str] | None = None, owners: list[int] | None = None,
              anchor: torch.Tensor | None = None) -> dict:
    if family not in ("split","lse"):
        raise ValueError(family)
    if kd_weight and teacher is None:
        raise ValueError("Nonzero KD requires fixed QT Teacher scores")
    if teacher is not None:
        overlap = teacher.direct.candidate_mask & teacher.evidence.candidate_mask
        if not torch.equal(teacher.direct.logits[overlap],teacher.evidence.logits[overlap]):
            raise ValueError("D and E require identical QT Teacher values")
    ds,dk,da = list_terms(scores.direct,None if teacher is None else teacher.direct)
    es,ek,ea = list_terms(scores.evidence,None if teacher is None else teacher.evidence)
    zero = scores.direct.logits.sum()*0
    if family == "split":
        sup,kd = (ds+es).mean(),(dk+ek).mean()
    else:
        d = scores.direct.logits.masked_fill(~scores.direct.candidate_mask,-torch.inf)
        e = scores.evidence.logits.masked_fill(~scores.evidence.candidate_mask,-torch.inf)
        mask = scores.direct.candidate_mask | scores.evidence.candidate_mask
        branches = torch.stack([d,e],-1).masked_fill(~mask[...,None],0)
        fused = torch.logsumexp(branches,-1).masked_fill(~mask,0)
        fs = ListScores(fused,mask,scores.direct.positive_indices,scores.direct.positive_mask)
        fteacher = None if teacher is None else ListScores(teacher.direct.logits,mask,fs.positive_indices,fs.positive_mask)
        per_sup,per_kd,_ = list_terms(fs,fteacher)
        units = da.to(fused.dtype)+ea.to(fused.dtype)
        sup,kd = (units*per_sup).mean(),(units*per_kd).mean()
    uniform = zero
    if uniform_weight:
        if uniform_scores is None or uniform_relations is None or owners is None:
            raise ValueError("Uniform needs the frozen per-query auxiliary graph")
        uniform = query_uniform(uniform_scores,uniform_relations,owners,len(ds))
    if anchor is None:
        anchor = zero
    return {"loss":sup+kd_weight*kd+uniform_weight*uniform+anchor,"supervised_loss":sup,"kd_loss":kd,
            "fused_supervised_loss":sup if family == "lse" else zero,"fused_kd_loss":kd if family == "lse" else zero,
            "uniform_loss":uniform,"weighted_anchor_loss":anchor,"direct_active":int(da.sum()),"evidence_active":int(ea.sum())}
