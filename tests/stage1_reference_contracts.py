"""Stage-1 CQET auditable math contracts, not an integrated training implementation."""
from __future__ import annotations
import hashlib
import math
from collections import defaultdict
from typing import Mapping, Sequence
import numpy as np
import torch
from torch import Tensor


def student_score(left: Tensor, relation: Tensor, right: Tensor) -> Tensor:
    """Row-vector form; right may be one vector or a batch."""
    if relation.ndim != 2 or relation.shape[0] != left.shape[-1] or relation.shape[1] != right.shape[-1]:
        raise ValueError('incompatible relation dimensions')
    return ((left @ relation) * right).sum(-1)


def ann_query(left: Tensor, relation: Tensor) -> Tensor:
    return left @ relation


def global_features(q: Tensor, t: Tensor, pair: Tensor,
                    e: Tensor | None = None, etype: Tensor | None = None) -> Tensor:
    if not (q.shape == t.shape == pair.shape):
        raise ValueError('shape mismatch')
    base = torch.cat([q, t, q*t, (q-t).abs(), pair], dim=-1)
    if e is None:
        if etype is not None:
            raise ValueError('empty evidence cannot carry type')
        extra = q.new_zeros((*q.shape[:-1], 6*q.shape[-1]))
    else:
        if etype is None or e.shape != q.shape or etype.shape != q.shape:
            raise ValueError('invalid evidence')
        extra = torch.cat([e, q*e, (q-e).abs(), e*t, (e-t).abs(), etype], dim=-1)
    return torch.cat([base, extra], dim=-1)


def _validate_paths(f0: Tensor, path: Tensor, target: Tensor) -> None:
    if f0.ndim != 1 or path.ndim != 1 or path.shape != target.shape or target.dtype != torch.long:
        raise ValueError('invalid path arrays')
    if target.numel() and (target.min().item() < 0 or target.max().item() >= len(f0)):
        raise ValueError('path target outside list')


def aggregate_cqet(f0: Tensor, path: Tensor, target: Tensor) -> Tensor:
    """Nonempty natural bag: LME QET only. Empty: f0."""
    _validate_paths(f0, path, target)
    result = []
    for i in range(len(f0)):
        scores = path[target == i]
        result.append(torch.logsumexp(scores, 0)-math.log(len(scores)) if len(scores) else f0[i])
    return torch.stack(result) if result else f0


def aggregate_lse(f0: Tensor, path: Tensor, target: Tensor) -> Tensor:
    _validate_paths(f0, path, target)
    return torch.stack([torch.logsumexp(torch.cat([f0[i:i+1], path[target == i]]), 0)
                        for i in range(len(f0))]) if len(f0) else f0


def rank_mass(scores: Tensor, positive: Tensor, valid: Tensor | None = None) -> Tensor | None:
    if scores.ndim != 1 or positive.shape != scores.shape or positive.dtype != torch.bool:
        raise ValueError('invalid positive mask')
    if valid is None:
        valid = torch.ones_like(positive)
    if valid.dtype != torch.bool or valid.shape != scores.shape or bool((positive & ~valid).any()):
        raise ValueError('invalid validity mask')
    if not bool(positive.any()) or not bool((valid & ~positive).any()):
        return None
    return torch.logsumexp(scores[valid], 0)-torch.logsumexp(scores[positive], 0)


def list_kd(student: Tensor, teacher: Tensor, valid: Tensor | None = None) -> Tensor | None:
    if student.ndim != 1 or teacher.shape != student.shape:
        raise ValueError('KL score shape mismatch')
    if not torch.is_grad_enabled() or not student.requires_grad:
        raise RuntimeError('KD must execute with Student gradient enabled')
    if valid is not None:
        if valid.dtype != torch.bool or valid.shape != student.shape:
            raise ValueError('invalid KL mask')
        student, teacher = student[valid], teacher[valid]
    if student.numel() < 2:
        return None
    t = torch.log_softmax(teacher.detach(), dim=0)
    s = torch.log_softmax(student, dim=0)
    return (t.exp()*(t-s)).sum()


def support_pair(pos: Tensor, competitor: Tensor) -> Tensor | None:
    if pos.ndim != 1 or competitor.ndim != 1:
        raise ValueError('support scores must be 1D')
    if not len(pos) or not len(competitor):
        return None
    diff = competitor[None, :]-pos[:, None]
    return torch.logsumexp(torch.cat([pos.new_zeros((len(pos),1)),diff], dim=1), dim=1).mean()


def hierarchical_mean(groups: Sequence[Sequence[Tensor]]) -> Tensor | None:
    means = [torch.stack(list(g)).mean() for g in groups if len(g)]
    return torch.stack(means).mean() if means else None


def normalize_content_hash(content_hash: str, asset_id: str) -> str:
    if content_hash == asset_id or len(content_hash) != 64:
        raise ValueError('real SHA256 required; asset IDs are not hashes')
    try:
        int(content_hash, 16)
    except ValueError as e:
        raise ValueError('invalid SHA256') from e
    return content_hash.lower()


def hash_order(ids: Sequence[str], namespace: str, seed: int, context: str = '') -> list[str]:
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate IDs in ordering input')
    return sorted(ids, key=lambda i: (hashlib.sha256(f'{namespace}|{seed}|{context}|{i}'.encode()).digest(), i.encode()))


def sample_competitors(ranked: Sequence[str], library: Sequence[str], protect: set[str],
                       hard_count: int, uniform_count: int, seed: int, anchor: str) -> dict:
    hard = list(dict.fromkeys(i for i in ranked if i not in protect))[:hard_count]
    candidates = sorted(set(library)-protect-set(hard), key=lambda s:s.encode())
    uniform = hash_order(candidates, 'UNIFORM', seed, anchor)[:uniform_count]
    return {'hard':hard,'uniform':uniform,'hard_shortfall':hard_count-len(hard),
            'uniform_shortfall':uniform_count-len(uniform)}


def rank_metrics(ids: Sequence[str], gold: set[str], k: int) -> dict[str, float | int]:
    if k < 1 or len(ids) != len(set(ids)):
        raise ValueError('k must be positive; rankings cannot contain duplicates')
    if not gold:
        raise ValueError('zero-gold query must be explicitly excluded, not silently scored 0')
    hit = len(set(ids[:k]) & gold)
    return {'hits':hit,'gold_count':len(gold),'recall':hit/len(gold),'hit_rate':float(hit>0)}


def pool_metrics(ids: Sequence[str], gold: set[str], oracle_k: int=10) -> dict[str,float|int]:
    if not gold or len(ids) != len(set(ids)) or oracle_k < 1:
        raise ValueError('invalid pool or gold')
    count = len(set(ids) & gold)
    return {'size':len(ids),'gold_in_pool':count,'coverage':count/len(gold),
            'hit_rate':float(count>0),'oracle':min(oracle_k,count)/len(gold)}


def source_group_bootstrap(deltas: Sequence[float], groups: Sequence[str],
                           replicates: int=10000, seed: int=20260925) -> dict:
    d = np.asarray(deltas,dtype=np.float64)
    if len(d)==0 or len(d)!=len(groups) or not np.isfinite(d).all() or replicates < 1:
        raise ValueError('invalid bootstrap inputs')
    by = defaultdict(list)
    for value, group in zip(d,groups):
        if not group:
            raise ValueError('source group required')
        by[group].append(float(value))
    keys = sorted(by, key=lambda x:x.encode())
    sums=np.array([sum(by[g]) for g in keys]); counts=np.array([len(by[g]) for g in keys])
    rng=np.random.default_rng(seed); vals=np.empty(replicates)
    for i in range(replicates):
        sampled=rng.integers(0,len(keys),size=len(keys))
        vals[i]=sums[sampled].sum()/counts[sampled].sum()
    return {'query_macro_delta':float(d.mean()), 'ci95':np.quantile(vals,[.025,.975]).tolist(),
            'n_queries':len(d),'n_source_groups':len(keys),'replicates':replicates,
            'bootstrap_mean_not_point_estimate':float(vals.mean()),
            'W':int((d>1e-12).sum()),'L':int((d< -1e-12).sum()),'T':int((np.abs(d)<=1e-12).sum())}


def state_digest(state: Mapping[str, Tensor]) -> str:
    h=hashlib.sha256()
    for key in sorted(state):
        value=state[key].detach().cpu().contiguous()
        h.update(key.encode()+b'\0'+str(value.dtype).encode()+b'\0'+str(tuple(value.shape)).encode()+b'\0')
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_stage_parent(model: torch.nn.Module, parent_state: Mapping[str, Tensor]) -> str:
    model.load_state_dict(parent_state, strict=True)
    actual, expected = state_digest(model.state_dict()), state_digest(parent_state)
    if actual != expected:
        raise RuntimeError('C2 step0 != selected C1')
    return actual


def p3_admission(qt_scores: Mapping[str,float], d1_scores: Mapping[str,float], budget: int=150) -> list[str]:
    if set(d1_scores)-set(qt_scores) or budget < 1:
        raise ValueError('all evidence targets need real QT scores')
    if not all(math.isfinite(v) for v in list(qt_scores.values())+list(d1_scores.values())):
        raise ValueError('nonfinite scores')
    qorder=sorted(qt_scores,key=lambda t:(-qt_scores[t],t.encode()))
    eorder=sorted(d1_scores,key=lambda t:(-d1_scores[t],t.encode()))
    qr={t:i+1 for i,t in enumerate(qorder)};er={t:i+1 for i,t in enumerate(eorder)}
    score={t:1/(60+qr[t])+(1/(60+er[t]) if t in er else 0) for t in qorder}
    return sorted(qorder,key=lambda t:(-score[t],t.encode()))[:budget]
