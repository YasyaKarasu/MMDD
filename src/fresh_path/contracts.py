"""Small, CPU-testable mathematical contracts for FRESH-PATH v2.

These functions are not the production MMDD pipeline or a substitute for its
required real-data integration tests. All example data in tests are synthetic.
"""
from __future__ import annotations
import hashlib
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def local_rng(namespace: str) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(namespace.encode('utf-8')).digest()[:8], 'big'))


def stable_rank(ids: Sequence[str], scores: Sequence[float]) -> list[str]:
    if len(ids) != len(scores) or len(ids) != len(set(ids)):
        raise ValueError('IDs and scores must be one-to-one')
    if not all(torch.isfinite(torch.tensor(s)).item() for s in scores):
        raise ValueError('non-finite ranking score')
    return [x for x, _ in sorted(zip(ids, scores), key=lambda p: (-p[1], p[0].encode('utf-8')))]


def pin_sets(legal: Iterable[str], positive: Iterable[str], q_positive: Iterable[str], e_positive: Iterable[str]) -> tuple[set[str], set[str], set[str]]:
    legal, p = set(legal), set(positive)
    if not p <= legal:
        raise ValueError('positive outside legal target universe')
    ignore = ((set(q_positive) | set(e_positive)) & legal) - p
    return p, ignore, legal - p - ignore


def masks(ids: Sequence[str], p: set[str], ignore: set[str], *, device=None) -> tuple[Tensor, Tensor]:
    if len(ids) != len(set(ids)) or p & ignore or not p <= set(ids):
        raise ValueError('invalid candidate membership or positive/ignore overlap')
    return (torch.tensor([x in p for x in ids], device=device, dtype=torch.bool),
            torch.tensor([x not in ignore for x in ids], device=device, dtype=torch.bool))


def rank_loss(logits: Tensor, positive: Tensor, allowed: Tensor | None = None, *,
              active: bool | None = None, validate_finite: bool = True,
              validate_masks: bool = True) -> Tensor | None:
    if logits.ndim != 1 or positive.shape != logits.shape or positive.dtype != torch.bool:
        raise ValueError('rank_loss expects aligned one-dimensional logits and Boolean positive mask')
    a = torch.ones_like(positive) if allowed is None else allowed
    if a.shape != positive.shape or a.dtype != torch.bool:
        raise ValueError('positive must be in allowed set')
    if validate_masks and torch.any(positive & ~a):
        raise ValueError('positive must be in allowed set')
    if active is False:
        return None
    if active is None and (not positive.any().item() or not (a & ~positive).any().item()):
        return None
    if validate_finite and not torch.isfinite(logits[a]).all():
        raise ValueError('non-finite allowed score')
    return torch.logsumexp(logits[a], 0) - torch.logsumexp(logits[positive], 0)


def streamed_full_loss(query: Tensor, keys: Tensor, positive: Tensor, allowed: Tensor, chunk: int, *,
                       active: bool | None = None,
                       chunk_activity: Sequence[tuple[bool, bool]] | None = None,
                       validate_finite: bool = True) -> Tensor | None:
    """Full denominator, not a mean of per-chunk losses; differentiable reference."""
    if query.ndim != 1 or keys.ndim != 2 or query.shape[0] != keys.shape[1]:
        raise ValueError('query/key dimensions')
    if chunk <= 0 or positive.shape != keys.shape[:1] or allowed.shape != positive.shape:
        raise ValueError('invalid chunk or masks')
    if torch.any(positive & ~allowed):
        raise ValueError('excluded positive')
    if active is False:
        return None
    if active is None and (not positive.any().item() or not (allowed & ~positive).any().item()):
        return None
    if chunk_activity is not None and len(chunk_activity) != (len(keys) + chunk - 1) // chunk:
        raise ValueError('chunk activity does not align with keys')
    alls, poss = [], []
    for chunk_no, start in enumerate(range(0, len(keys), chunk)):
        end = min(start + chunk, len(keys))
        score = keys[start:end] @ query
        aa, pp = allowed[start:end], positive[start:end]
        # Empty logsumexp entries are not inserted in differentiable logaddexp:
        # logaddexp(-inf,-inf) has undefined derivative.
        chunk_has_allowed, chunk_has_positive = (
            chunk_activity[chunk_no] if chunk_activity is not None else (None, None)
        )
        if chunk_has_allowed if chunk_activity is not None else aa.any().item():
            if validate_finite and not torch.isfinite(score[aa]).all():
                raise ValueError('non-finite allowed score')
            alls.append(torch.logsumexp(score[aa], 0))
        if chunk_has_positive if chunk_activity is not None else pp.any().item():
            poss.append(torch.logsumexp(score[pp], 0))
    return torch.logsumexp(torch.stack(alls), 0) - torch.logsumexp(torch.stack(poss), 0)


def kd_loss(student: Tensor, teacher: Tensor, allowed: Tensor, temperature: float = 2.0) -> Tensor:
    if temperature <= 0 or student.ndim != 1 or teacher.shape != student.shape or allowed.shape != student.shape:
        raise ValueError('KD dimensions/temperature')
    if not allowed.any().item():
        raise ValueError('empty KD comparison set')
    t = teacher.detach()[allowed] / temperature
    ss = student[allowed] / temperature
    tp = torch.softmax(t, dim=0)
    return temperature ** 2 * torch.sum(tp * (torch.log_softmax(t, dim=0) - torch.log_softmax(ss, dim=0)))


def clipped_query(base: Tensor, residual: Tensor, rho: float = 0.5, eps: float = 1e-12) -> Tensor:
    if base.shape != residual.shape or base.ndim < 1 or not 0 < rho < 1 or eps <= 0:
        raise ValueError('invalid residual trust-region arguments')
    radius = rho * torch.linalg.vector_norm(base, dim=-1, keepdim=True)
    raw_norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
    scale = torch.clamp(radius / (raw_norm + eps), max=1.0)
    return base + residual * scale


class ConditionalAdapter(nn.Module):
    def __init__(self, dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.hidden = nn.Linear(4*dim, hidden)
        self.output = nn.Linear(hidden, dim)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, q: Tensor, e: Tensor, base: Tensor, *, read_q: bool = True) -> Tensor:
        if q.shape != e.shape or base.shape != e.shape:
            raise ValueError('q/e/base shape mismatch')
        if read_q:
            h = torch.cat([q, e, q*e, torch.abs(q-e)], dim=-1)
        else:
            zero = torch.zeros_like(q)
            h = torch.cat([zero, e, zero, zero], dim=-1)
        return clipped_query(base, self.output(F.gelu(self.hidden(h))))


def projected(z: Tensor, p: Tensor) -> Tensor:
    return z @ p.T


def bilinear_query(u: Tensor, r: Tensor) -> Tensor:
    return u @ r


def target_path_lse(zero_hop: Tensor, evidence: Sequence[Tensor]) -> Tensor:
    if zero_hop.numel() != 1 or any(x.numel() != 1 for x in evidence):
        raise ValueError('one scalar per actual path required')
    return torch.logsumexp(torch.stack([zero_hop.reshape(()), *[x.reshape(()) for x in evidence]]), 0)


def unique_paths(paths: Iterable[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    return sorted(set(paths), key=lambda p: tuple(x.encode('utf-8') for x in p))


def augment_shared(query: str, targets: Sequence[str], graph: Mapping[str, Sequence[str]], evidence: str | None) -> dict[str, list[str]]:
    # No label-dependent choice of which target receives the new evidence.
    return {t: sorted(set(graph.get(t, ())) | ({evidence} if evidence is not None else set())) for t in targets}


def equal_rrf(direct: Sequence[str], evidence: Sequence[str], k: int = 60, budget: int | None = 100) -> list[str]:
    if k < 0 or (budget is not None and budget < 0) or len(set(direct)) != len(direct) or len(set(evidence)) != len(evidence):
        raise ValueError('invalid RRF inputs')
    score: dict[str, float] = {}
    for channel in (direct, evidence):
        for i, x in enumerate(channel, 1):
            score[x] = score.get(x, 0.0) + 1.0 / (k+i)
    result = sorted(score, key=lambda x: (-score[x], x.encode('utf-8')))
    return result if budget is None else result[:budget]


def fixed_bins(content_hidden: Tensor, limit: int = 64) -> Tensor:
    """Apply only to text/image content states; never call this on Query rows."""
    if content_hidden.ndim != 2 or len(content_hidden) < 1 or limit < 1:
        raise ValueError('nonempty content token matrix required')
    n = len(content_hidden)
    if n <= limit:
        return content_hidden.float()
    return torch.stack([content_hidden[(j*n)//limit:((j+1)*n)//limit].float().mean(0) for j in range(limit)])


def strict_eo(gold: set[str], evidence: set[str], d_ann: set[str], d_exact: set[str]) -> set[str]:
    return gold & evidence - (d_ann | d_exact)


def macro_recall(gold: Mapping[str, set[str]], ranking: Mapping[str, Sequence[str]], k: int) -> float:
    if not gold or k < 0 or any(not v for v in gold.values()):
        raise ValueError('nonempty gold population required')
    if set(ranking) - set(gold):
        raise ValueError('unknown query in rankings')
    return sum(len(set(ranking.get(q, ())[:k]) & p) / len(p) for q, p in gold.items()) / len(gold)


def wlt(gold: Mapping[str, set[str]], a: Mapping[str, Sequence[str]], b: Mapping[str, Sequence[str]], k: int, tol: float=1e-12) -> tuple[int,int,int]:
    win=loss=tie=0
    for q,p in gold.items():
        if not p: raise ValueError('empty gold')
        d=(len(set(a.get(q, ())[:k])&p)-len(set(b.get(q, ())[:k])&p))/len(p)
        win += d > tol
        loss += d < -tol
        tie += abs(d) <= tol
    return int(win),int(loss),int(tie)


def pair_query_macro(values: Sequence[tuple[str,float]]) -> float:
    buckets: dict[str,list[float]]={}
    for q,v in values: buckets.setdefault(q,[]).append(v)
    if not buckets: raise ValueError('empty probe')
    return sum(sum(v)/len(v) for v in buckets.values())/len(buckets)


@dataclass(frozen=True)
class Artifact:
    name: str
    kind: str
    run_id: str | None
    parents: tuple[str, ...] = ()
    pure_verified: bool = False


def validate_lineage(artifacts: Mapping[str,Artifact], sink: str, run_id: str) -> None:
    seen, visiting = set(), set()
    def walk(name: str):
        if name in visiting: raise ValueError('cycle')
        if name in seen: return
        if name not in artifacts: raise ValueError('missing parent')
        item=artifacts[name]
        if item.kind in {'dataset','public_backbone'}:
            if item.parents: raise ValueError('root cannot depend on trained model')
        elif item.kind=='pure_backbone_cache':
            if not item.pure_verified or not item.parents: raise ValueError('unverified pure cache')
            if any(artifacts.get(p, Artifact('', '', None)).kind not in {'dataset','public_backbone'} for p in item.parents):
                raise ValueError('learned output is not a pure cache')
        elif item.kind in {'task_checkpoint','optimizer','training_list','PCA','teacher_logits','index','result'}:
            if item.run_id != run_id: raise ValueError('external task artifact forbidden')
        else:
            raise ValueError('unrecognized artifact kind')
        visiting.add(name)
        for p in item.parents: walk(p)
        visiting.remove(name); seen.add(name)
    walk(sink)


def repeat_gate(m: Mapping[str, float]) -> tuple[bool, list[str]]:
    rules = {
        'overall_gain': m['main_R10']-m['raw_same_T_R10'] >= 0.005-1e-12,
        'implicit_guard': m['main_implicit_R10'] >= m['raw_same_T_implicit_R10']-1e-12,
        'path_guard': m['main_R10'] >= m['same_pool_QT_R10']-0.005-1e-12,
        'trained_qt_guard': m['main_R10'] >= m['trained_QT_same_pool_R10']-0.005-1e-12,
        'evidence_content': m['main_implicit_R10']-m['swap_implicit_R10'] >= 0.005-1e-12,
        'q_increment': m['QE_ET_R10']-m['EONLY_ET_R10'] >= 0.005-1e-12,
        'kd_guard': m['main_R10'] >= m['SUP_R10']-0.005-1e-12,
        'strict_population': m['strict_pairs']>=20,
        'strict_retention': m['QE_strict_retained']>=m['EONLY_strict_retained']
    }
    failed=[name for name, ok in rules.items() if not ok]
    return not failed,failed
