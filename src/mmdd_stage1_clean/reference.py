"""CLEAN-R1 mathematical reference.

This module is a byte-faithful copy of the reference that shipped with the
experiment package (``mmdd_s1_clean_r1/reference_core.py``).  It is kept as a
self-contained module so that production code never has to import anything
from the historical repository layout, and so the exact semantics used by the
run can be pinned by hash.

Provenance
----------
spec_package_reference_sha256 : ef40ebb0816a79100b214ebedb0a881b967a3bb2bf21f0c50ceb5dc5a0d18aff
spec_package_test_sha256      : 849875ab4e0ffb1fafb08705bc9a58f3a43d23bf2333fd693526b56ec5805433

The equations implemented here are those of ``EXPERIMENT_SPEC.zh-CN.md``:
Eq. (1) Teacher scoring, Eq. (6) support labels, Eq. (10)-(15) Student,
Eq. (17) ranking loss, Eq. (21) KD, and the round-robin budget schedule.
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

EPS = 1e-6

SPEC_REFERENCE_SHA256 = (
    "ef40ebb0816a79100b214ebedb0a881b967a3bb2bf21f0c50ceb5dc5a0d18aff"
)
SPEC_TEST_SHA256 = (
    "849875ab4e0ffb1fafb08705bc9a58f3a43d23bf2333fd693526b56ec5805433"
)


def stable_order(values: Iterable[str], namespace: str) -> list[str]:
    return sorted(
        set(values),
        key=lambda s: (hashlib.sha256((namespace + "\0" + s).encode()).hexdigest(), s),
    )


def unit(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if not torch.isfinite(x).all() or torch.any(x.norm(dim=-1) < EPS):
        raise ValueError("Non-finite or degenerate vector before unit normalization")
    return F.normalize(x, p=2, dim=-1, eps=EPS)


def rms(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x / torch.sqrt(x.square().mean(-1, keepdim=True) + EPS)


def rank_loss(
    scores: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
) -> torch.Tensor:
    """One query/list.  Positives do not compete with other positives.

    Unknown qrel-positive candidates must have positive=False, negative=False.
    """
    if (
        scores.ndim != 1
        or scores.shape != positive.shape
        or scores.shape != negative.shape
    ):
        raise ValueError("Expected equal one-dimensional tensors")
    p, n = positive.bool(), negative.bool()
    if (p & n).any():
        raise ValueError("Conflicting labels")
    if not p.any() or not n.any():
        return scores.sum() * 0
    delta = scores[n][None, :] - scores[p][:, None]
    return torch.logsumexp(
        torch.cat([torch.zeros_like(delta[:, :1]), delta], dim=1), dim=1
    ).mean()


def kd_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    valid: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    if (
        temperature <= 0
        or student.shape != teacher.shape
        or valid.shape != student.shape
        or student.ndim != 1
    ):
        raise ValueError("Invalid distillation arguments")
    if valid.sum() < 2:
        return student.sum() * 0
    s = student[valid].float() / temperature
    t = teacher[valid].detach().float() / temperature
    return temperature**2 * F.kl_div(
        F.log_softmax(s, 0), F.softmax(t, 0), reduction="sum"
    )


def support_label(
    target: str,
    context: set[str],
    direct: set[str],
    implicit: set[str],
    witnesses: dict[str, set[str]],
    explicit_negative: set[str],
) -> int | None:
    """Support of the supplied context, NOT potential reachability of the target."""
    if target in explicit_negative:
        if target in direct or target in implicit:
            raise ValueError("Conflicting qrel")
        return 0
    if target in direct:
        return 1
    if target in implicit:
        if witnesses.get(target, set()) & context:
            return 1
        if not context:
            return 0
        return None  # missing positive witness annotation is not a negative
    return None


def round_robin(
    streams: Sequence[Sequence[str]], limit: int | None = None
) -> list[str]:
    """One NEW target per active stream per turn; skip already emitted IDs."""
    positions = [0] * len(streams)
    seen: set[str] = set()
    out: list[str] = []
    while True:
        progress = False
        for j, stream in enumerate(streams):
            while positions[j] < len(stream) and stream[positions[j]] in seen:
                positions[j] += 1
            if positions[j] == len(stream):
                continue
            target = stream[positions[j]]
            positions[j] += 1
            seen.add(target)
            out.append(target)
            progress = True
            if limit is not None and len(out) >= limit:
                return out
        if not progress:
            return out


class CompactStudent(nn.Module):
    """A single Student: shared object pooling plus directed/conditional queries."""

    def __init__(
        self, input_dim: int = 4096, d: int = 1024, rank: int = 64, kinds: int = 11
    ):
        super().__init__()
        self.input_dim, self.d = input_dim, d
        self.project = nn.Linear(input_dim, d, bias=False)
        self.modality = nn.Embedding(3, d)
        self.kind = nn.Embedding(kinds, d)
        self.norm = nn.LayerNorm(d, eps=EPS)
        self.pool_query = nn.Parameter(torch.zeros(d))
        self.direct = nn.Linear(d, d, bias=False)
        self.evidence = nn.Linear(d, d, bias=False)
        self.base_e = nn.Linear(d, d, bias=False)
        self.q_factor = nn.Linear(d, rank, bias=False)
        self.e_factor = nn.Linear(d, rank, bias=False)
        self.out_factor = nn.Linear(rank, d, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in (self.direct, self.evidence, self.base_e):
            nn.init.eye_(m.weight)
        nn.init.zeros_(self.pool_query)

    def encode(
        self,
        cache: torch.Tensor,
        valid: torch.Tensor,
        modality: torch.Tensor,
        kind: torch.Tensor,
    ) -> torch.Tensor:
        # cache [B,K,D0], valid/kind [B,K], modality [B]
        if cache.ndim != 3 or cache.shape[:2] != valid.shape or valid.shape != kind.shape:
            raise ValueError("Invalid object cache shapes")
        if not valid.any(-1).all():
            raise ValueError("Every object requires a valid global slot")
        h = self.norm(
            self.project(rms(cache))
            + self.modality(modality)[:, None]
            + self.kind(kind)
        )
        a = (h @ self.pool_query) / math.sqrt(self.d)
        a = F.softmax(a.masked_fill(~valid.bool(), -torch.inf), dim=-1)
        return unit((a[..., None] * h).sum(-2))

    def query_direct(self, q: torch.Tensor) -> torch.Tensor:
        return unit(self.direct(q))

    def query_evidence(self, q: torch.Tensor) -> torch.Tensor:
        return unit(self.evidence(q))

    def query_next(self, q: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        if q.shape != e.shape:
            raise ValueError("Q and E shapes must match")
        interaction = torch.tanh(self.q_factor(q)) * torch.tanh(self.e_factor(e))
        return unit(self.base_e(e) + self.out_factor(interaction))

    @staticmethod
    def logits(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        return (query @ keys.T) / 0.07


class UnifiedTeacher(nn.Module):
    """One shared 3-layer Transformer and one scalar readout.

    mode 0 = potential/retrieval (P); mode 1 = supplied-context support (J).
    Layer 1 is block-local to preserve object groups; layers 2 and 3 allow joint
    Q/E/T interaction.  There is no Fbase/Fbridge/gate.
    """

    def __init__(
        self,
        input_dim: int = 4096,
        d: int = 512,
        heads: int = 8,
        ffn: int = 1024,
        kinds: int = 11,
    ):
        super().__init__()
        self.project = nn.Linear(input_dim, d, bias=False)
        self.modality = nn.Embedding(3, d)
        self.role = nn.Embedding(3, d)  # Q, candidate, E context
        self.kind = nn.Embedding(kinds, d)
        self.task = nn.Embedding(2, d)
        self.rel = nn.Parameter(torch.empty(d))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d,
                    heads,
                    dim_feedforward=ffn,
                    dropout=0.1,
                    activation="gelu",
                    layer_norm_eps=EPS,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(3)
            ]
        )
        self.final_norm = nn.LayerNorm(d, eps=EPS)
        self.readout = nn.Linear(d, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, p in self.named_parameters():
            if p.ndim == 2:
                if any(x in name for x in ("modality.", "role.", "kind.", "task.")):
                    nn.init.normal_(p, std=0.02)
                else:
                    nn.init.xavier_uniform_(p)
            elif name == "rel":
                nn.init.normal_(p, std=0.02)
            elif name.endswith("weight"):
                nn.init.ones_(p)
            else:
                nn.init.zeros_(p)

    def forward(
        self,
        cache: torch.Tensor,
        valid: torch.Tensor,
        modality: torch.Tensor,
        role: torch.Tensor,
        kind: torch.Tensor,
        mode: torch.Tensor,
    ) -> torch.Tensor:
        # cache [B,O,K,D0]; validity [B,O,K]; modality/role [B,O]; kind [B,O,K]
        if cache.ndim != 4 or cache.shape[:3] != valid.shape or kind.shape != valid.shape:
            raise ValueError("Invalid Teacher inputs")
        b, o, k, _ = cache.shape
        h = (
            self.project(rms(cache))
            + self.modality(modality)[:, :, None]
            + self.role(role)[:, :, None]
            + self.kind(kind)
        ).reshape(b, o * k, -1)
        rel = self.rel[None] + self.task(mode)
        h = torch.cat([rel[:, None], h], dim=1)
        padding = torch.cat(
            [
                torch.zeros(b, 1, dtype=torch.bool, device=cache.device),
                ~valid.bool().reshape(b, -1),
            ],
            1,
        )
        groups = torch.cat(
            [
                torch.tensor([-1], device=cache.device),
                torch.arange(o, device=cache.device).repeat_interleave(k),
            ]
        )
        blocked = groups[:, None] != groups[None, :]
        # Fully padded object groups can otherwise give all-masked attention rows.
        # Padding queries may attend CLS; real object tokens never may in layer 1.
        heads = self.layers[0].self_attn.num_heads
        masks = blocked[None].expand(b, -1, -1).clone()
        for i in range(b):
            masks[i, padding[i], 0] = False
        mask = masks[:, None].expand(b, heads, -1, -1).reshape(
            b * heads, 1 + o * k, 1 + o * k
        )
        h = self.layers[0](h, src_mask=mask, src_key_padding_mask=padding)
        for layer in self.layers[1:]:
            h = layer(h, src_key_padding_mask=padding)
        return self.readout(self.final_norm(h[:, 0])).squeeze(-1)


def feature_bytes(objects: int, dim: int = 4096, summary_slots: int = 8) -> int:
    if objects < 0:
        raise ValueError("objects must be nonnegative")
    return objects * dim * (4 + summary_slots * 2)
