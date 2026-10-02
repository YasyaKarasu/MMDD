"""Teacher and Student architectures for Stage-1 CQET."""
from __future__ import annotations

import gc
import hashlib
import traceback
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .losses import global_features

KINDS = ("table", "text", "image")
STUDENT_RELATIONS = ("QT", "Q_text", "Q_image", "text_T", "image_T")
# (left kind, right kind) of every Student relation; the query/target side is always a table.
RELATION_KINDS = {
    "QT": ("table", "table"), "Q_text": ("table", "text"), "Q_image": ("table", "image"),
    "text_T": ("text", "table"), "image_T": ("image", "table"),
}
PATH_FROZEN_PREFIXES = ("adapters", "poolers", "globals", "modality", "table_kind")
TEACHER_INFERENCE_CHUNK = 256

# An encoded object is addressed by ``(kind, key)``; a scoring cache maps that to its
# role-tagged relation segment and global vector. Roles: 0 = query, 1 = target, 2 = evidence.
ObjectRef = tuple[str, Any]
ObjectCache = Mapping[ObjectRef, tuple[Tensor, Tensor]]


def state_sha(state: Mapping[str, Tensor]) -> str:
    """SHA256 over a state dict's names, dtypes, shapes and bytes (the model identity in receipts)."""
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode("utf-8"))
        h.update(str(array.dtype).encode("ascii"))
        h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        h.update(array.tobytes())
    return h.hexdigest()


def model_state_sha(model: nn.Module) -> str:
    return state_sha(model.state_dict())


class QueryPool(nn.Module):
    """Multi-query cross-attention pooling for variable-length token sequences."""

    def __init__(self, width: int, heads: int, slots: int):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(slots, width))
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        q = self.queries.unsqueeze(0).expand(x.size(0), -1, -1)
        pooled, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return self.norm(q + pooled)


class FreshPathTeacher(nn.Module):
    """Shared-head Teacher: relation transformer over [rel; query; sep; (evidence; sep;) target]
    plus an 11h global-feature MLP, summed into one scoring head."""

    def __init__(
        self,
        input_dim: int = 4096,
        width: int = 512,
        heads: int = 8,
        layers: int = 3,
        ffn: int = 2048,
        text_slots: int = 16,
        image_slots: int = 24,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.width = width
        self.dropout = dropout
        self.adapters = nn.ModuleDict({k: nn.Linear(input_dim, width) for k in KINDS})
        self.poolers = nn.ModuleDict(
            {
                "text": QueryPool(width, heads, text_slots),
                "image": QueryPool(width, heads, image_slots),
            }
        )
        self.globals = nn.ModuleDict(
            {k: nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width)) for k in KINDS}
        )
        self.roles = nn.Embedding(3, width)
        self.modality = nn.Embedding(3, width)
        self.table_kind = nn.Embedding(2, width)
        self.pair_kind = nn.Embedding(9, width)
        self.rel = nn.Parameter(torch.empty(width))
        self.sep = nn.Parameter(torch.empty(width))

        self.relation = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                width, heads, ffn, dropout, activation="gelu", batch_first=True, norm_first=True
            ),
            layers,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        self.global_relation = nn.Sequential(
            nn.Linear(11 * width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.scoring_head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )
        self.reset_fresh()

    def reset_fresh(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        for x in (self.rel, self.sep, self.poolers["text"].queries, self.poolers["image"].queries):
            nn.init.normal_(x, std=0.02)

    # ------------------------------------------------------------------ encoding --

    def encode_one(self, kind: str, z: Tensor, content: Tensor) -> tuple[Tensor, Tensor]:
        """Untagged token segment ``(L, W)`` and global vector ``(W,)`` of one object."""
        if kind not in KINDS:
            raise ValueError(f"unknown object kind {kind!r}")
        adapter = self.adapters[kind]
        x = adapter(content.to(adapter.weight.dtype))
        if kind == "table":
            kinds = torch.ones(len(x), dtype=torch.long, device=x.device)
            kinds[0] = 0
            x = x + self.table_kind(kinds)
        else:
            x = self.poolers[kind](x.unsqueeze(0))[0]
        return x, self.globals[kind](z.to(adapter.weight.dtype))

    def encode_many(self, kind: str, z: Tensor, tokens: Sequence[Tensor]) -> tuple[Tensor, list[int], Tensor]:
        """Batched ``encode_one``: segments ``(N, Lmax, W)`` (tables padded to the longest, text/image
        pooled to their slot count), per-object segment lengths, and globals ``(N, W)``."""
        if kind not in KINDS:
            raise ValueError(f"unknown object kind {kind!r}")
        adapter = self.adapters[kind]
        dtype = adapter.weight.dtype
        if not tokens:
            return torch.empty(0, 0, self.width, device=z.device), [], torch.empty(0, self.width, device=z.device)
        lengths = [t.shape[0] for t in tokens]
        x = adapter(pad_sequence(list(tokens), batch_first=True).to(dtype))
        g = self.globals[kind](z.to(dtype))
        if kind == "table":
            kinds = torch.ones(x.shape[1], dtype=torch.long, device=x.device)
            kinds[0] = 0
            return x + self.table_kind(kinds), lengths, g
        padding = None
        if len(set(lengths)) > 1:
            padding = torch.arange(x.shape[1], device=x.device)[None, :] >= torch.tensor(lengths, device=x.device)[:, None]
        pooled = self.poolers[kind](x, key_padding_mask=padding)
        return pooled, [pooled.shape[1]] * len(tokens), g

    def tag(self, kind: str, role: int, tokens: Tensor, g: Tensor) -> Tensor:
        """Role-tagged relation segment(s); evidence (role 2) segments start with the global vector."""
        if role == 2:
            tokens = torch.cat([g.unsqueeze(-2), tokens], dim=-2)
        return tokens + (self.modality.weight[KINDS.index(kind)] + self.roles.weight[role])

    # ------------------------------------------------------------------- scoring --

    def score_pairs(self, cache: ObjectCache, refs: Sequence[tuple[ObjectRef, ObjectRef]]) -> Tensor:
        """Scores ``(N,)`` of (a, b) pairs whose tagged segments are in ``cache``."""
        if not refs:
            return torch.empty(0, device=self.rel.device)
        a_entries = [cache[a] for a, _ in refs]
        b_entries = [cache[b] for _, b in refs]
        pair = self.pair_kind.weight[torch.tensor(
            [3 * KINDS.index(a[0]) + KINDS.index(b[0]) for a, b in refs], device=self.rel.device
        )]
        relation = (self.rel + pair).unsqueeze(1)
        sep = self.sep.unsqueeze(0)
        seqs = [[relation[i], sa, sep, sb] for i, ((sa, _), (sb, _)) in enumerate(zip(a_entries, b_entries))]
        ga = torch.stack([g for _, g in a_entries])
        gb = torch.stack([g for _, g in b_entries])
        return self._score(seqs, global_features(ga, gb, pair, evidence=None, evidence_type_embedding=None))

    def score_triplets(self, cache: ObjectCache, refs: Sequence[tuple[ObjectRef, ObjectRef, ObjectRef]]) -> Tensor:
        """Scores ``(N,)`` of (query table, evidence, target table) triplets from ``cache``."""
        if not refs:
            return torch.empty(0, device=self.rel.device)
        if any(q[0] != "table" or t[0] != "table" or e[0] == "table" for q, e, t in refs):
            raise ValueError("QET is only defined for table-evidence-table")
        q_entries = [cache[q] for q, _, _ in refs]
        e_entries = [cache[e] for _, e, _ in refs]
        t_entries = [cache[t] for _, _, t in refs]
        pair = self.pair_kind.weight[3 * KINDS.index("table") + KINDS.index("table")].expand(len(refs), -1)
        etype = self.modality.weight[torch.tensor([KINDS.index(e[0]) for _, e, _ in refs], device=self.rel.device)]
        relation = (self.rel + pair).unsqueeze(1)
        sep = self.sep.unsqueeze(0)
        seqs = [
            [relation[i], sq, sep, se, sep, st]
            for i, ((sq, _), (se, _), (st, _)) in enumerate(zip(q_entries, e_entries, t_entries))
        ]
        gq = torch.stack([g for _, g in q_entries])
        ge = torch.stack([g for _, g in e_entries])
        gt = torch.stack([g for _, g in t_entries])
        return self._score(seqs, global_features(gq, gt, pair, evidence=ge, evidence_type_embedding=etype))

    def _score(self, seqs: list[list[Tensor]], globs: Tensor) -> Tensor:
        """Pack variable-length segment lists into one padded batch, run the relation transformer
        and combine its first token with the global-feature path."""
        lengths = [sum(len(s) for s in segs) for segs in seqs]
        batch, total = len(seqs), max(lengths)
        device = globs.device
        flat = torch.cat([s for segs in seqs for s in segs], dim=0)
        tokens = flat.shape[0]
        length_t = torch.tensor(lengths, device=device, dtype=torch.long)
        rows = torch.repeat_interleave(torch.arange(batch, device=device), length_t, output_size=tokens)
        starts = torch.cumsum(length_t, 0) - length_t
        cols = torch.arange(tokens, device=device) - torch.repeat_interleave(starts, length_t, output_size=tokens)
        dst = rows * total + cols
        x = flat.new_zeros(batch * total, self.width)
        x[dst] = flat
        pad = torch.ones(batch * total, dtype=torch.bool, device=device)
        pad[dst] = False
        out = self.relation(x.view(batch, total, self.width), src_key_padding_mask=pad.view(batch, total))
        return self.scoring_head(out[:, 0] + self.global_relation(globs)).squeeze(-1)

    def score_query_lists(
        self,
        q: tuple[Tensor, Tensor],
        targets: tuple[Tensor, Sequence[Tensor]],
        evidence: Mapping[str, tuple[str, Tensor, Tensor]],
        paths: Sequence[tuple[int, str]],
        chunk: int = TEACHER_INFERENCE_CHUNK,
    ) -> tuple[Tensor, Tensor]:
        """Inference scoring of one query: ``f0`` over ``targets`` and one logit per ``(target index,
        evidence id)`` path. Under ``no_grad`` an OOM halves ``chunk`` and retries the whole query;
        training callers keep ordinary autograd semantics without implicit retries."""
        if chunk < 1:
            raise ValueError("inference chunk must be positive")
        if self.training or torch.is_grad_enabled():
            return self._score_query_lists_once(q, targets, evidence, paths, chunk)
        while True:
            try:
                return self._score_query_lists_once(q, targets, evidence, paths, chunk)
            except BaseException as error:
                is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
                    isinstance(error, RuntimeError) and "out of memory" in str(error).lower())
                if not is_oom:
                    raise
                if chunk == 1:
                    raise RuntimeError("BLOCKED_RESOURCE: inference chunk 1 OOM") from error
                traceback.clear_frames(error.__traceback__)
                error.__traceback__ = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[Teacher inference OOM] full-query retry chunk={chunk}->{chunk // 2}", flush=True)
            chunk //= 2

    def _score_query_lists_once(self, q, targets, evidence, paths, chunk: int) -> tuple[Tensor, Tensor]:
        zq, cq = q
        zt, ct_list = targets
        xq, gq = self.encode_one("table", zq, cq)
        cache: dict[ObjectRef, tuple[Tensor, Tensor]] = {("table", "q"): (self.tag("table", 0, xq, gq), gq)}
        if ct_list:
            x_t, lengths, g_t = self.encode_many("table", zt, ct_list)
            tagged = self.tag("table", 1, x_t, g_t)
            cache.update({("table", k): (tagged[k, :length], g_t[k]) for k, length in enumerate(lengths)})
        evidence_ids = list(dict.fromkeys(e for _, e in paths))
        for kind in ("text", "image"):
            ids = [e for e in evidence_ids if evidence[e][0] == kind]
            if ids:
                seg, _, g = self.encode_many(kind, torch.stack([evidence[e][1] for e in ids]), [evidence[e][2] for e in ids])
                tagged = self.tag(kind, 2, seg, g)
                cache.update({(kind, e): (tagged[i], g[i]) for i, e in enumerate(ids)})
        pair_refs = [(("table", "q"), ("table", k)) for k in range(len(ct_list))]
        trip_refs = [(("table", "q"), (evidence[e][0], e), ("table", t)) for t, e in paths]
        f0 = torch.cat([self.score_pairs(cache, pair_refs[i : i + chunk]) for i in range(0, len(pair_refs), chunk)]
                       or [torch.empty(0, device=zq.device)])
        path_scores = torch.cat([self.score_triplets(cache, trip_refs[i : i + chunk]) for i in range(0, len(trip_refs), chunk)]
                                or [torch.empty(0, device=zq.device)])
        return f0, path_scores

    def set_tb_trainable(self) -> list[nn.Parameter]:
        """Freeze object representations and poolers for T_B stages (SPEC 13.2)."""
        params = []
        for name, param in self.named_parameters():
            frozen = name.startswith(PATH_FROZEN_PREFIXES)
            param.requires_grad_(not frozen)
            if not frozen:
                params.append(param)
        return params


class NativeStudent(nn.Module):
    """Native linear projection + bilinear relation Student (no adapter, SPEC 14)."""

    def __init__(
        self,
        pca_basis: Tensor,  # (1024, 4096)
        pca_mean: Tensor,   # (4096,)
        dim: int = 1024,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("pca_mean", pca_mean.clone().detach().float())
        self.register_buffer("pca_basis", pca_basis.clone().detach().float())

        self.P = nn.ParameterDict(
            {kind: nn.Parameter(pca_basis.clone().detach().float()) for kind in KINDS}
        )
        self.R = nn.ParameterDict(
            {rel: nn.Parameter(torch.eye(dim, dtype=torch.float32)) for rel in STUDENT_RELATIONS}
        )

    def u(self, kind: str, z: Tensor) -> Tensor:
        return (z - self.pca_mean) @ self.P[kind].T

    def ann_query(self, relation: str, z_left: Tensor) -> Tensor:
        """Transformed left query whose dot with ``index_vectors`` equals the bilinear score."""
        return self.u(RELATION_KINDS[relation][0], z_left) @ self.R[relation]

    def index_vectors(self, relation: str, z_right: Tensor) -> Tensor:
        return self.u(RELATION_KINDS[relation][1], z_right)

    def score(self, a_kind: str, z_a: Tensor, b_kind: str, z_b: Tensor) -> Tensor:
        relation = next(r for r, kinds in RELATION_KINDS.items() if kinds == (a_kind, b_kind))
        return (self.u(a_kind, z_a) @ self.R[relation] * self.u(b_kind, z_b)).sum(dim=-1)

    def anchor_loss(self) -> Tensor:
        p_loss = torch.stack([F.mse_loss(self.P[k], self.pca_basis) for k in KINDS]).mean()
        eye = torch.eye(self.dim, device=self.pca_basis.device, dtype=torch.float32)
        r_loss = torch.stack([F.mse_loss(self.R[r], eye) for r in STUDENT_RELATIONS]).mean()
        return p_loss + r_loss

    def param_groups(self, p_lr: float, r_lr: float) -> list[dict]:
        return [
            {"params": list(self.P.values()), "lr": p_lr},
            {"params": list(self.R.values()), "lr": r_lr},
        ]


class QTStudent(nn.Module):
    """QT-only linear projection + bilinear relation Student (SPEC 14.1)."""

    def __init__(
        self,
        pca_basis: Tensor,
        pca_mean: Tensor,
        dim: int = 1024,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("pca_mean", pca_mean.clone().detach().float())
        self.register_buffer("pca_basis", pca_basis.clone().detach().float())

        self.P_table = nn.Parameter(pca_basis.clone().detach().float())
        self.R_QT = nn.Parameter(torch.eye(dim, dtype=torch.float32))

    def u(self, z: Tensor) -> Tensor:
        return (z - self.pca_mean) @ self.P_table.T

    def ann_query(self, relation: str, z_left: Tensor) -> Tensor:
        if relation != "QT":
            raise ValueError(f"QT Student has no {relation} relation")
        return self.u(z_left) @ self.R_QT

    def index_vectors(self, relation: str, z_right: Tensor) -> Tensor:
        if relation != "QT":
            raise ValueError(f"QT Student has no {relation} relation")
        return self.u(z_right)

    def score(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        return (self.u(z_a) @ self.R_QT * self.u(z_b)).sum(dim=-1)

    def anchor_loss(self) -> Tensor:
        eye = torch.eye(self.dim, device=self.pca_basis.device, dtype=torch.float32)
        return F.mse_loss(self.P_table, self.pca_basis) + F.mse_loss(self.R_QT, eye)

    def param_groups(self, p_lr: float, r_lr: float) -> list[dict]:
        return [
            {"params": [self.P_table], "lr": p_lr},
            {"params": [self.R_QT], "lr": r_lr},
        ]
