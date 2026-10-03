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
TEACHER_INFERENCE_CHUNK = 1024
# How a Q-E-T path is scored. ``triplet``: one relation-transformer pass over [q; e; t] plus the
# 11h global MLP (the score of the path replaces f0 in the bag aggregation). ``pairwise_residual``:
# the path score is f0(q, t) + path_head(q, e) + path_head(e, t), two local relations (方案.md's
# J(a -> b)) through the shared relation transformer and a zero-initialised head, so every path
# equals f0 at initialisation and the residual can only be moved by evidence-dependent signals.
PATH_MODES = ("triplet", "pairwise_residual")

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


class SegmentCache:
    """Encoded, role-tagged objects for relation scoring.

    Every ``encode_many`` result is one *group*: its tagged segments ``(N, L, W)`` flattened to
    ``(N * L, W)``, its lengths and its global vectors. ``pool`` concatenates the groups once
    (plus the relation/pair, separator and zero rows) so a scoring call packs its sequences with
    a single gather instead of thousands of per-object views, whose backward would allocate and
    accumulate a full-size zero tensor each. ``wrap`` accepts the plain
    ``{ref: (tagged tokens, global)}`` mapping used by tests and probes.
    """

    def __init__(self) -> None:
        self.tokens: list[Tensor] = []
        self.globs: list[Tensor] = []
        self.entry: dict[ObjectRef, tuple[int, int, int, int]] = {}  # ref -> (group, token start, length, global row)
        self._pool: Optional[tuple[Tensor, np.ndarray, int, Tensor, np.ndarray]] = None

    @classmethod
    def wrap(cls, cache) -> "SegmentCache":
        if isinstance(cache, cls):
            return cache
        wrapped = cls()
        for ref, (tokens, g) in cache.items():
            wrapped.add(tokens.unsqueeze(0), [tokens.shape[0]], g.unsqueeze(0), [ref])
        return wrapped

    def __contains__(self, ref: ObjectRef) -> bool:
        return ref in self.entry

    def __iter__(self):
        return iter(self.entry)

    def add(self, tagged: Tensor, lengths: Sequence[int], g: Tensor, refs: Sequence[ObjectRef]) -> None:
        group, (n, length, width) = len(self.tokens), tagged.shape
        self.tokens.append(tagged.reshape(n * length, width))
        self.globs.append(g)
        self.entry.update({ref: (group, k * length, int(lengths[k]), k) for k, ref in enumerate(refs)})
        self._pool = None

    def pool(self, model: "FreshPathTeacher") -> tuple[Tensor, np.ndarray, int]:
        """``(token pool, group token offsets, specials offset)``: rows ``specials + 0..8`` are
        ``rel + pair_kind``, ``specials + 9`` the separator and ``specials + 10`` a zero row."""
        if self._pool is None:
            offsets = np.cumsum([0, *(t.shape[0] for t in self.tokens)])
            specials = torch.cat([model.rel + model.pair_kind.weight, model.sep[None], model.sep.new_zeros(1, model.width)])
            pool = torch.cat([*self.tokens, specials])
            glob_offsets = np.cumsum([0, *(g.shape[0] for g in self.globs)])
            self._pool = (pool, offsets[:-1], int(offsets[-1]), torch.cat(self.globs), glob_offsets[:-1])
        return self._pool[:3]

    def locate(self, refs: Sequence[ObjectRef], offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(token start in the pool, length)`` of each ref."""
        entries = [self.entry[ref] for ref in refs]
        start = np.fromiter((offsets[g] + s for g, s, _, _ in entries), dtype=np.int64, count=len(entries))
        length = np.fromiter((n for _, _, n, _ in entries), dtype=np.int64, count=len(entries))
        return start, length

    def globals(self, refs: Sequence[ObjectRef]) -> Tensor:
        """Global vectors ``(N, W)`` of ``refs`` (one gather from the concatenated globals)."""
        if self._pool is None:
            raise RuntimeError("SegmentCache.pool must be built before globals are gathered")
        _pool, _offsets, _specials, globs, glob_offsets = self._pool
        index = [glob_offsets[g] + k for g, _, _, k in (self.entry[ref] for ref in refs)]
        return globs[torch.tensor(index, dtype=torch.long, device=globs.device)]


class FreshPathTeacher(nn.Module):
    """Shared-head Teacher: relation transformer over [rel; query; sep; (evidence; sep;) target]
    plus an 11h global-feature MLP, summed into one scoring head. ``path_mode`` (see
    ``PATH_MODES``) selects how evidence paths are scored."""

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
        path_mode: str = "triplet",
    ) -> None:
        super().__init__()
        if path_mode not in PATH_MODES:
            raise ValueError(f"unknown path_mode {path_mode!r}")
        self.width = width
        self.dropout = dropout
        self.path_mode = path_mode
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
        if path_mode == "pairwise_residual":
            self.path_head = nn.Sequential(
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
        if self.path_mode == "pairwise_residual":  # every path starts exactly at f0
            nn.init.zeros_(self.path_head[-1].weight)
            nn.init.zeros_(self.path_head[-1].bias)

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

    def score_pairs(self, cache: "SegmentCache | ObjectCache", refs: Sequence[tuple[ObjectRef, ObjectRef]]) -> Tensor:
        """Scores ``(N,)`` of (a, b) pairs whose tagged segments are in ``cache``."""
        return self._pair_scores(cache, refs, self.scoring_head)

    def score_path_pairs(self, cache: "SegmentCache | ObjectCache", refs: Sequence[tuple[ObjectRef, ObjectRef]]) -> Tensor:
        """``pairwise_residual`` local relations ``(N,)`` of (query, evidence) or (evidence, target)
        pairs: the same relation transformer and global MLP as ``score_pairs`` read by the
        zero-initialised ``path_head``."""
        if self.path_mode != "pairwise_residual":
            raise RuntimeError("score_path_pairs needs path_mode='pairwise_residual'")
        return self._pair_scores(cache, refs, self.path_head)

    def _pair_scores(self, cache, refs: Sequence[tuple[ObjectRef, ObjectRef]], head: nn.Module) -> Tensor:
        if not refs:
            return torch.empty(0, device=self.rel.device)
        cache = SegmentCache.wrap(cache)
        cache.pool(self)
        pair = torch.tensor([3 * KINDS.index(a[0]) + KINDS.index(b[0]) for a, b in refs], device=self.rel.device)
        a_glob, b_glob = cache.globals([a for a, _ in refs]), cache.globals([b for _, b in refs])
        globs = global_features(a_glob, b_glob, self.pair_kind.weight[pair], evidence=None, evidence_type_embedding=None)
        return self._score(cache, [[a for a, _ in refs], [b for _, b in refs]], pair, globs, head)

    def score_triplets(self, cache: "SegmentCache | ObjectCache", refs: Sequence[tuple[ObjectRef, ObjectRef, ObjectRef]]) -> Tensor:
        """Scores ``(N,)`` of (query table, evidence, target table) triplets from ``cache``."""
        if not refs:
            return torch.empty(0, device=self.rel.device)
        if any(q[0] != "table" or t[0] != "table" or e[0] == "table" for q, e, t in refs):
            raise ValueError("QET is only defined for table-evidence-table")
        cache = SegmentCache.wrap(cache)
        cache.pool(self)
        pair = torch.full((len(refs),), 3 * KINDS.index("table") + KINDS.index("table"), dtype=torch.long, device=self.rel.device)
        etype = self.modality.weight[torch.tensor([KINDS.index(e[0]) for _, e, _ in refs], device=self.rel.device)]
        gq, ge, gt = (cache.globals([ref[slot] for ref in refs]) for slot in range(3))
        globs = global_features(gq, gt, self.pair_kind.weight[pair], evidence=ge, evidence_type_embedding=etype)
        return self._score(cache, [[r[0] for r in refs], [r[1] for r in refs], [r[2] for r in refs]], pair, globs, head=self.scoring_head)

    def _score(self, cache: "SegmentCache", slots: list[list[ObjectRef]], pair: Tensor, globs: Tensor, head: nn.Module) -> Tensor:
        """Sequences ``[rel + pair_kind; slot_0; sep; slot_1 (; sep; slot_2)]``, tightly packed into one
        padded batch by a single gather from the cache's token pool, through the relation
        transformer; its first token plus the global-feature path is read by ``head``."""
        pool, offsets, specials = cache.pool(self)
        batch = len(slots[0])
        parts = [(specials + pair.cpu().numpy(), np.ones(batch, dtype=np.int64))]
        for i, refs in enumerate(slots):
            if i:
                parts.append((np.full(batch, specials + 9, dtype=np.int64), np.ones(batch, dtype=np.int64)))
            parts.append(cache.locate(refs, offsets))
        lengths = sum(length for _, length in parts)
        total = int(lengths.max())
        src = np.full(batch * total, specials + 10, dtype=np.int64)  # the zero row
        col = np.zeros(batch, dtype=np.int64)
        row_base = np.arange(batch, dtype=np.int64) * total
        for start, length in parts:
            n = int(length.sum())
            rows = np.repeat(np.arange(batch), length)
            within = np.arange(n) - np.repeat(np.cumsum(length) - length, length)
            src[row_base[rows] + col[rows] + within] = np.repeat(start, length) + within
            col += length
        index = torch.from_numpy(src).to(pool.device, non_blocking=True)
        x = pool[index].view(batch, total, self.width)
        pad = torch.from_numpy(np.arange(total)[None, :] >= lengths[:, None]).to(pool.device, non_blocking=True)
        out = self.relation(x, src_key_padding_mask=pad)
        return head(out[:, 0] + self.global_relation(globs)).squeeze(-1)

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
        cache = SegmentCache()
        xq, gq = self.encode_one("table", zq, cq)
        cache.add(self.tag("table", 0, xq, gq).unsqueeze(0), [xq.shape[0]], gq.unsqueeze(0), [("table", "q")])
        if ct_list:
            x_t, lengths, g_t = self.encode_many("table", zt, ct_list)
            cache.add(self.tag("table", 1, x_t, g_t), lengths, g_t, [("table", k) for k in range(len(ct_list))])
        evidence_ids = list(dict.fromkeys(e for _, e in paths))
        for kind in ("text", "image"):
            ids = [e for e in evidence_ids if evidence[e][0] == kind]
            if ids:
                seg, lengths, g = self.encode_many(kind, torch.stack([evidence[e][1] for e in ids]), [evidence[e][2] for e in ids])
                cache.add(self.tag(kind, 2, seg, g), [n + 1 for n in lengths], g, [(kind, e) for e in ids])
        pair_refs = [(("table", "q"), ("table", k)) for k in range(len(ct_list))]
        f0 = self._chunked(self.score_pairs, cache, pair_refs, chunk)
        if self.path_mode == "triplet":
            trip_refs = [(("table", "q"), (evidence[e][0], e), ("table", t)) for t, e in paths]
            return f0, self._chunked(self.score_triplets, cache, trip_refs, chunk)
        qe = self._chunked(self.score_path_pairs, cache, [(("table", "q"), (evidence[e][0], e)) for e in evidence_ids], chunk)
        et = self._chunked(self.score_path_pairs, cache, [((evidence[e][0], e), ("table", t)) for t, e in paths], chunk)
        position = {e: i for i, e in enumerate(evidence_ids)}
        index = torch.tensor([[t, position[e]] for t, e in paths], dtype=torch.long, device=zq.device).reshape(-1, 2)
        return f0, f0[index[:, 0]] + qe[index[:, 1]] + et

    def _chunked(self, score, cache: ObjectCache, refs: Sequence[tuple], chunk: int) -> Tensor:
        if not refs:
            return torch.empty(0, device=self.rel.device)
        return torch.cat([score(cache, refs[i : i + chunk]) for i in range(0, len(refs), chunk)])

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
