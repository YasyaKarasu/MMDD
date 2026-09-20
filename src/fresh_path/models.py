"""Fresh Teacher and Student architectures (SPEC 6 and 8).

All task parameters here are constructed from scratch by this module.  Nothing
loads a checkpoint: the only permitted initialisers are the public PCA basis
(computed in this run) and the deterministic initialisation in SPEC 6.2/8.2.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .contracts import ConditionalAdapter

KINDS = ("table", "text", "image")
ROLES = ("source", "target", "bridge")


class QueryPool(nn.Module):
    """Learned query pooler for text/image content tokens (attention dropout 0)."""

    def __init__(self, width: int, heads: int, slots: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(slots, width))
        self.attn = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, L, width) -> (batch, slots, width)
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        pooled, _ = self.attn(q, x, x, need_weights=False)
        return self.norm(q + pooled)


class FreshPathTeacher(nn.Module):
    """Single fresh Teacher with one shared head for pair and QET scoring."""

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
            {"text": QueryPool(width, heads, text_slots), "image": QueryPool(width, heads, image_slots)}
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
        # Every layer is constructed independently; nothing is cloned after init.
        self.relation = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                width, heads, ffn, dropout, activation="gelu", batch_first=True, norm_first=True
            ),
            layers,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        self.global_relation = nn.Sequential(nn.Linear(5 * width, width), nn.GELU(), nn.Linear(width, width))
        self.scoring_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1)
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

    # ------------------------------------------------------------------ encode
    def encode_one(self, kind: str, z: Tensor, content: Tensor) -> tuple[Tensor, Tensor]:
        """Encode one object. z:(input_dim) content:(L,input_dim) -> tokens (L',width), g (width)."""
        if kind not in KINDS:
            raise ValueError(f"unknown object kind {kind!r}")
        x = self.adapters[kind](content)
        if kind == "table":
            kinds = torch.ones(len(x), dtype=torch.long, device=x.device)
            kinds[0] = 0
            x = x + self.table_kind(kinds)
        else:
            x = self.poolers[kind](x.unsqueeze(0))[0]
        g = self.globals[kind](z)
        return x, g

    def _encode_memo(self, cache, kind: str, z: Tensor, content: Tensor, object_key=None) -> tuple[Tensor, Tensor]:
        """Memoize ``encode_one`` for one query/optimizer step.

        ``encode_one`` contains no dropout (the pooler uses attention dropout 0
        and the adapters/globals are Linear/LayerNorm), so encoding a given
        object once instead of once per candidate sequence is bit-identical.
        Stage helpers clear the cache after each optimizer step.  Stable object
        IDs are used whenever available, so the same query/evidence/target
        encoding is shared across relation and path forwards without relying
        on recyclable ``data_ptr`` values.
        """
        if cache is None:
            return self.encode_one(kind, z, content)
        # Stable object keys are supplied by the stage helpers.  The pointer
        # fallback preserves the public low-level API for callers that do not
        # have object IDs, while query/step caches never rely on recyclable
        # ``data_ptr`` values.
        key = (kind, object_key) if object_key is not None else (
            kind, z.data_ptr(), content.data_ptr(), content.shape[0]
        )
        hit = cache.get(key)
        if hit is None:
            hit = self.encode_one(kind, z, content)
            cache[key] = hit
        return hit

    def _seg(self, kind: str, tokens: Tensor, g: Tensor, role: int, *, prepend_g: bool = False) -> Tensor:
        extra = self.modality.weight[KINDS.index(kind)] + self.roles.weight[role]
        if prepend_g:
            return torch.cat([g.unsqueeze(0), tokens], 0) + extra
        return tokens + extra

    def _pair_embedding(self, a_kind: str, b_kind: str) -> Tensor:
        return self.pair_kind.weight[3 * KINDS.index(a_kind) + KINDS.index(b_kind)]

    def score_pairs(self, pairs: Sequence[tuple[str, Tensor, Tensor, str, Tensor, Tensor]],
                    cache: dict | None = None, cache_keys: Sequence[tuple] | None = None) -> Tensor:
        """pairs: (a_kind, z_a, C_a, b_kind, z_b, C_b) -> scores (N,)."""
        seqs: list[list[Tensor]] = []
        globs: list[Tensor] = []
        if cache_keys is not None and len(cache_keys) != len(pairs):
            raise ValueError("score_pairs cache_keys must align with pairs")
        for i, (a_kind, z_a, c_a, b_kind, z_b, c_b) in enumerate(pairs):
            key = None if cache_keys is None else cache_keys[i]
            ca, ga = self._encode_memo(cache, a_kind, z_a, c_a,
                                       None if key is None else key[0])
            cb, gb = self._encode_memo(cache, b_kind, z_b, c_b,
                                       None if key is None else key[1])
            pair = self._pair_embedding(a_kind, b_kind)
            seqs.append(
                [
                    (self.rel + pair).unsqueeze(0),
                    self._seg(a_kind, ca, ga, 0),
                    self.sep.unsqueeze(0),
                    self._seg(b_kind, cb, gb, 1),
                ]
            )
            globs.append(torch.cat([ga, gb, ga * gb, (ga - gb).abs(), pair]))
        return self._score(seqs, globs)

    def score_triplets(
        self, triplets: Sequence[tuple[str, Tensor, Tensor, str, Tensor, Tensor, str, Tensor, Tensor]],
        cache: dict | None = None,
        cache_keys: Sequence[tuple] | None = None,
    ) -> Tensor:
        """triplets: (q_kind,z_q,C_q, e_kind,z_e,C_e, t_kind,z_t,C_t) -> scores (N,)."""
        seqs: list[list[Tensor]] = []
        globs: list[Tensor] = []
        if cache_keys is not None and len(cache_keys) != len(triplets):
            raise ValueError("score_triplets cache_keys must align with triplets")
        for i, (q_kind, z_q, c_q, e_kind, z_e, c_e, t_kind, z_t, c_t) in enumerate(triplets):
            if q_kind != "table" or t_kind != "table" or e_kind == "table":
                raise ValueError("QET is only defined for table-evidence-table")
            key = None if cache_keys is None else cache_keys[i]
            cq, gq = self._encode_memo(cache, q_kind, z_q, c_q,
                                       None if key is None else key[0])
            ce, ge = self._encode_memo(cache, e_kind, z_e, c_e,
                                       None if key is None else key[1])
            ct, gt = self._encode_memo(cache, t_kind, z_t, c_t,
                                       None if key is None else key[2])
            pair = self._pair_embedding("table", "table")
            seqs.append(
                [
                    (self.rel + pair).unsqueeze(0),
                    self._seg(q_kind, cq, gq, 0),
                    self.sep.unsqueeze(0),
                    self._seg(e_kind, ce, ge, 2, prepend_g=True),
                    self.sep.unsqueeze(0),
                    self._seg(t_kind, ct, gt, 1),
                ]
            )
            globs.append(torch.cat([gq, gt, gq * gt, (gq - gt).abs(), pair]))
        return self._score(seqs, globs)

    def _score(self, seqs: list[list[Tensor]], globs: list[Tensor]) -> Tensor:
        """Assemble one padded batch with two vectorised ops instead of per-segment copies.

        The resulting ``x`` and padding mask are element-for-element identical to
        the previous slice-assignment loop, so this is a pure speedup.
        """
        width = self.width
        n = len(seqs)
        lengths = [sum(len(s) for s in segs) for segs in seqs]
        # Preserve the caller's input order.  Length bucketing is mathematically
        # harmless in eval mode, but training dropout assigns random masks in
        # batch order, so reordering would change the optimisation trajectory.
        total = max(lengths)
        batch = n
        device = globs[0].device
        flat = torch.cat([s for segs in seqs for s in segs], dim=0)
        tokens = flat.shape[0]
        length_t = torch.tensor(lengths, device=device, dtype=torch.long)
        rows = torch.repeat_interleave(torch.arange(batch, device=device), length_t)
        starts = torch.cumsum(length_t, 0) - length_t
        cols = torch.arange(tokens, device=device) - torch.repeat_interleave(starts, length_t)
        dst = rows * total + cols
        x = flat.new_zeros(batch * total, width)
        x[dst] = flat
        x = x.view(batch, total, width)
        pad = torch.ones(batch * total, dtype=torch.bool, device=device)
        pad[dst] = False
        pad = pad.view(batch, total)
        out = self.relation(x, src_key_padding_mask=pad)
        local = out[:, 0]
        glob = self.global_relation(torch.stack(globs))
        return self.scoring_head(local + glob).squeeze(-1)


class PCAStudent(nn.Module):
    """Student with three projections, five directed relations and optional adapter.

    When ``adapter`` is None this is the old-style unconditioned structure used
    by KD-NATIVE (SPEC 9.6): there is no conditional module to instantiate.
    """

    RELATIONS = ("QT", "Q_to_text", "Q_to_image", "text_to_T", "image_to_T")
    _TYPE_RELATION = {"QT": "QT", "Q_to_text": "Q_to_text", "Q_to_image": "Q_to_image"}

    def __init__(
        self,
        basis: Tensor,
        *,
        adapter: bool = True,
        adapter_hidden: int = 256,
        adapter_last_zero: bool = True,
    ) -> None:
        super().__init__()
        dim = basis.shape[0]
        self.dim = dim
        self.projections = nn.ModuleDict(
            {k: nn.Linear(basis.shape[1], dim, bias=False) for k in KINDS}
        )
        for k in KINDS:
            with torch.no_grad():
                self.projections[k].weight.copy_(basis)
        self.relations = nn.ParameterDict({r: nn.Parameter(torch.eye(dim)) for r in self.RELATIONS})
        self.adapter = ConditionalAdapter(dim, adapter_hidden) if adapter else None

    # ------------------------------------------------------------- projections
    def u(self, kind: str, z: Tensor) -> Tensor:
        return z @ self.projections[kind].weight.T

    def v(self, source_kind: str, z_source: Tensor) -> Tensor:
        """u_source @ R_{source->T} (row-vector convention, SPEC 8.2)."""
        rel = f"{source_kind}_to_T"
        return self.u(source_kind, z_source) @ self.relations[rel]

    def qt_score(self, z_q: Tensor, z_t: Tensor) -> Tensor:
        return (self.u("table", z_q) @ self.relations["QT"]) @ self.u("table", z_t).T

    def first_hop(self, z_q: Tensor, evidence_kind: str, z_e: Tensor) -> Tensor:
        rel = f"Q_to_{evidence_kind}"
        return (self.u("table", z_q) @ self.relations[rel]) @ self.u(evidence_kind, z_e).T

    def second_hop(self, evidence_kind: str, z_e: Tensor, z_t: Tensor) -> Tensor:
        return self.v(evidence_kind, z_e) @ self.u("table", z_t).T

    def conditional_vector(self, z_q: Tensor, evidence_kind: str, z_e: Tensor, *, read_q: bool = True) -> Tensor:
        if self.adapter is None:
            raise RuntimeError("conditional_vector requires an adapter (KD-NATIVE has none)")
        u_q = self.u("table", z_q)
        u_e = self.u(evidence_kind, z_e)
        base = self.v(evidence_kind, z_e)
        return self.adapter(u_q, u_e, base, read_q=read_q)

    def native_second_hop(self, evidence_kind: str, z_e: Tensor, z_t: Tensor) -> Tensor:
        return self.second_hop(evidence_kind, z_e, z_t)


class QTOnlyStudent(nn.Module):
    """Independent Direct-only Student: only P_table and R_QT (SPEC 9.7)."""

    def __init__(self, basis: Tensor) -> None:
        super().__init__()
        dim = basis.shape[0]
        self.P_table = nn.Parameter(basis.detach().clone())
        self.R_QT = nn.Parameter(torch.eye(dim, dtype=basis.dtype))

    def keys(self, z_target: Tensor) -> Tensor:
        return z_target @ self.P_table.T

    def query(self, z_query: Tensor) -> Tensor:
        return (z_query @ self.P_table.T) @ self.R_QT

    def forward(self, z_query: Tensor, z_target: Tensor) -> Tensor:
        return self.query(z_query) @ self.keys(z_target).T
