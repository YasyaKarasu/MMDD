"""Teacher and Student architectures for CLEAN-QET v4.0."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union
import torch
import gc
import traceback
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .execution_layout import TEACHER_INFERENCE_CHUNK
from .losses import global_features

KINDS = ("table", "text", "image")
STUDENT_RELATIONS = ("QT", "Q_text", "Q_image", "text_T", "image_T")
PATH_FROZEN_PREFIXES = ("adapters", "poolers", "globals", "modality", "table_kind")


class QueryPool(nn.Module):
    """Multi-query cross-attention pooling for variable-length token sequences."""

    def __init__(self, width: int, heads: int, slots: int):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(slots, width))
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: Tensor) -> Tensor:
        q = self.queries.unsqueeze(0).expand(x.size(0), -1, -1)
        pooled, _ = self.attn(q, x, x, need_weights=False)
        return self.norm(q + pooled)


class FreshPathTeacher(nn.Module):
    """Shared-head Teacher with 11h global_relation input."""

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
        # v4.0: global_relation input expanded to 11 * width
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

    def encode_one(self, kind: str, z: Tensor, content: Tensor) -> tuple[Tensor, Tensor]:
        if kind not in KINDS:
            raise ValueError(f"unknown object kind {kind!r}")
        adapter = self.adapters[kind]
        if content.dtype != adapter.weight.dtype:
            content = content.to(adapter.weight.dtype)
        if z.dtype != adapter.weight.dtype:
            z = z.to(adapter.weight.dtype)
        x = adapter(content)
        if kind == "table":
            kinds = torch.ones(len(x), dtype=torch.long, device=x.device)
            kinds[0] = 0
            x = x + self.table_kind(kinds)
        else:
            x = self.poolers[kind](x.unsqueeze(0))[0]
        g = self.globals[kind](z)
        return x, g

    def _encode_memo(
        self, cache: Optional[dict], kind: str, z: Tensor, content: Tensor, object_key=None
    ) -> tuple[Tensor, Tensor]:
        if cache is None:
            return self.encode_one(kind, z, content)
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

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, Tensor, Tensor, str, Tensor, Tensor]],
        cache: Optional[dict] = None,
        cache_keys: Optional[Sequence[tuple]] = None,
    ) -> Tensor:
        """pairs: (a_kind, z_a, C_a, b_kind, z_b, C_b) -> scores (N,)."""
        seqs: list[list[Tensor]] = []
        globs: list[Tensor] = []
        if cache_keys is not None and len(cache_keys) != len(pairs):
            raise ValueError("score_pairs cache_keys must align with pairs")
        for i, (a_kind, z_a, c_a, b_kind, z_b, c_b) in enumerate(pairs):
            key = None if cache_keys is None else cache_keys[i]
            ca, ga = self._encode_memo(cache, a_kind, z_a, c_a, None if key is None else key[0])
            cb, gb = self._encode_memo(cache, b_kind, z_b, c_b, None if key is None else key[1])
            pair = self._pair_embedding(a_kind, b_kind)
            seqs.append(
                [
                    (self.rel + pair).unsqueeze(0),
                    self._seg(a_kind, ca, ga, 0),
                    self.sep.unsqueeze(0),
                    self._seg(b_kind, cb, gb, 1),
                ]
            )
            # Empty evidence: exact 6h zeros extension via global_features
            globs.append(global_features(ga, gb, pair, evidence=None, evidence_type_embedding=None))
        return self._score(seqs, globs)

    def score_triplets(
        self,
        triplets: Sequence[tuple[str, Tensor, Tensor, str, Tensor, Tensor, str, Tensor, Tensor]],
        cache: Optional[dict] = None,
        cache_keys: Optional[Sequence[tuple]] = None,
    ) -> Tensor:
        """triplets: (q_kind, z_q, C_q, e_kind, z_e, C_e, t_kind, z_t, C_t) -> scores (N,)."""
        seqs: list[list[Tensor]] = []
        globs: list[Tensor] = []
        if cache_keys is not None and len(cache_keys) != len(triplets):
            raise ValueError("score_triplets cache_keys must align with triplets")
        for i, (q_kind, z_q, c_q, e_kind, z_e, c_e, t_kind, z_t, c_t) in enumerate(triplets):
            if q_kind != "table" or t_kind != "table" or e_kind == "table":
                raise ValueError("QET is only defined for table-evidence-table")
            key = None if cache_keys is None else cache_keys[i]
            cq, gq = self._encode_memo(cache, q_kind, z_q, c_q, None if key is None else key[0])
            ce, ge = self._encode_memo(cache, e_kind, z_e, c_e, None if key is None else key[1])
            ct, gt = self._encode_memo(cache, t_kind, z_t, c_t, None if key is None else key[2])
            pair = self._pair_embedding("table", "table")
            etype = self.modality.weight[KINDS.index(e_kind)]
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
            globs.append(global_features(gq, gt, pair, evidence=ge, evidence_type_embedding=etype))
        return self._score(seqs, globs)

    def _score(self, seqs: list[list[Tensor]], globs: Union[list[Tensor], Tensor]) -> Tensor:
        width = self.width
        n = len(seqs)
        if n == 0:
            dev = globs.device if isinstance(globs, Tensor) else (globs[0].device if globs else self.rel.device)
            return torch.empty(0, device=dev)
        lengths = [sum(len(s) for s in segs) for segs in seqs]
        total = max(lengths)
        batch = n
        device = globs.device if isinstance(globs, Tensor) else globs[0].device
        flat = torch.cat([s for segs in seqs for s in segs], dim=0)
        tokens = flat.shape[0]
        length_t = torch.tensor(lengths, device=device, dtype=torch.long)
        rows = torch.repeat_interleave(torch.arange(batch, device=device), length_t, output_size=tokens)
        starts = torch.cumsum(length_t, 0) - length_t
        cols = torch.arange(tokens, device=device) - torch.repeat_interleave(starts, length_t, output_size=tokens)
        dst = rows * total + cols
        x = flat.new_zeros(batch * total, width)
        x[dst] = flat
        x = x.view(batch, total, width)
        pad = torch.ones(batch * total, dtype=torch.bool, device=device)
        pad[dst] = False
        pad = pad.view(batch, total)
        out = self.relation(x, src_key_padding_mask=pad)
        local = out[:, 0]
        globs_t = globs if isinstance(globs, Tensor) else torch.stack(globs)
        glob = self.global_relation(globs_t)
        return self.scoring_head(local + glob).squeeze(-1)

    def encode_many(
        self, kind: str, z: Tensor, tokens: Sequence[Tensor]
    ) -> tuple[Tensor, list[int], Tensor]:
        """Batched encoding of objects of the same kind.
        Returns:
            segments: (N, Lmax, W) for table; (N, 1 + slots, W) for text/image with prepended g
            lengths: list of token lengths for each object
            g: (N, W) global features
        """
        if kind not in KINDS:
            raise ValueError(f"unknown object kind {kind!r}")
        n = len(tokens)
        adapter = self.adapters[kind]
        dev = z.device if isinstance(z, Tensor) else adapter.weight.device
        if n == 0:
            return torch.empty(0, 0, self.width, device=dev), [], torch.empty(0, self.width, device=dev)

        if z.dtype != adapter.weight.dtype:
            z = z.to(adapter.weight.dtype)

        if kind == "table":
            lengths = [t.shape[0] for t in tokens]
            pad_tokens = pad_sequence(tokens, batch_first=True)
            if pad_tokens.dtype != adapter.weight.dtype:
                pad_tokens = pad_tokens.to(adapter.weight.dtype)
            x = adapter(pad_tokens)  # (N, Lmax, W)
            kinds = torch.ones(x.shape[1], dtype=torch.long, device=x.device)
            kinds[0] = 0
            x = x + self.table_kind(kinds)
            g = self.globals["table"](z)
            return x, lengths, g
        else:
            lengths = [t.shape[0] for t in tokens]
            if len(set(lengths)) <= 1:
                stack_tokens = torch.stack(list(tokens))
                if stack_tokens.dtype != adapter.weight.dtype:
                    stack_tokens = stack_tokens.to(adapter.weight.dtype)
                x = adapter(stack_tokens)  # (N, L, W)
                pooled = self.poolers[kind](x)  # (N, slots, W)
            else:
                pooled_list = []
                for t in tokens:
                    if t.dtype != adapter.weight.dtype:
                        t = t.to(adapter.weight.dtype)
                    xt = adapter(t).unsqueeze(0)
                    pooled_list.append(self.poolers[kind](xt)[0])
                pooled = torch.stack(pooled_list)
            g = self.globals[kind](z)  # (N, W)
            seg = torch.cat([g.unsqueeze(1), pooled], dim=1)  # (N, 1 + slots, W)
            lengths = [seg.shape[1]] * n
            return seg, lengths, g

    def score_query_lists(self, q, targets, evidence, paths,
                          chunk: int = TEACHER_INFERENCE_CHUNK) -> tuple[Tensor, Tensor]:
        """Same full query scores, larger execution chunk; eval-only OOM retries.

        Training callers keep ordinary autograd semantics without implicit retries.
        Eval fallback retries the full query and never truncates target/path lists.
        """
        if chunk < 1:
            raise ValueError("inference chunk must be positive")
        if self.training or torch.is_grad_enabled():
            return self._score_query_lists_once(q, targets, evidence, paths, chunk)
        while True:
            failed = False
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
                failed = True
            if failed:
                old_chunk = chunk
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                chunk = max(1, chunk // 2)
                print(f"[Teacher inference OOM] full-query retry chunk={old_chunk}->{chunk}", flush=True)

    def _score_query_lists_once(
        self,
        q: tuple[Tensor, Tensor],
        targets: tuple[Tensor, Sequence[Tensor]],
        evidence: Mapping[str, tuple[str, Tensor, Tensor]],
        paths: Sequence[tuple[int, str]],
        chunk: int = 1024,
    ) -> tuple[Tensor, Tensor]:
        """Batched scoring for one query: returns f0 (Nt,) and path_scores (P,)."""
        zq, cq = q
        xq, gq = self.encode_one("table", zq, cq)
        pair = self._pair_embedding("table", "table")
        prefix = [(self.rel + pair).unsqueeze(0), self._seg("table", xq, gq, 0), self.sep.unsqueeze(0)]
        sep = self.sep.unsqueeze(0)

        zt, ct_list = targets
        Nt = len(ct_list)
        dev = zq.device

        if Nt > 0:
            x_t, L_t, g_t = self.encode_many("table", zt, ct_list)
            x_t = x_t + self.modality.weight[0] + self.roles.weight[1]
            seqs_pairs = [prefix + [x_t[k, :L_t[k]]] for k in range(Nt)]
            globs_pairs = global_features(gq.expand(Nt, -1), g_t, pair.expand(Nt, -1), None, None)
            f0 = torch.cat([self._score(seqs_pairs[i : i + chunk], globs_pairs[i : i + chunk]) for i in range(0, Nt, chunk)])
        else:
            f0 = torch.empty(0, device=dev)

        P = len(paths)
        if P > 0:
            e_ids = list(dict.fromkeys(p[1] for p in paths))
            seg_e: dict[str, Tensor] = {}
            g_e: dict[str, Tensor] = {}
            etype_e: dict[str, Tensor] = {}

            for k in ("text", "image"):
                ids_k = [e for e in e_ids if evidence[e][0] == k]
                if not ids_k:
                    continue
                k_idx = KINDS.index(k)
                z_k = torch.stack([evidence[e][1] for e in ids_k])
                toks_k = [evidence[e][2] for e in ids_k]
                seg_k, _, g_k = self.encode_many(k, z_k, toks_k)
                seg_k = seg_k + self.modality.weight[k_idx] + self.roles.weight[2]
                for i, e in enumerate(ids_k):
                    seg_e[e] = seg_k[i]
                    g_e[e] = g_k[i]
                    etype_e[e] = self.modality.weight[k_idx]

            t_rows = torch.tensor([p[0] for p in paths], device=dev, dtype=torch.long)
            seqs_trip = [prefix + [seg_e[p[1]], sep, x_t[p[0], :L_t[p[0]]]] for p in paths]
            globs_trip = global_features(
                gq.expand(P, -1),
                g_t[t_rows],
                pair.expand(P, -1),
                torch.stack([g_e[p[1]] for p in paths]),
                torch.stack([etype_e[p[1]] for p in paths]),
            )
            path_scores = torch.cat([self._score(seqs_trip[i : i + chunk], globs_trip[i : i + chunk]) for i in range(0, P, chunk)])
        else:
            path_scores = torch.empty(0, device=dev)

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
        centered = z - self.pca_mean
        return centered @ self.P[kind].T

    def query_vector(self, z_query: Tensor) -> Tensor:
        return self.u("table", z_query)

    def ann_query(self, relation: str, z_left: Tensor) -> Tensor:
        """Transformed row query whose dot with indexed right u equals bilinear score."""
        if relation not in STUDENT_RELATIONS:
            raise ValueError(f"unknown Student relation: {relation}")
        left_kind = "table" if relation.startswith("Q") else relation.removesuffix("_T")
        return self.u(left_kind, z_left) @ self.R[relation]

    def index_vectors(self, relation: str, z_right: Tensor) -> Tensor:
        if relation == "QT" or relation.endswith("_T"):
            right_kind = "table"
        elif relation == "Q_text":
            right_kind = "text"
        elif relation == "Q_image":
            right_kind = "image"
        else:
            raise ValueError(f"unknown Student relation: {relation}")
        return self.u(right_kind, z_right)

    def score(self, a_kind: str, z_a: Tensor, b_kind: str, z_b: Tensor) -> Tensor:
        ua = self.u(a_kind, z_a)
        ub = self.u(b_kind, z_b)
        rel_key = self._rel_key(a_kind, b_kind)
        return (ua @ self.R[rel_key] * ub).sum(dim=-1)

    def _rel_key(self, a_kind: str, b_kind: str) -> str:
        if a_kind == "table" and b_kind == "table":
            return "QT"
        if a_kind == "table" and b_kind == "text":
            return "Q_text"
        if a_kind == "table" and b_kind == "image":
            return "Q_image"
        if a_kind == "text" and b_kind == "table":
            return "text_T"
        if a_kind == "image" and b_kind == "table":
            return "image_T"
        raise ValueError(f"unsupported student relation pair ({a_kind}, {b_kind})")

    def anchor_loss(self) -> Tensor:
        p_loss = torch.stack(
            [F.mse_loss(self.P[k], self.pca_basis) for k in KINDS]
        ).mean()
        eye = torch.eye(self.dim, device=self.pca_basis.device, dtype=torch.float32)
        r_loss = torch.stack(
            [F.mse_loss(self.R[r], eye) for r in STUDENT_RELATIONS]
        ).mean()
        return p_loss + r_loss

    def param_groups(self, p_lr: float = 1e-6, r_lr: float = 1e-5) -> list[dict]:
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
        centered = z - self.pca_mean
        return centered @ self.P_table.T

    def query_vector(self, z_query: Tensor) -> Tensor:
        return self.u(z_query)

    def ann_query(self, z_left: Tensor) -> Tensor:
        return self.u(z_left) @ self.R_QT

    def index_vectors(self, z_right: Tensor) -> Tensor:
        return self.u(z_right)

    def score(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        ua = self.u(z_a)
        ub = self.u(z_b)
        return (ua @ self.R_QT * ub).sum(dim=-1)

    def anchor_loss(self) -> Tensor:
        p_loss = F.mse_loss(self.P_table, self.pca_basis)
        eye = torch.eye(self.dim, device=self.pca_basis.device, dtype=torch.float32)
        r_loss = F.mse_loss(self.R_QT, eye)
        return p_loss + r_loss

    def param_groups(self, p_lr: float = 1e-6, r_lr: float = 1e-5) -> list[dict]:
        return [
            {"params": [self.P_table], "lr": p_lr},
            {"params": [self.R_QT], "lr": r_lr},
        ]
