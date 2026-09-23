"""Batched Teacher scoring that reproduces ``FreshPathTeacher.score_pairs`` / ``score_triplets``.

The per-pair reference loop launches ~90 CUDA kernels per candidate (one
adapter Linear per object, per-pair segment adds, per-pair concatenations).
This module applies the same Linear/LayerNorm/embedding/pooler operations to
the concatenated object tokens once, then assembles exactly the same padded
sequence batch and runs the same Transformer, global MLP and head.  The
mathematics is unchanged (row-wise ops on concatenated rows); the difference
is float summation order only.  ``integration.probe_fast_scoring`` verifies
values and gradients against the reference implementation.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from fresh_path.models import KINDS


def _encode_kind(model, kind: str, z: Tensor, contents: Sequence[Tensor]) -> tuple[list[Tensor], Tensor]:
    """encode_one for many objects of one kind: returns per-object tokens and global g (n, width)."""
    lengths = [int(c.shape[0]) for c in contents]
    flat = torch.cat(list(contents), 0)
    x = model.adapters[kind](flat)
    if kind == "table":
        kinds = torch.ones(flat.shape[0], dtype=torch.long, device=flat.device)
        starts = torch.cumsum(torch.tensor(lengths, device=flat.device), 0) - torch.tensor(lengths, device=flat.device)
        kinds[starts] = 0
        x = x + model.table_kind(kinds)
        tokens = list(torch.split(x, lengths, 0))
    else:
        width = x.shape[1]
        n, lmax = len(lengths), max(lengths)
        padded = x.new_zeros(n, lmax, width)
        mask = torch.ones(n, lmax, dtype=torch.bool, device=x.device)
        pieces = torch.split(x, lengths, 0)
        for i, piece in enumerate(pieces):
            padded[i, : lengths[i]] = piece
            mask[i, : lengths[i]] = False
        pooler = model.poolers[kind]
        q = pooler.queries.unsqueeze(0).expand(n, -1, -1)
        pooled, _ = pooler.attn(q, padded, padded, key_padding_mask=mask, need_weights=False)
        out = pooler.norm(q + pooled)
        tokens = [out[i] for i in range(n)]
    g = model.globals[kind](z)
    return tokens, g


def _seg_extra(model, kind: str, role: int) -> Tensor:
    return model.modality.weight[KINDS.index(kind)] + model.roles.weight[role]


def _run(model, seqs: list[list[Tensor]], globs: Tensor) -> Tensor:
    width = model.width
    lengths = [sum(int(s.shape[0]) for s in segs) for segs in seqs]
    total = max(lengths)
    batch = len(seqs)
    device = globs.device
    flat = torch.cat([s for segs in seqs for s in segs], 0)
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
    out = model.relation(x, src_key_padding_mask=pad)
    local = out[:, 0]
    glob = model.global_relation(globs)
    return model.scoring_head(local + glob).squeeze(-1)


def pair_logits_batched(model, bank, anchor_id: str, dest_ids: Sequence[str], device, *, chunk: int = 256) -> Tensor:
    dest_ids = list(dest_ids)
    if not dest_ids:
        return torch.zeros(0, device=device)
    a_kind = bank.kind(anchor_id)
    a_tokens, ga = _encode_kind(model, a_kind, bank.z(anchor_id).to(device).unsqueeze(0), [bank.tokens(anchor_id).to(device)])
    seg_a = a_tokens[0] + _seg_extra(model, a_kind, 0)
    ga = ga[0]
    out = []
    for start in range(0, len(dest_ids), chunk):
        group = dest_ids[start : start + chunk]
        by_kind: dict[str, list[int]] = {}
        for i, d in enumerate(group):
            by_kind.setdefault(bank.kind(d), []).append(i)
        seg_d: list[Tensor | None] = [None] * len(group)
        gd = ga.new_empty(len(group), ga.shape[0])
        pair_emb = ga.new_empty(len(group), ga.shape[0])
        for kind, idx in by_kind.items():
            ids = [group[i] for i in idx]
            tokens, g = _encode_kind(model, kind, bank.z_many(ids).to(device), [bank.tokens(d).to(device) for d in ids])
            extra = _seg_extra(model, kind, 1)
            pe = model._pair_embedding(a_kind, kind)
            for j, i in enumerate(idx):
                seg_d[i] = tokens[j] + extra
            index = torch.tensor(idx, device=device)
            gd[index] = g
            pair_emb[index] = pe
        seqs = [[(model.rel + pair_emb[i]).unsqueeze(0), seg_a, model.sep.unsqueeze(0), seg_d[i]] for i in range(len(group))]
        ga_b = ga.unsqueeze(0).expand(len(group), -1)
        globs = torch.cat([ga_b, gd, ga_b * gd, (ga_b - gd).abs(), pair_emb], 1)
        out.append(_run(model, seqs, globs))
    return torch.cat(out)


def triplet_logits_batched(model, bank, query_id: str, pairs: Sequence[tuple[str, str]], device, *, chunk: int = 128) -> Tensor:
    pairs = list(pairs)
    if not pairs:
        return torch.zeros(0, device=device)
    q_tokens, gq = _encode_kind(model, "table", bank.z(query_id).to(device).unsqueeze(0), [bank.tokens(query_id).to(device)])
    seg_q = q_tokens[0] + _seg_extra(model, "table", 0)
    gq = gq[0]
    pair_emb = model._pair_embedding("table", "table")
    out = []
    for start in range(0, len(pairs), chunk):
        group = pairs[start : start + chunk]
        evidence = list(dict.fromkeys(e for e, _ in group))
        targets = list(dict.fromkeys(t for _, t in group))
        seg_e: dict[str, Tensor] = {}
        by_kind: dict[str, list[str]] = {}
        for e in evidence:
            by_kind.setdefault(bank.kind(e), []).append(e)
        for kind, ids in by_kind.items():
            tokens, g = _encode_kind(model, kind, bank.z_many(ids).to(device), [bank.tokens(e).to(device) for e in ids])
            extra = _seg_extra(model, kind, 2)
            for j, e in enumerate(ids):
                seg_e[e] = torch.cat([g[j].unsqueeze(0), tokens[j]], 0) + extra
        t_tokens, gt = _encode_kind(model, "table", bank.z_many(targets).to(device), [bank.tokens(t).to(device) for t in targets])
        extra_t = _seg_extra(model, "table", 1)
        seg_t = {t: t_tokens[j] + extra_t for j, t in enumerate(targets)}
        t_pos = {t: j for j, t in enumerate(targets)}
        head = (model.rel + pair_emb).unsqueeze(0)
        sep = model.sep.unsqueeze(0)
        seqs = [[head, seg_q, sep, seg_e[e], sep, seg_t[t]] for e, t in group]
        gt_b = gt[torch.tensor([t_pos[t] for _, t in group], device=device)]
        gq_b = gq.unsqueeze(0).expand(len(group), -1)
        globs = torch.cat([gq_b, gt_b, gq_b * gt_b, (gq_b - gt_b).abs(), pair_emb.unsqueeze(0).expand(len(group), -1)], 1)
        out.append(_run(model, seqs, globs))
    return torch.cat(out)
