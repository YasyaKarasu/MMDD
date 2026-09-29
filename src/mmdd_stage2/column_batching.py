"""Batch frozen column features without changing per-pair training semantics."""
from __future__ import annotations

from typing import Any

import torch

from .verifier import CandidateColumnScorer


def score_records(scorer: CandidateColumnScorer, records: list[dict[str, Any]], *,
                  execution: str = 'batched') -> list[torch.Tensor]:
    """Share dense matrix multiplies; keep each pair's original dropout draw.

    Columns are concatenated, never padded or pruned. The returned tensors retain
    pair boundaries, so the caller still normalizes its loss over *all* columns
    of each pair. Batched FP32 GEMMs can differ in their last bits from scalar
    GEMMs; this is not a promise of bit-identical trained checkpoints.
    """
    if execution not in {'scalar', 'batched'}:
        raise ValueError(f'Unknown head execution: {execution}')
    if execution == 'scalar':
        return [scorer(r['open_states'], r['close_states']) for r in records]
    if not records:
        return []
    for record in records:
        if record['open_states'].shape != record['close_states'].shape:
            raise ValueError('Opening and closing marker states must have equal shapes')
    lengths = [r['open_states'].shape[0] for r in records]
    states = torch.cat([torch.cat([r['open_states'], r['close_states']], dim=-1)
                        for r in records])
    if scorer.head_type == 'linear':
        logits = scorer.weight(states)
    else:
        linear, activation, dropout, output = scorer.weight
        hidden = activation(linear(states))
        # A single larger dropout call changes the RNG sequence. Retain the
        # original calls and shapes, including when candidate counts vary.
        hidden = torch.cat([dropout(part) for part in hidden.split(lengths)])
        logits = output(hidden)
    return list(logits.squeeze(-1).split(lengths))
