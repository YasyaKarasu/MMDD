"""Query-bound common-graph supervision for the R26 Edge/path comparison."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

import torch

from .data import EdgeExample, TargetExample
from .r25_objectives import _optional_list_loss
from .scoring import ListScores


def graph_edges(example: TargetExample, known: Mapping[tuple[str, str, str], set[str]],
                object_type: Callable[[str], str]) -> list[EdgeExample]:
    """Only actual graph memberships, with labels from the train registry."""
    candidates = defaultdict(set)
    candidates[(example.query_id, "table", "table")].update(c.target_id for c in example.candidates)
    for target in example.candidates:
        for evidence in target.evidence_ids:
            modality = object_type(evidence)
            candidates[(example.query_id, "table", modality)].add(evidence)
            candidates[(evidence, modality, "table")].add(target.target_id)
    rows = []
    for key, values in sorted(candidates.items()):
        ids = tuple(sorted(values))
        positives = tuple(sorted(values & known.get(key, set())))
        rows.append(EdgeExample(key[0], ids, ids.index(positives[0]) if positives else -1,
                                source_type=key[1], destination_type=key[2], positive_ids=positives,
                                split="train", dataset=example.dataset))
    return rows


def edge_query_loss(scores: ListScores, examples: Sequence[EdgeExample], owners: Sequence[int],
                    query_count: int, anchor: torch.Tensor) -> dict:
    """D + .5 mean(QE) + .5 mean(ET), then mean over the frozen queries."""
    mask = scores.candidate_mask
    positive = scores.positive_mask & mask
    active = positive.any(-1) & (mask & ~positive).any(-1)
    losses = scores.logits.sum(-1) * 0
    selected_logits = scores.logits[active]
    losses[active] = torch.logsumexp(selected_logits.masked_fill(~mask[active], -torch.inf), -1) - torch.logsumexp(selected_logits.masked_fill(~positive[active], -torch.inf), -1)
    owner_tensor = torch.tensor(owners, device=scores.logits.device)
    kinds = ["D" if e.source_type == e.destination_type == "table" else "QE" if e.source_type == "table" else "ET" for e in examples]
    terms = {}
    for kind in ("D", "QE", "ET"):
        selected = torch.tensor([k == kind for k in kinds], device=scores.logits.device)
        sums = scores.logits.new_zeros(query_count).scatter_add(0, owner_tensor[selected], losses[selected])
        counts = scores.logits.new_zeros(query_count).scatter_add(0, owner_tensor[selected], torch.ones_like(losses[selected]))
        terms[kind] = (sums / counts.clamp_min(1)).mean()
    relation_counts = defaultdict(int)
    for e, is_active in zip(examples, active.detach().cpu().tolist()):
        relation_counts[f"{e.source_type}->{e.destination_type}"] += int(is_active)
    return {"loss": terms["D"] + .5 * terms["QE"] + .5 * terms["ET"] + anchor,
            "direct_supervised_loss": terms["D"], "qe_supervised_loss": terms["QE"], "et_supervised_loss": terms["ET"],
            "weighted_anchor_loss": anchor, "active_relations": dict(relation_counts)}
