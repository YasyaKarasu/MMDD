"""R28 split target supervision; coverage changes only Evidence logits."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import random

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .data import TargetExample
from .features import FeatureStore
from .objectives import PathAggregator
from .r26_training import edge_query_loss, graph_edges
from .scoring import TargetScores, score_edge_batch, score_target_batch, target_positive_mask
from .training import _anchor_losses, _path_supervised_losses


def row_support(rows: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
    """Frozen Qwen cosine support, [rows, paths], without a learned scorer."""
    return ((F.normalize(rows.detach().float(), dim=-1) @
             F.normalize(evidence.detach().float(), dim=-1).T + 1) / 2).clamp(0, 1).detach()


def coverage_logit(paths: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    coverage = (support.detach() * paths.sigmoid().unsqueeze(0)).max(dim=1).values.mean()
    return torch.logit(coverage.clamp(1e-6, 1 - 1e-6))


def coverage_scores(scores: TargetScores, examples: Sequence[TargetExample],
                    store: FeatureStore) -> TargetScores:
    """Retain exact Direct tensor/masks and all sorted historical path entries."""
    result = []
    for example, bags in zip(examples, scores.path_logits):
        rows = store.embedding_features(example.query_id).row_embeddings
        if rows is None or not len(rows):
            raise ValueError(f"Missing frozen row embeddings: {example.query_id}")
        values = []
        for candidate, paths in zip(example.candidates, bags):
            if not candidate.evidence_ids:
                values.append(scores.direct.logits.new_zeros(()))
                continue
            evidence = torch.stack([store.embedding_features(e).embedding
                                    for e in sorted(candidate.evidence_ids)])
            support = row_support(rows, evidence).to(paths.device)
            values.append(coverage_logit(paths, support))
        result.append(torch.stack(values))
    return TargetScores(scores.direct, replace(scores.evidence,
                        logits=pad_sequence(result, batch_first=True)), scores.path_logits)


def split_terms(scores: TargetScores, anchor: torch.Tensor) -> dict[str, torch.Tensor]:
    total, direct, evidence = _path_supervised_losses(scores, positive_loss_mode="sum_probability")
    return {"loss": total + anchor, "direct_supervised_loss": direct,
            "evidence_supervised_loss": evidence, "weighted_anchor_loss": anchor}


def objective(model, examples: Sequence[TargetExample], store: FeatureStore,
              device: torch.device, family: str, registry: dict, *, student: bool) -> dict:
    anchor = _anchor_losses(model, .1, .1)[1] if student else next(model.parameters()).new_zeros(())
    if family == "EDGE":
        edges, owners = [], []
        for owner, example in enumerate(examples):
            rows = graph_edges(example, registry, lambda oid: store.embedding_features(oid).object_type)
            edges.extend(rows)
            owners.extend([owner] * len(rows))
        scores = score_edge_batch(model, edges, store, device, student_score_space="raw_logit")
        return edge_query_loss(scores, edges, owners, len(examples), anchor)
    if family not in ("SPLIT-LSE", "SPLIT-COV"):
        raise ValueError(family)
    scores = score_target_batch(model, examples, store, device,
                                PathAggregator("logsumexp", 4, path_combination="sum"),
                                student_score_space="raw_logit")
    if family == "SPLIT-COV":
        scores = coverage_scores(scores, examples, store)
    return split_terms(scores, anchor)


def evidence_active_count(examples: Sequence[TargetExample]) -> int:
    count = 0
    for e in examples:
        mask = target_positive_mask([e], len(e.candidates), torch.device("cpu"), channel="evidence")[0]
        present = torch.tensor([bool(c.evidence_ids) for c in e.candidates])
        count += int((mask & present).any() and (~mask & present).any())
    return count


def backward_batch(model, examples: Sequence[TargetExample], store: FeatureStore,
                   device: torch.device, family: str, registry: dict, *,
                   student: bool, microbatch: int) -> dict:
    """Preserve logical-batch optional-E denominator during memory-bounded training."""
    active = evidence_active_count(examples) if family != "EDGE" else 0
    totals = {}
    for start in range(0, len(examples), microbatch):
        part = examples[start:start + microbatch]
        terms = objective(model, part, store, device, family, registry, student=student)
        query_weight = len(part) / len(examples)
        if family == "EDGE":
            weighted = {k: v * query_weight for k,v in terms.items() if isinstance(v, torch.Tensor)}
        else:
            e_weight = evidence_active_count(part) / max(1, active)
            weighted = {"direct_supervised_loss": terms["direct_supervised_loss"] * query_weight,
                        "evidence_supervised_loss": terms["evidence_supervised_loss"] * e_weight,
                        "weighted_anchor_loss": terms["weighted_anchor_loss"] * query_weight}
            weighted["loss"] = sum(weighted.values())
        weighted["loss"].backward()
        for key,value in weighted.items():
            totals[key] = totals.get(key, 0.) + float(value.detach())
    return totals


def shuffled_examples(examples: Sequence[TargetExample], source_groups: dict[str, str],
                      object_types: dict[str, str], seed: int = 280915) -> tuple[list[TargetExample], list[dict]]:
    """Donor selection uses only source identity and exact modality counts, never GT."""
    pools = {}
    def signature(candidate):
        return tuple(sorted(object_types[e] for e in candidate.evidence_ids))
    for example in examples:
        for candidate in example.candidates:
            if candidate.evidence_ids:
                pools.setdefault(signature(candidate), []).append((example.query_id, candidate))
    rng = random.Random(seed)
    result, receipts = [], []
    for example in examples:
        candidates = []
        for candidate in example.candidates:
            if not candidate.evidence_ids:
                candidates.append(candidate)
                continue
            eligible = [(q, c) for q, c in pools[signature(candidate)]
                        if source_groups[q] != source_groups[example.query_id]]
            if not eligible:
                raise ValueError("No cross-source donor with identical evidence modality composition")
            donor_q, donor = rng.choice(eligible)
            candidates.append(replace(candidate, evidence_ids=donor.evidence_ids))
            receipts.append({"query_id": example.query_id, "target_id": candidate.target_id,
                             "source_group": source_groups[example.query_id],
                             "donor_query_id": donor_q, "donor_target_id": donor.target_id,
                             "donor_source_group": source_groups[donor_q],
                             "evidence_ids": donor.evidence_ids, "signature": signature(candidate)})
        result.append(replace(example, candidates=tuple(candidates)))
    return result, receipts
