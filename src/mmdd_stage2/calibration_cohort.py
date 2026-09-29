"""Nested, label-blind train cohorts with whole source groups kept together."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def choose_cohort(
    population: Sequence[dict[str, Any]], partitions: Mapping[str, str],
    fit_queries: int, validation_queries: int, seed: int = 13,
) -> dict[str, Any]:
    """Select a fixed random group order separately within fit and inner-val.

    Budgets are minimum query counts: the last source group is never split.
    Increasing the fit budget preserves the validation set and all smaller fit
    cohorts. Labels, selector outcomes, and generation success are not inputs.
    """
    if min(fit_queries, validation_queries) <= 0:
        raise ValueError("Both fit and validation budgets must be positive")
    groups: dict[str, list[str]] = defaultdict(list)
    roles: dict[str, str] = {}
    seen: set[str] = set()
    for row in population:
        q, group = row["query_id"], row["source_group"]
        if row["split"] != "train" or q in seen or not group:
            raise ValueError("Expected unique train queries with source groups")
        seen.add(q)
        role = partitions[q]
        if role not in {"fit", "inner_val"} or roles.get(group, role) != role:
            raise ValueError("Source group crosses fit and inner-val")
        roles[group] = role
        groups[group].append(q)
    selected: dict[str, list[str]] = {}
    selected_groups: dict[str, list[str]] = {}
    for role, budget in [("fit", fit_queries), ("inner_val", validation_queries)]:
        order = sorted((g for g in groups if roles[g] == role),
                       key=lambda g: (hashlib.sha256(f"{seed}\0{role}\0{g}".encode()).digest(), g))
        ids: list[str] = []
        picked: list[str] = []
        for group in order:
            ids.extend(sorted(groups[group]))
            picked.append(group)
            if len(ids) >= budget:
                break
        if len(ids) < budget:
            raise ValueError(f"Not enough {role} queries for budget {budget}")
        selected[role], selected_groups[role] = ids, picked
    ids = sorted(selected["fit"] + selected["inner_val"])
    return {"query_ids": ids, "partitions": {q: partitions[q] for q in ids},
            "source_groups": selected_groups, "seed": seed,
            "requested_fit_queries": fit_queries, "requested_validation_queries": validation_queries,
            "fit_queries": len(selected["fit"]), "validation_queries": len(selected["inner_val"]),
            "sampling": "nested_uniform_source_groups_within_original_partition",
            "labels_used_for_selection": False, "full_train_queries": len(population)}
