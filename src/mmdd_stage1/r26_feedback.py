"""Fixed-priority feedback gate and five-relation Teacher-list construction."""
from __future__ import annotations

from dataclasses import replace

from .data import EdgeExample

PRIORITY = ("O-UQTKD","O-QTKD","O-U","O-SUP","O-NATIVE","E-GRAPH")


def feedback_gate(metrics: dict, changed_lists: dict[str,int], *, extensions: bool = True) -> dict:
    priority = PRIORITY if extensions else PRIORITY[3:]
    records = []
    baseline = metrics.get("B13")
    if baseline is None:
        return {"status":"unassessable","reason":"missing_B13_own_retrieval","priority":priority,"candidates":records}
    for arm in priority:
        names = [f"R26-{arm}/seed{s}/step178" for s in (13,29)]
        values = []
        for name in names:
            model = metrics.get(name)
            values.append(None if model is None else {
                "U_raw_delta":model["overall"]["U"]["raw_recall"]-baseline["overall"]["U"]["raw_recall"],
                "implicit_EO_delta":model["implicit"]["admission"]["EO_ANN"]-baseline["implicit"]["admission"]["EO_ANN"]})
        record = {"arm":arm,"generators":names,"seed_deltas":dict(zip(("13","29"),values))}
        records.append(record)
        # A measured failure in either seed rules out this arm regardless of
        # the other seed; otherwise missing inputs cannot be skipped by rank.
        if any(v is not None and min(v.values()) < -.02-1e-12 for v in values):
            record["health"] = "failed"
            continue
        if any(v is None for v in values):
            return {"status":"unassessable","reason":"missing_priority_own_retrieval","priority":priority,"candidates":records}
        record["health"] = "passed"
        if any(name not in changed_lists for name in names):
            return {"status":"unassessable","reason":"missing_actual_H_lists","eligible_for_mining":names,
                    "priority":priority,"candidates":records}
        counts = [changed_lists[name] for name in names]
        if any(n < 0 for n in counts):
            raise ValueError("Hard membership counts cannot be negative")
        record["hard_membership_diff_lists"] = dict(zip(("13","29"),counts))
        if any(counts):
            return {"status":"triggered","selected_arm":arm,"selected_generators":names,"priority":priority,"candidates":records}
        record["hard_membership"] = "identical"
    return {"status":"not_triggered","reason":"all_priority_arms_fail_health_or_have_identical_hard_membership",
            "priority":priority,"candidates":records}


def augment_teacher_lists(examples: list[EdgeExample], hard: dict[str,list[str]], known: dict) -> tuple[list[EdgeExample],dict]:
    """Keep every historical list; append hard32 only to its corresponding QT list."""
    output = []
    added = protected = filtered_hard = 0
    for example in examples:
        key = (example.query_id,example.source_type,example.destination_type)
        positives = set(example.positive_ids)
        if not positives and example.positive_index >= 0:
            positives.add(example.candidate_ids[example.positive_index])
        positives |= known.get(key,set())
        ids = list(example.candidate_ids)
        if example.source_type == example.destination_type == "table":
            if example.query_id not in hard:
                raise ValueError("Every train QT query needs an actual hard list")
            mined = list(dict.fromkeys(hard[example.query_id]))
            if len(mined) > 32:
                raise ValueError("Refinement hard budget exceeds32")
            filtered_hard += sum(t in positives for t in mined)
            before = len(ids)
            ids = list(dict.fromkeys([*ids,*(t for t in mined if t not in positives)]))
            added += len(ids)-before
        positive_ids = tuple(t for t in ids if t in positives)
        protected += len(set(positive_ids)-set(example.positive_ids))
        old_labels = dict(zip(example.candidate_ids,example.confirmed_labels or (None,)*len(example.candidate_ids)))
        labels = tuple(1 if t in positives else old_labels.get(t) for t in ids)
        output.append(replace(example,candidate_ids=tuple(ids),positive_ids=positive_ids,
                              positive_index=ids.index(positive_ids[0]) if positive_ids else -1,confirmed_labels=labels))
    return output,{"lists":len(output),"appended_QT_candidates":added,"additional_known_positives_protected":protected,
                   "historical_or_mined_hard_positives_filtered":filtered_hard}
