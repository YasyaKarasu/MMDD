"""Frozen-R5 experiments: preparation, paired comparisons and inspectable recovery audits.

Preparation never reads labels. Recovery arms keep the source C30, table scores, selected
attributes and donor links. Only evidence selection or text localization changes.
"""
from __future__ import annotations

import csv
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import digest, iter_jsonl, read_json, write_json, write_jsonl
from .evaluate import _csv, bootstrap, metrics

ARMS = ("baseline", "teacher", "cosine", "span_lexical", "span_joint", "span_entity", "span_attribute",
        "text_only", "image_only")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_population(population: list[dict], groups: int) -> list[dict]:
    """Nested, label-blind source-group samples; zero means the full population."""
    if groups < 0:
        raise ValueError("groups must be nonnegative")
    order = sorted({r["source_group"] for r in population}, key=lambda g: digest(["R5_EVIDENCE_SPAN_V1", g]))
    keep = set(order[:groups] if groups else order)
    return [r for r in population if r["source_group"] in keep]


class CosineVectors:
    def __init__(self, folder: Path) -> None:
        meta = read_json(folder / "z_index.json")
        self.index = {key: i for i, key in enumerate(meta["ids"])}
        self.values = np.load(folder / "z.f32.npy", mmap_mode="r", allow_pickle=False)

    def score(self, query: str, evidence: str) -> float:
        a = np.asarray(self.values[self.index[query]], dtype=np.float32)
        b = np.asarray(self.values[self.index[evidence]], dtype=np.float32)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator == 0:
            raise ValueError("zero embedding in cosine baseline")
        return float(a @ b) / denominator


def choose_evidence(view: dict, paths: dict[str, dict[str, dict]], query_id: str, arm: str,
                    cosine: Any = None) -> tuple[list[str], list[dict]]:
    """One evidence per present modality from the SAME original view bag in both arms.

    Teacher score is mean path residual across donor links. Subtracting donor f0 prevents
    a donor's direct score from deciding which evidence wins across merged views.
    """
    candidates = []
    for evidence in view["evidence_ids"]:
        support = [paths[link["target_id"]][evidence] for link in view["donor_links"]]
        modalities = {r["modality"] for r in support}
        if len(modalities) != 1:
            raise ValueError("inconsistent evidence modality")
        score = (float(np.mean([r["residual"] for r in support])) if arm == "teacher"
                 else cosine.score(query_id, evidence))
        candidates.append({"evidence_id": evidence, "modality": support[0]["modality"], "score": score})
    chosen = []
    for modality in ("text", "image"):
        eligible = [r for r in candidates if r["modality"] == modality]
        if eligible:
            chosen.append(min(eligible, key=lambda r: (-r["score"], r["evidence_id"]))["evidence_id"])
    return chosen, candidates


def prepare(source: Path, output: Path, arm: str, stage1_run: Path | None, z_dir: Path | None,
            groups: int, splits: list[str]) -> None:
    source, output = source.resolve(), output.resolve()
    if arm not in ARMS:
        raise ValueError("unknown experiment arm")
    if output == source or source in output.parents or output.exists():
        raise ValueError("use a fresh output directory outside the frozen source run")
    config = read_json(source / "config.json")
    if config.get("text_span", {}).get("mode", "prefix") != "prefix":
        raise ValueError("experiment source must be the frozen prefix baseline")
    config["paths"]["run_root"] = str(output)
    if arm.startswith("span_"):
        config["text_span"] = {"mode": arm.removeprefix("span_"), "span_tokens": 192,
                               "window_tokens": 1024, "overlap_tokens": 128,
                               "layers": [15, 23, 27], "max_input_tokens": 8192}
    cosine = None
    if arm in {"teacher", "cosine"}:
        if stage1_run is None:
            raise ValueError("teacher/cosine arms require --stage1-run")
        if arm == "cosine":
            if z_dir is None:
                z_dir = Path(read_json(stage1_run / "protocol.json")["paths"]["pure_cache_dir"]) / "z"
            cosine = CosineVectors(z_dir)
    # Complete transformations before creating the output, so missing scored paths fail early.
    prepared, provenance, audit = {}, {}, []
    for split in ("dev", "test"):
        population = select_population(read_json(source / "population" / f"{split}.json"), groups) if split in splits else []
        ids = {q["query_id"] for q in population}
        plans = {r["query_id"]: r for r in iter_jsonl(source / "plans" / f"{split}.jsonl") if r["query_id"] in ids}
        if set(plans) != ids:
            raise ValueError("source plans do not cover selected population")
        path_map: dict[str, dict] = defaultdict(dict)
        if arm in {"teacher", "cosine"} and ids:
            identities = {r["query_id"]: r["stage1"] for r in iter_jsonl(
                Path(config["paths"]["stage1_handoff"]) / f"retrieval.{split}.jsonl") if r["query_id"] in ids}
            if set(identities) != ids or any(r["generator"] != "native_sup" for r in identities.values()):
                raise ValueError("Teacher experiment expects a complete native_sup handoff")
            logits = stage1_run / f"seed13/eval/{split}/native_sup/logits.TB_CQET.Real.jsonl.gz"
            for r in iter_jsonl(logits):
                if r["query_id"] in ids:
                    if r["teacher_state_hash"] != identities[r["query_id"]]["teacher_state_hash"]:
                        raise ValueError("Teacher logits do not match the frozen handoff Teacher")
                    path_map[r["query_id"]][r["target_id"]] = {
                        p["evidence_id"]: {"modality": p["modality"], "residual": p["raw_QET"] - r["f0"]}
                        for p in r["paths"]}
            provenance[str(logits)] = file_hash(logits)
        catalog = None
        if arm in {"text_only", "image_only"}:
            from .catalog import Catalog
            catalog = Catalog(source)
        for qid, plan in plans.items():
            for view in plan["views"]:
                original = list(view["evidence_ids"])
                scores = []
                if arm in {"teacher", "cosine"}:
                    view["evidence_ids"], scores = choose_evidence(view, path_map[qid], qid, arm, cosine)
                elif catalog is not None:
                    wanted = "text" if arm == "text_only" else "image"
                    view["evidence_ids"] = [e for e in original if catalog.get("asset", e)["asset_type"] == wanted]
                if view["evidence_ids"] != original:
                    view["view_id"] = digest([view["attribute"], view["evidence_ids"]])
                audit.append({"split": split, "query_id": qid, "view_id": view["view_id"],
                              "attribute": view["attribute"], "donor_links": view["donor_links"],
                              "original": original, "selected": view["evidence_ids"], "scores": scores})
        if catalog is not None:
            catalog.db.close()
        prepared[split] = (population, list(plans.values()))
        for relative in (f"population/{split}.json", f"plans/{split}.jsonl"):
            provenance[str(source / relative)] = file_hash(source / relative)
        handoff_path = Path(config["paths"].get("stage1_handoff", source)) / f"retrieval.{split}.jsonl"
        if handoff_path.exists():
            provenance[str(handoff_path)] = file_hash(handoff_path)
    provenance[str(source / "config.json")] = file_hash(source / "config.json")
    if z_dir is not None:
        for name in ("z_index.json", "z.f32.npy"):
            provenance[str(z_dir / name)] = file_hash(z_dir / name)
    output.mkdir(parents=True)
    (output / "catalog.sqlite").symlink_to(source / "catalog.sqlite")  # Catalog opens mode=ro.
    write_json(output / "config.json", config)
    for split, (population, plans) in prepared.items():
        write_json(output / "population" / f"{split}.json", population)
        write_jsonl(output / "plans" / f"{split}.jsonl", plans)
        (output / "recovery" / split).mkdir(parents=True)
        if arm == "baseline":
            for row in population:
                name = row["query_id"] + ".json"
                shutil.copy2(source / "recovery" / split / name, output / "recovery" / split / name)
    # Private, writable copies: never let score append to the frozen source's cache.
    if (source / "matching" / "vectors.npy").exists():
        (output / "matching").mkdir()
        for name in ("vectors.npy", "texts.jsonl"):
            shutil.copy2(source / "matching" / name, output / "matching" / name)
    write_jsonl(output / "EVIDENCE_SELECTION.jsonl", audit)
    code_root = Path(__file__).resolve().parent
    code_files = [code_root / n for n in ("experiments.py", "text_span.py", "recovery.py", "localizer.py",
                                         "rerank.py", "matching.py", "visible.py", "stage1.py", "evaluate.py")]
    code_files += [code_root.parent / "run_stage2.py", code_root.parent / "run_stage2_experiments.py"]
    own_files = [output / "config.json", output / "EVIDENCE_SELECTION.jsonl", *[output / kind / f"{split}.{suffix}"
                 for split in ("dev", "test") for kind, suffix in (("plans", "jsonl"), ("population", "json"))]]
    write_json(output / "EXPERIMENT.json", {"source": str(source), "arm": arm, "groups": groups,
               "splits": splits, "population_sha256": digest({s: p for s, (p, _) in prepared.items()}),
               "source_files": provenance, "code_files": {str(p): file_hash(p) for p in code_files},
               "frozen_files": {str(p): file_hash(p) for p in own_files},
               "frozen": ["C30", "table_scores", "selected_attributes", "donor_links", "image_crop_policy"],
               "teacher_scope": "one per modality within retained natural bag; fixed selector plans"})
    print({"run": str(output), "queries": {s: len(p) for s, (p, _) in prepared.items()}})


def verify(run: Path) -> None:
    """Reject resuming an experimental run after changing code, plans or configuration."""
    manifest = read_json(run / "EXPERIMENT.json")
    for category in ("code_files", "frozen_files"):
        for name, expected in manifest[category].items():
            if file_hash(Path(name)) != expected:
                raise ValueError(f"{category} changed: {name}; prepare a new arm rather than mixing cached outputs")


def paired_rows(method: list[dict], reference: list[dict], method_name: str, reference_name: str,
                metric_names: list[str], replicates: int = 10000) -> list[dict]:
    left = {(r["split"], r["query_id"]): r for r in method}
    right = {(r["split"], r["query_id"]): r for r in reference}
    if len(left) != len(method) or len(right) != len(reference) or left.keys() != right.keys():
        raise ValueError("paired comparison requires identical unique query populations")
    for key in left:
        if any(left[key][field] != right[key][field] for field in ("kind", "source_group")):
            raise ValueError("paired comparison population metadata mismatch")
    result = []
    for split in ("dev", "test"):
        for population in ("overall", "implicit", "explicit"):
            keys = sorted(k for k in left if k[0] == split and (population == "overall" or left[k]["kind"] == population))
            if not keys:
                continue
            for metric in metric_names:
                result.append({"split": split, "population": population, "method": method_name,
                               "reference": reference_name, "metric": metric, "queries": len(keys),
                               **bootstrap([left[k]["source_group"] for k in keys],
                                           [float(left[k][metric]) - float(right[k][metric]) for k in keys],
                                           replicates, 20260925)})
    return result


def compare(method: Path, reference: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    for run in (method, reference):
        if (run / "EXPERIMENT.json").exists():
            verify(run)
    manifests = [read_json(r / "EXPERIMENT.json") for r in (method, reference)
                 if (r / "EXPERIMENT.json").exists()]
    if len(manifests) == 2 and any(manifests[0][k] != manifests[1][k] for k in ("source", "population_sha256")):
        raise ValueError("experimental arms must share source and sampled population")
    # Reject changes to Stage-1 order or partial runs before producing any contrasts.
    for split in ("dev", "test"):
        runs = []
        for run in (method, reference):
            records = list(iter_jsonl(run / "scores" / f"{split}.jsonl"))
            expected = {r["query_id"] for r in read_json(run / "population" / f"{split}.json")}
            if len(records) != len(expected) or {r["query_id"] for r in records} != expected:
                raise ValueError("incomplete or duplicated score population")
            runs.append({r["query_id"]: r["rankings"]["STAGE1"] for r in records})
        if runs[0] != runs[1]:
            raise ValueError("recovery-arm comparison changed Stage-1 ranking/population")
    all_rows = []
    for run in (method, reference):
        with (run / "evaluation/PER_QUERY.csv").open() as handle:
            all_rows.append(list(csv.DictReader(handle)))
    contrasts = []
    for policy in ("BIDF_RRF60", "BRIDGE_RRF60", "VISIBLE_IDF_RRF60", "STAGE1"):
        contrasts.extend({"policy": policy, **r} for r in paired_rows(
            *[[r for r in rs if r["policy"] == policy] for rs in all_rows], method.name, reference.name,
            ["R10", "NDCG10", "R20", "NDCG20"]))
    _csv(output / "CONTRASTS.csv", contrasts)
    if all((r / "EVIDENCE_SELECTION.jsonl").exists() for r in (method, reference)):
        selections = [list(iter_jsonl(r / "EVIDENCE_SELECTION.jsonl")) for r in (method, reference)]
        if len(selections[0]) != len(selections[1]):
            raise ValueError("arms changed the number of planned views")
        changed = set()
        for a, b in zip(*selections):
            if any(a[k] != b[k] for k in ("split", "query_id", "attribute", "donor_links", "original")):
                raise ValueError("arms changed frozen plans or original evidence bags")
            if a["selected"] != b["selected"]:
                changed.add((a["split"], a["query_id"]))
        write_json(output / "SELECTION_DIFFERENCES.json", {"views": len(selections[0]),
                   "queries_with_different_evidence": len(changed), "query_keys": sorted(changed)})


def teacher_ranking(stage1_run: Path, source: Path, output: Path) -> None:
    """CPU-only ranking controls on the same C150; C30 result is conditional on R5 selection."""
    if output.exists():
        raise FileExistsError(output)
    config = read_json(source / "config.json")
    from mmdd_dataset.wdc_runtime import iter_dataset_artifact
    gold, reasons = defaultdict(set), defaultdict(set)
    for r in iter_dataset_artifact(Path(config["paths"]["dataset_root"]), "qrels"):
        if float(r.get("rel", 0)) > 0:
            gold[r["query_table_id"]].add(r["target_table_id"])
            reasons[r["query_table_id"]].add(r["reason"])
    result = []
    for split in ("dev", "test"):
        population = {r["query_id"]: r for r in read_json(source / "population" / f"{split}.json")}
        scores: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        for view in ("Real", "Swap"):
            path = stage1_run / f"seed13/eval/{split}/native_sup/logits.TB_CQET.{view}.jsonl.gz"
            for r in iter_jsonl(path):
                q, t = r["query_id"], r["target_id"]
                if q not in population:
                    continue
                residual = r["aggregated_score"] - r["f0"]
                scores[q]["FULL" if view == "Real" else "SWAP"][t] = r["f0"] + .5 * residual
                if view == "Real":
                    scores[q]["F0"][t] = r["f0"]
        for r in iter_jsonl(stage1_run / f"seed13/eval/{split}/native_sup/logits.TB_QT.Direct.jsonl.gz"):
            if r["query_id"] in population:
                scores[r["query_id"]]["QT_ONLY"][r["target_id"]] = r["aggregated_score"]
        handoff = {r["query_id"]: r for r in iter_jsonl(Path(config["paths"]["stage1_handoff"]) / f"retrieval.{split}.jsonl")}
        if set(scores) != set(population):
            raise ValueError("teacher logits missing selected queries")
        for q, policies in scores.items():
            if set(policies) != {"FULL", "F0", "SWAP", "QT_ONLY"} or any(set(v) != set(policies["FULL"]) for v in policies.values()):
                raise ValueError("Teacher contrasts must share the exact candidate set")
            for scope in ("C150", "R5_C30_CONDITIONAL"):
                allowed = set(policies["FULL"]) if scope == "C150" else {t["target_id"] for t in handoff[q]["results"][:30]}
                if not allowed <= set(policies["FULL"]):
                    raise ValueError("handoff contains unscored targets")
                for policy, values in policies.items():
                    order = sorted(allowed, key=lambda t: (-values[t], t))
                    kind = ("implicit" if reasons[q] == {"model_recoverable_join_column"} else
                            "explicit" if reasons[q] == {"explicit_visible_join_column"} else "mixed")
                    result.append({"query_id": q, "split": split, "source_group": population[q]["source_group"],
                                   "kind": kind, "scope": scope, "policy": policy,
                                   **metrics(order, gold[q], [10, 20])})
    contrasts = []
    for scope in ("C150", "R5_C30_CONDITIONAL"):
        for reference in ("F0", "SWAP", "QT_ONLY"):
            contrasts.extend({"scope": scope, **r} for r in paired_rows(
                [r for r in result if r["scope"] == scope and r["policy"] == "FULL"],
                [r for r in result if r["scope"] == scope and r["policy"] == reference],
                "FULL", reference, ["R10", "NDCG10", "R20", "NDCG20"]))
    _csv(output / "PER_QUERY.csv", result)
    _csv(output / "CONTRASTS.csv", contrasts)
    summary = []
    for split in ("dev", "test"):
        for scope in ("C150", "R5_C30_CONDITIONAL"):
            for population in ("overall", "implicit", "explicit"):
                for policy in ("FULL", "F0", "SWAP", "QT_ONLY"):
                    rows = [r for r in result if r["split"] == split and r["scope"] == scope and r["policy"] == policy
                            and (population == "overall" or r["kind"] == population)]
                    if rows:
                        summary.append({"split": split, "scope": scope, "population": population, "policy": policy,
                                        "queries": len(rows), **{m: float(np.mean([r[m] for r in rows]))
                                                                 for m in ("R10", "NDCG10", "R20", "NDCG20")}})
    _csv(output / "METRICS.csv", summary)


def audit_recovery(run: Path, output: Path, review_groups: int = 60) -> None:
    """Coverage/cost and a label-blind manual-review sample. VALUE is NOT correctness."""
    if output.exists():
        raise FileExistsError(output)
    from .catalog import Catalog
    catalog = Catalog(run)
    summary, review = [], []
    for split in ("dev", "test"):
        population = read_json(run / "population" / f"{split}.json")
        sampled = {r["query_id"] for r in select_population(population, review_groups)}
        for q in population:
            r = read_json(run / "recovery" / split / f"{q['query_id']}.json")
            tasks = r.get("tasks", [])
            slots = [s for b in r["bridges"] for s in b["slots"]]
            spans = [s for t in tasks for s in t.get("text_spans", [])]
            summary.append({**q, "status": r["status"], "model_inputs": r.get("model_inputs"),
                            "seconds": r.get("seconds"), "tasks": len(tasks),
                            "value_slots": sum(s["status"] == "VALUE" for s in slots),
                            "unique_rows_with_value": len({s["row_id"] for s in slots if s["status"] == "VALUE"}),
                            "conflict_slots": sum(s["status"] == "CONFLICT" for s in slots),
                            "span_forwards": r.get("text_localization", {}).get("forwards", 0),
                            "span_seconds": r.get("text_localization", {}).get("seconds", 0.0),
                            "generation_prompt_tokens": r.get("generation_tokens", {}).get("prompt_tokens"),
                            "generation_output_tokens": r.get("generation_tokens", {}).get("generated_tokens"),
                            "span_input_tokens": sum(s["input_tokens"] for s in spans),
                            "span_selected_tokens": sum(s["selected_tokens"] for s in spans)})
            if q["query_id"] in sampled:
                # One deterministic task per query avoids selecting only successful outputs.
                chosen = sorted(tasks, key=lambda t: digest(["REVIEW", t["task_id"]]))[:1]
                for task in chosen:
                    review.append({**q, "task": task, "row": catalog.get("query_rows", q["query_id"])[task["row_id"]],
                                   "evidence": [catalog.evidence(e) for e in task["evidence_ids"]],
                                   "annotation": {"entity_correct": None, "attribute_supported": None,
                                                  "value_correct": None, "span_contains_support": None,
                                                  "notes": "", "reviewer": ""}})
    catalog.db.close()
    _csv(output / "RECOVERY.csv", summary)
    write_jsonl(output / "MANUAL_REVIEW.jsonl", review)
    write_json(output / "NOTES.json", {"VALUE_is_correctness": False,
               "sample": "source-group hash then one task per query; includes null/error outputs",
               "cost": "span seconds are included in recovery seconds; do not add twice",
               "input_tokens": "span token counts describe localization input/output, not billed generator tokens"})
