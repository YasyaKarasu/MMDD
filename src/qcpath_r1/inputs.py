"""S0: resolve and lock the QCPATH-R1 input roles.

Implements EXPERIMENT_SPEC.zh-CN.md sections 2.1-2.3:
  * every role is located on the server and written with real path/size/sha256,
  * the repository commit and the production source modules are locked,
  * the retained production functions are described field by field.

Nothing here trains or loads a model.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from . import config


def _sha(path: Path) -> str:
    return config.file_sha256(path)


def _verify(path: Path, *, optional: bool = False) -> dict[str, Any]:
    """Verify one member file of a role against the S0-recorded hash."""
    entry = config.record(path)
    entry["expected_sha256"] = config.EXPECTED_SHA256.get(str(path))
    if not entry["exists"]:
        entry["status"] = "MISSING" if not optional else "OPTIONAL_MISSING"
        return entry
    if entry["expected_sha256"] and entry["sha256"] != entry["expected_sha256"]:
        entry["status"] = "HASH_MISMATCH"
    else:
        entry["status"] = "VERIFIED"
    return entry


def _role(
    role: str,
    rule: str,
    purpose: str,
    members: list[Path],
    *,
    training_read_allowed: bool,
    optional_members: list[Path] | None = None,
) -> dict[str, Any]:
    verified = [_verify(path) for path in members]
    verified += [_verify(path, optional=True) for path in (optional_members or [])]
    blocking = [entry for entry in verified if entry["status"] in {"MISSING", "HASH_MISMATCH"}]
    return {
        "role": role,
        "resolution_rule": rule,
        "purpose": purpose,
        "training_read_allowed": training_read_allowed,
        "status": "RESOLVED" if not blocking else "BLOCKED",
        "members": verified,
    }


def _teacher_coverage(all_evidence_ids: list[str], retained_evidence_ids: list[str]) -> dict[str, Any]:
    """Coverage of the frozen T0 pooling-before cache (spec sections 9.1, 9.4).

    The cache is spread over the primary feature directory plus every historical
    backfill.  An object is usable only when its manifest entry resolves to a
    file that still exists on disk.
    """
    resolvable: dict[str, str] = {}
    manifests = sorted(set(config.ROOT.glob("work/**/teacher_manifest.jsonl")))
    for manifest in manifests:
        base = manifest.parent
        for line in manifest.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            object_id = str(row.get("object_id") or "")
            feature = row.get("teacher_feature_path")
            if not object_id or not feature or object_id in resolvable:
                continue
            full = base / feature
            if full.is_file():
                resolvable[object_id] = str(full)
    counts: dict[str, int] = {}
    for object_id in resolvable:
        counts[object_id.split("_")[0]] = counts.get(object_id.split("_")[0], 0) + 1

    content_key_to_ids: dict[str, list[str]] = {}
    object_key: dict[str, str] = {}
    for line in config.CONTENT_KEYS.open(encoding="utf-8"):
        row = json.loads(line)
        object_key[str(row["object_id"])] = str(row["content_key"])
        content_key_to_ids.setdefault(str(row["content_key"]), []).append(str(row["object_id"]))

    def _partition(ids: list[str]) -> dict[str, Any]:
        missing = sorted({eid for eid in ids if eid not in resolvable})
        alias = [
            eid for eid in missing
            if any(sibling in resolvable for sibling in content_key_to_ids.get(object_key.get(eid, ""), []))
        ]
        truly = [eid for eid in missing if eid not in set(alias)]
        return {
            "slots": len(ids),
            "distinct_missing_direct": len(missing),
            "content_alias_resolvable": len(alias),
            "truly_missing": len(truly),
            "truly_missing_ids": truly,
            "coverage_fraction": 1.0 - len(missing) / len(ids) if ids else None,
        }

    target_ids = json.loads((config.TARGET_INDEX / "table_ids.json").read_text())
    retained = _partition(retained_evidence_ids)
    return {
        "manifests_scanned": [str(path) for path in manifests],
        "resolvable_objects": len(resolvable),
        "by_id_prefix": counts,
        "target_tables_required": len(target_ids),
        "target_tables_covered": sum(1 for t in target_ids if t in resolvable),
        "dev_first_hop_evidence": _partition(all_evidence_ids),
        "dev_retained_evidence": retained,
        "status": "COMPLETE" if not retained["truly_missing"] else "INCOMPLETE_BLOCKS_B",
        "spec_rule": (
            "section 9.1: missing cache rows must not be papered over; the round may "
            "not re-encode the lake, so B is blocked and reported; section 9.4: slots "
            "may not be deleted later when a feature turns out to be missing."
        ),
    }


def _dev_evidence_ids() -> tuple[list[str], list[str]]:
    """Return (all dev first-hop evidence, retained dev evidence).

    The second list is what the B Teacher actually forwards: the retained real
    paths recorded by the locked production admission for the dev population.
    """
    import gzip

    all_ids: list[str] = []
    retained: list[str] = []
    with gzip.open(config.BASE_RANKINGS, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for target in row.get("E_pre_retention", []):
                for path in target.get("paths", []):
                    if path.get("kind") == "evidence":
                        all_ids.append(str(path["evidence_id"]))
            for target in row.get("E_paths", []):
                for evidence_id in target.get("selected_evidence_ids", []):
                    retained.append(str(evidence_id))
    return all_ids, retained


def resolve_inputs() -> dict[str, Any]:
    """Build RESOLVED_INPUTS.json and return it."""
    roles = [
        _role(
            "dataset_root",
            "the 20K EntiTables snapshot shared by the original QC-ET/B13 rounds; "
            "original split and object library retained",
            "source of the raw train/dev annotations and the data-lake table corpus",
            [config.DATASET_QRELS, config.DATASET_RECOVERIES, config.DATASET_SPLITS,
             config.DATASET_MANIFEST, config.DATASET_QUERY_TABLES],
            training_read_allowed=True,
        ),
        _role(
            "raw_train_annotations",
            "original train qrels plus query-specific witness/recovery annotations",
            "rebuild G/D/W/Epos/P for round-1 supervision (section 3.1)",
            [config.DATASET_QRELS, config.DATASET_RECOVERIES, config.SPLIT_PROTOCOL],
            training_read_allowed=True,
            optional_members=[config.TRAIN_WITNESS],
        ),
        _role(
            "dev_annotations",
            "original development set, read only by the evaluation entrypoint",
            "evaluation population and verified dev witnesses",
            [config.DEV_WITNESS, config.DATASET_QRELS, config.DATASET_RECOVERIES],
            training_read_allowed=False,
        ),
        _role(
            "student_base",
            "2026-09-16 QC-ET INPUT_LOCK parent: R27 B13 exact-replay final healthy Student; "
            "weights verified, not just the filename",
            "frozen BASE used by both A arms; supplies projections, relations and index vectors",
            [config.STUDENT_BASE],
            training_read_allowed=True,
        ),
        _role(
            "frozen_object_embeddings",
            "same Qwen object encodings as BASE, with prompt/tokenization/object-content fingerprint",
            "input to every frozen projection; no new Qwen forward this round",
            [config.FEATURE_MANIFEST, config.FEATURES / "metadata.json",
             config.FEATURES / "merge_summary.json"],
            training_read_allowed=True,
        ),
        _role(
            "target_vectors/index",
            "BASE production target vectors and index; text/image must share the same static space",
            "legal target universe for the full-target denominator and ANN evaluation",
            [config.TARGET_INDEX / "manifest.json", config.TARGET_INDEX / "table_ids.json",
             config.TARGET_INDEX / "table.hnsw"],
            training_read_allowed=True,
        ),
        _role(
            "base_first_hop",
            "BASE complete Direct and 20-text + 20-image first hop",
            "D_q / E_q^BASE and the strict-cohort exclusions",
            [config.BASE_RANKINGS],
            training_read_allowed=False,
        ),
        _role(
            "production_functions",
            "original QC-ET Evidence ranking, retention, Equal, filtering and tie-break functions",
            "candidate budget only; never a substitute for the final Teacher",
            [config.ROOT / "src/run_stage1_r11_task_e.py",
             config.ROOT / "src/mmdd_stage1/r26_metrics.py",
             config.ROOT / "src/mmdd_stage1/retrieval.py",
             config.ROOT / "src/evaluate_stage1_r26.py",
             config.ROOT / "src/mmdd_stage1/row_support.py",
             config.CONTENT_KEYS],
            training_read_allowed=True,
        ),
        _role(
            "teacher_parent",
            "original strong T0: R22 T1-B seed13, historical sha256 ab0e3c3f...",
            "the only permitted Teacher parent for B; never substituted",
            [config.TEACHER_PARENT],
            training_read_allowed=True,
        ),
        _role(
            "teacher_feature_store",
            "T0-compatible fine-grained object features and object global z; not a CLEAN-R1 summary",
            "frozen compressor inputs for the real Q/E/T forward",
            [config.TEACHER_MANIFEST],
            training_read_allowed=True,
            optional_members=[config.TEACHER_OBJECTS],
        ),
    ]

    protocol = config.load_protocol()
    all_evidence, retained_evidence = _dev_evidence_ids()
    coverage = _teacher_coverage(all_evidence, retained_evidence)
    teacher_role = next(role for role in roles if role["role"] == "teacher_feature_store")
    if coverage["status"] == "INCOMPLETE_BLOCKS_B":
        teacher_role["status"] = "BLOCKED_FOR_B"
    teacher_role["coverage"] = coverage

    payload = {
        "protocol_id": protocol["protocol_id"],
        "protocol_version": protocol["version"],
        "stage": "S0",
        "resolved_by": "run_stage1_qcpath_r1.py resolve-inputs",
        "dataset_root": str(config.DATASET_ROOT),
        "run_root": str(config.RUN),
        "roles": roles,
        "role_status_summary": {role["role"]: role["status"] for role in roles},
        "blocking_roles_for_A": [
            role["role"] for role in roles
            if role["status"] == "BLOCKED"
            and role["role"] not in {"teacher_parent", "teacher_feature_store"}
        ],
        "blocking_roles_for_B": [
            role["role"] for role in roles if role["status"] in {"BLOCKED", "BLOCKED_FOR_B"}
        ],
        "historical_identity_clues": {
            "targets": 22886,
            "dev_queries": 1198,
            "implicit_explicit": [599, 599],
            "probe_pairs": 939,
            "historical_train_pairs": 14592,
            "note": "identity clues only; this round re-enumerates and explains differences",
        },
    }
    config.RUN.mkdir(parents=True, exist_ok=True)
    (config.RUN / "RESOLVED_INPUTS.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def source_lock() -> dict[str, Any]:
    """SOURCE_LOCK.json: repository commit, dirty diff, imported module hashes."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=config.ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=config.ROOT, capture_output=True, text=True, check=True
    ).stdout.strip().splitlines()
    modules = {}
    for relative, expected in config.SOURCE_HASHES.items():
        path = config.ROOT / relative
        actual = _sha(path)
        modules[relative] = {
            "sha256": actual,
            "expected_sha256": expected,
            "matches_spec_snapshot": actual == expected,
        }
    payload = {
        "stage": "S0",
        "repository_commit": commit,
        "dirty_entries": dirty,
        "dirty_count": len(dirty),
        "modules": modules,
        "all_locked_modules_unmodified_at_commit": all(
            not any(entry.endswith(relative) for entry in dirty) for relative in modules
        ),
        "role": (
            "the QC-ET production modules used by QCPATH-R1 are clean at the locked commit; "
            "no whole-repository file-by-file hashing is performed (section 2.2)"
        ),
    }
    (config.RUN / "SOURCE_LOCK.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def production_contract() -> dict[str, Any]:
    """PRODUCTION_CONTRACT.json: field-by-field description of retained functions."""
    payload = {
        "stage": "S0",
        "frozen": True,
        "redefinition_allowed": False,
        "functions": [
            {
                "qualified_name": "mmdd_stage1.retrieval.StudentANNIndices.search_many",
                "source": "src/mmdd_stage1/retrieval.py",
                "signature": "search_many(self, source_ids: list[str], destination_type: str, k: int) "
                             "-> list[list[tuple[str, float]]]",
                "non_default_arguments": {"destination_type": "table", "k": 100},
                "score_space": "raw inner product (hnswlib space='ip', returns 1.0 - distance)",
                "retention_rule": None,
                "dedup_key": None,
                "ranking_and_tie_rule": "hnswlib label order; the protocol re-sorts any returned set by "
                                        "(-score, target_id)",
                "candidate_source_order": "not applicable (single direct channel)",
            },
            {
                "qualified_name": "mmdd_stage1.retrieval.retrieve_zero_one_hop_detailed_many",
                "source": "src/mmdd_stage1/retrieval.py",
                "signature": "retrieve_zero_one_hop_detailed_many(query_ids, indices, *, k=10, direct_k=None, "
                             "evidence_k=None, targets_per_evidence=None, evidence_types=('text','image'), "
                             "evidence_aggregation='logsumexp', query_batch_size=32, ...)",
                "non_default_arguments": {"direct_k": 100, "evidence_k": 20,
                                          "targets_per_evidence": 20,
                                          "evidence_types": ["text", "image"],
                                          "evidence_aggregation": "logsumexp",
                                          "query_batch_size": 16},
                "score_space": "raw inner product for every hop",
                "retention_rule": "per-evidence Top-20 returned by the target ANN; no extra budget per evidence",
                "dedup_key": None,
                "ranking_and_tie_rule": "(-score, target_id)",
                "candidate_source_order": "Direct channel then evidence channel; retained order preserved",
            },
            {
                "qualified_name": "run_stage1_r11_task_e.select_evidence",
                "source": "src/run_stage1_r11_task_e.py",
                "signature": "select_evidence(strategy, paths, *, query_id, store, content_keys, top_l, budget, "
                             "support_cache) -> tuple[list[str], float | None]",
                "non_default_arguments": {"strategy": "e2_row_coverage", "top_l": 20, "budget": 4,
                                          "threshold": 0.0},
                "score_space": "path_score = query_evidence_score + evidence_target_score; quality = sigmoid(path_score)",
                "retention_rule": "content-key dedup -> first top_l=20 by (-path_score, evidence_id) -> "
                                  "greedy_row_bundle(budget=4, threshold=0.0)",
                "dedup_key": "evidence content_key from evidence_content_keys.jsonl",
                "ranking_and_tie_rule": "candidate order (-path_score, evidence_id); greedy choice "
                                        "(-gain, -quality, evidence_id); stop when gain <= 0",
                "candidate_source_order": "evidence channel only",
            },
            {
                "qualified_name": "mmdd_stage1.r26_metrics.fuse_channels",
                "source": "src/mmdd_stage1/r26_metrics.py",
                "signature": "fuse_channels(direct, evidence: Sequence[dict] | None, column_alpha=None) -> dict",
                "non_default_arguments": {"column_alpha": None},
                "score_space": "reciprocal rank fusion with k=60",
                "retention_rule": "Equal ranking truncated to the first 100 for C100",
                "dedup_key": "target_id",
                "ranking_and_tie_rule": "d + e summed per target, sorted by (-score, target_id)",
                "candidate_source_order": "direct channel union evidence channel",
            },
            {
                "qualified_name": "evaluate_stage1_r26.retain_evidence",
                "source": "src/evaluate_stage1_r26.py",
                "signature": "retain_evidence(query_id, retrieved, store, content_keys) -> list[dict]",
                "non_default_arguments": {"strategy": "e2_row_coverage", "top_l": 20, "budget": 4},
                "score_space": "row coverage strength in [0, 1]",
                "retention_rule": "identical to select_evidence; final sort (-evidence_score, target_id)",
                "dedup_key": "evidence content_key",
                "ranking_and_tie_rule": "(-evidence_score, target_id)",
                "candidate_source_order": "evidence channel only",
            },
            {
                "qualified_name": "mmdd_stage1.row_support.greedy_row_bundle",
                "source": "src/mmdd_stage1/row_support.py",
                "signature": "greedy_row_bundle(candidates, *, row_support, budget, threshold) -> tuple[list[str], float]",
                "non_default_arguments": {"budget": 4, "threshold": 0.0},
                "score_space": "coverage = mean over query rows of the best covered support",
                "retention_rule": "greedy until budget or no positive gain",
                "dedup_key": None,
                "ranking_and_tie_rule": "min by (-gain, -quality, evidence_id)",
                "candidate_source_order": "input order",
            },
            {
                "qualified_name": "mmdd_stage1.models.StudentJoinabilityModel.project",
                "source": "src/mmdd_stage1/models.py",
                "signature": "project(self, embedding, object_type, *, role=None) -> Tensor",
                "non_default_arguments": {"role": "query for query tables, target for target tables, None for evidence"},
                "score_space": "frozen linear projection P_type, no post-normalization",
                "retention_rule": None,
                "dedup_key": None,
                "ranking_and_tie_rule": None,
                "candidate_source_order": None,
            },
            {
                "qualified_name": "mmdd_stage1.models.StudentJoinabilityModel.relation_query",
                "source": "src/mmdd_stage1/models.py",
                "signature": "relation_query(self, source_embedding, source_type, destination_type, *, "
                             "source_role=None) -> Tensor",
                "non_default_arguments": {"destination_type": "table"},
                "score_space": "row-vector convention v_e = u_e @ R[e_type, table]; NOT u_e @ R.T",
                "retention_rule": None,
                "dedup_key": None,
                "ranking_and_tie_rule": None,
                "candidate_source_order": None,
            },
            {
                "qualified_name": "mmdd_stage1.models.StudentJoinabilityModel.index_vector",
                "source": "src/mmdd_stage1/models.py",
                "signature": "index_vector(self, destination_embedding, destination_type, source_type=None, *, "
                             "destination_role=None) -> Tensor",
                "non_default_arguments": {"destination_type": "table", "destination_role": "target"},
                "score_space": "full-rank branch returns the projected target vector unchanged",
                "retention_rule": None,
                "dedup_key": None,
                "ranking_and_tie_rule": None,
                "candidate_source_order": None,
            },
            {
                "qualified_name": "mmdd_stage1.models.StudentJoinabilityModel.transform_edge_scores",
                "source": "src/mmdd_stage1/models.py",
                "signature": "transform_edge_scores(self, raw_scores, source_type, destination_type, score_space) -> Tensor",
                "non_default_arguments": {"score_space": "raw_logit"},
                "score_space": "identity: R27 retrieval consumed the raw bilinear logit; D1 applies its own sigmoid",
                "retention_rule": None,
                "dedup_key": None,
                "ranking_and_tie_rule": None,
                "candidate_source_order": None,
            },
            {
                "qualified_name": "mmdd_stage1.checkpoints.load_student",
                "source": "src/mmdd_stage1/checkpoints.py",
                "signature": "load_student(path: Path, device: torch.device) -> StudentJoinabilityModel",
                "non_default_arguments": {},
                "score_space": None,
                "retention_rule": None,
                "dedup_key": None,
                "ranking_and_tie_rule": None,
                "candidate_source_order": None,
            },
        ],
        "target_index_identity": {
            "space": "ip",
            "student_dim": 1024,
            "ann_dim": 1024,
            "relation_param": "full",
            "hnsw_m": 32,
            "ef_construction": 200,
            "ef_search": 100,
            "table_objects": 22886,
            "text_objects": 84689,
            "image_objects": 141031,
            "table_ids_sha256": config.EXPECTED_SHA256[str(config.TARGET_INDEX / "table_ids.json")],
            "table_hnsw_sha256": config.EXPECTED_SHA256[str(config.TARGET_INDEX / "table.hnsw")],
        },
        "query_model_identity": {
            "student_checkpoint": str(config.STUDENT_BASE),
            "student_checkpoint_sha256": config.EXPECTED_SHA256[str(config.STUDENT_BASE)],
            "round_adapters": "new in this round; identity of the frozen part is unchanged",
        },
    }
    (config.RUN / "PRODUCTION_CONTRACT.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def resolve_all() -> dict[str, Any]:
    config.RUN.mkdir(parents=True, exist_ok=True)
    (config.RUN / "protocol.json").write_text(
        config.PROTOCOL_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    inputs = resolve_inputs()
    lock = source_lock()
    contract = production_contract()
    return {
        "resolved_inputs": inputs["role_status_summary"],
        "blocking_roles_for_A": inputs["blocking_roles_for_A"],
        "blocking_roles_for_B": inputs["blocking_roles_for_B"],
        "teacher_coverage": next(
            role for role in inputs["roles"] if role["role"] == "teacher_feature_store"
        )["coverage"],
        "source_lock_matches_spec_snapshot": all(
            entry["matches_spec_snapshot"] for entry in lock["modules"].values()
        ),
        "production_functions": len(contract["functions"]),
    }
