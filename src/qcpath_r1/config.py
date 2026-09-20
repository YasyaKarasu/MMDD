"""QCPATH-R1 frozen configuration: protocol, locked paths, run layout.

Everything here is read from the delivered specification package and the
resolved server inputs.  No stage may use a function default instead of a
value recorded here (EXPERIMENT_SPEC.zh-CN.md section 16).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path("/home/oycy/MMDD")
SPEC_DIR = ROOT / "MMDD_QCPATH_R1_20260919"
PROTOCOL_PATH = SPEC_DIR / "protocol.json"

#: The delivered specification's proposed ``run/`` directory, rooted at the
#: experiment work directory.  All run artifacts live below this path.
RUN = ROOT / "work/qcpath_r1_20260919"

# ---------------------------------------------------------------- spec assets

#: Roles resolved in S0 (EXPERIMENT_SPEC.zh-CN.md section 2.1).  Each entry is a
#: list of member files; ``required`` marks a role whose absence blocks A.
DATASET_ROOT = (
    ROOT / "output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9"
)
DATASET_QRELS = DATASET_ROOT / "qrels.jsonl"
DATASET_RECOVERIES = DATASET_ROOT / "evidence_recoveries/part-00000.jsonl"
DATASET_SPLITS = DATASET_ROOT / "splits.json"
DATASET_MANIFEST = DATASET_ROOT / "dataset_manifest.json"
DATASET_QUERY_TABLES = DATASET_ROOT / "query_tables/part-00000.jsonl"

R27 = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact"
R27_REPLAY = R27 / "historical_replay"
R27_EVAL = R27_REPLAY / "own_evaluation"
STUDENT_BASE = R27_REPLAY / "C2/seed13/checkpoints/step_000178.pt"

TARGET_INDEX = R27_EVAL / "indexes/H-C2-step000178"
BASE_RANKINGS = R27_EVAL / "rankings/H-C2-step000178/rankings.jsonl.gz"
BASE_TEACHER_RANKINGS = R27_EVAL / "teacher/H-C2-step000178/rankings.jsonl.gz"

SUPERVISION = ROOT / "work/stage1_optimization_r12_20260908/taskA_correctness/supervision"
TRAIN_WITNESS = SUPERVISION / "target_lists.train_fit.jsonl"
DEV_WITNESS = SUPERVISION / "target_lists.dev.jsonl"
EDGE_LISTS_TRAIN = SUPERVISION / "edge_lists.train_fit.jsonl"
SUPERVISION_MANIFEST = SUPERVISION / "manifest.json"
SPLIT_PROTOCOL = ROOT / "work/stage1_optimization_r10_20260907/taskA_protocol/splits.json"

FEATURES = ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"
FEATURE_MANIFEST = FEATURES / "manifest.jsonl"
TEACHER_MANIFEST = FEATURES / "teacher_manifest.jsonl"
TEACHER_OBJECTS = FEATURES / "teacher_objects"
CONTENT_KEYS = ROOT / "work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl"

QCET_OUT = ROOT / "work/stage1_query_conditioned_et_20260916"
QCET_VECTOR_CACHE = QCET_OUT / "vector_cache.pt"

TEACHER_PARENT = ROOT / "work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt"

QWEN_MODEL_DIR = ROOT / "hf_models/Qwen3-VL-Embedding-8B"

#: SHA256 values verified on this server during S0 (section 2.1 requires the
#: real hash, not the hash recorded in an old report).  A mismatch blocks.
EXPECTED_SHA256: dict[str, str] = {
    str(STUDENT_BASE): "a04ce39684a4ef0a9d1c797142e6251a41284117d3c55bcfd1c4ff183ccb81bd",
    str(TEACHER_PARENT): "ab0e3c3f85f006d2fdc4ba5194a0021680ab8fa1341441cb8eb003410ded68cc",
    str(DATASET_QRELS): "4d444170927aa0ab0c834c468ec96fe0e742f21d5bae5762bcb76b98bf762086",
    str(DATASET_RECOVERIES): "e8a1d99f9faef416c7b28b2462a153f15b19748f234104e28713cfcbc6be8a75",
    str(DATASET_SPLITS): "ade6e478206de8cb3e57c65c23e1c4af6e22179dfdf4dd88538d22f105bfd144",
    str(DATASET_QUERY_TABLES): "2a960ab307f4ae1c4c4bedf282d769e7930787a123d351beabd6c289a1ab3082",
    str(SPLIT_PROTOCOL): "a911572dd9de42908d02ab53989a4da60f4db2cbda8f0ec885ab55a8fb58710c",
    str(TRAIN_WITNESS): "0db7162748142ada692256f20eae1f51b4ddeb82800ddbad9da1bbb1694577b7",
    str(DEV_WITNESS): "ed3a221184fe96a15ba8f9edaf9159a702771119df08fd8ab8476200af5fbc52",
    str(FEATURE_MANIFEST): "c32099430feca4dae5d2f8fbbae60f965e3b0353fdd62c62a7bc929ba24216e1",
    str(TEACHER_MANIFEST): "0ebc11dc86f22514a5691b76b542b8afdd05ecbf957f4d3bbfd7e0f9a1db5c0e",
    str(CONTENT_KEYS): "d9558678c2eba75f31a25f7f865466e8c83ca08073ae772f278a0db865ebf5e5",
    str(TARGET_INDEX / "manifest.json"): "70d3180068acd99ae46ec71478e66bd2e3fb77c653baa28c14df0d6a93281f1a",
    str(TARGET_INDEX / "table_ids.json"): "94dd508230731015bed78b272dc3d46f4a903ea1fd1fe1c7ff3dfd759e7157fa",
    str(TARGET_INDEX / "table.hnsw"): "4dc2f6f548fba7a6521504c86b11681ebc0279d9119894f1cb88d4f511dd2766",
    str(BASE_RANKINGS): "19ea441ceb078d043723b785de8f74f245621b484311e41409e4874f492ae538",
}

#: Production source modules the frozen semantics are read from.  The three
#: marked ``spec_snapshot`` hashes are the ones recorded in
#: SOURCES_AND_BOUNDARIES.md section S4 and matched byte-for-byte in S0.
SOURCE_HASHES: dict[str, str] = {
    "src/mmdd_stage1/models.py": "2f59151b3c7194b14e34a4db31a2d0cdef55ee77e62ce997d2e1983f8beffffd",
    "src/run_stage1_r19.py": "a8439137b587f945ffe2ea9ddd24d9c75d4777eeeacdd96ddf6f4052f7635822",
    "src/mmdd_stage1/b13_recipe.py": "00688771844152107aa3fe0468e9133c4bfee29271e1f2fe37d3f571d0b16454",
    "src/mmdd_stage1/checkpoints.py": "b7861f7a1193f6b789cbbe8e76aa34d183ec0e53a63c155fe3c780bf5e3b81ed",
    "src/mmdd_stage1/features.py": "6cfdc76db6638b20b881d7536f9d8d12ca39fa74fa6193acf6cfd6796e3bba14",
    "src/mmdd_stage1/r26_metrics.py": "7f9a6b8802d9dfbf88b4f566e4edabe3603154a2fc34e682827e2399b28978e8",
    "src/mmdd_stage1/row_support.py": "24e4c48d7574e91f1a2c70420335a3ec07bca6af37b01d3f9a0e27e6969bea06",
    "src/mmdd_stage1/retrieval.py": "c7699064a0c79abc227da0c45e49624b98104f825d5a341f76fb0b65ec7d5043",
    "src/run_stage1_r11_task_e.py": "a6df137d10c0fa9a2212769e36761ec4e102e1006e237646e2f19f593442e181",
    "src/evaluate_stage1_r26.py": "21ee1c763916f950af0c0169b24b24db97fbcab9e81f3187f1ccd6071cb90a70",
}

STAGES = (
    "resolve-inputs",
    "build-labels",
    "test-contracts",
    "prepare-a",
    "train-a",
    "evaluate-a",
    "decide-a",
    "prepare-b",
    "train-b",
    "evaluate-b",
    "decide-b",
    "repeat-seed29",
    "summarize",
    "pack",
)


def load_protocol() -> dict[str, Any]:
    """Read the machine-readable protocol copy that ships with the spec."""
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def file_sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    """File identity record; hashing is optional for very large artifacts."""
    path = Path(path)
    entry: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not entry["exists"]:
        return entry
    entry["bytes"] = path.stat().st_size
    if hash_file:
        entry["sha256"] = file_sha256(path)
    return entry


def run_path(*parts: str) -> Path:
    return RUN.joinpath(*parts)


def namespace_seed(namespace: str) -> int:
    """Section 18: one SHA256 per sampling namespace, first 8 bytes big-endian."""
    return int.from_bytes(hashlib.sha256(namespace.encode("utf-8")).digest()[:8], "big")
