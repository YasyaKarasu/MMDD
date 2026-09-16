"""Build verified, size-bounded R29 data/report and source archives."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "work/stage1_optimization_r29_candidate_vs_drift_20260915"
DELIVERY = ROOT / "deliveries/r29_20260915"
LIMIT = 300_000_000
TAG = "20260915"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(path: Path) -> Path:
    """Reject protected files and symlinks before packaging."""
    if ".env.openai" in path.parts or path.is_symlink():
        raise ValueError(f"protected or symbolic path cannot be packaged: {path}")
    resolved = path.resolve()
    if ".env.openai" in resolved.parts:
        raise ValueError(f"protected path cannot be packaged: {path}")
    return resolved


def copy_record(path: Path, archive_path: str) -> dict:
    path = safe(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    record = {
        "archive_path": archive_path,
        "source_path": str(path),
        "bytes": stat.st_size,
        "sha256": sha256(path),
    }
    if path.stat().st_size != stat.st_size or path.stat().st_mtime_ns != stat.st_mtime_ns:
        raise ValueError(f"file changed during hashing: {path}")
    return record


def add(records: list[dict], relative: str, archive_prefix: str = "experiment/") -> None:
    path = RUN / relative
    records.append(copy_record(path, archive_prefix + relative))


def selected_data() -> list[dict]:
    records: list[dict] = []
    for relative in (
        "RESULTS.md",
        "SCIENTIFIC_REVIEW.md",
        "NEXT_DECISION.md",
        "EXECUTION_LEDGER.json",
        "PACKAGE_MANIFEST.json",
        "RESOLVED_INPUTS.json",
        "R29_TEACHER_EVALUATION.json",
        "correctness/CORRECTNESS.json",
        "statistics/main_table.csv",
        "statistics/per_query.jsonl.gz",
        "statistics/source_group_bootstrap.csv",
        "diagnostics/teacher_score_status.json",
        "diagnostics/supervision_effective_weights.jsonl.gz",
        "diagnostics/supervision_summary.csv",
        "diagnostics/supervision_summary.json",
        "diagnostics/candidate_universe/CANDIDATE_DIAGNOSTIC.md",
        "diagnostics/candidate_universe/GATE.json",
        "diagnostics/candidate_universe/candidate_overlap.csv",
        "diagnostics/candidate_universe/hub_exposure.jsonl.gz",
        "diagnostics/candidate_universe/margin_violation_per_query.jsonl.gz",
        "diagnostics/candidate_universe/matched32.jsonl.gz",
        "diagnostics/candidate_universe/overlap_per_query.jsonl.gz",
        "diagnostics/candidate_universe/r12_lists.jsonl.gz",
        "diagnostics/candidate_universe/summary.json",
        "diagnostics/candidate_universe/t0train_lists.jsonl.gz",
        "teacher_evaluation/MODEL_INVENTORY.json",
        "teacher_evaluation/PROTOCOL.json",
        "teacher_evaluation/common/dev_queries.jsonl",
        "teacher_evaluation/teacher/CACHE_IDENTITY.json",
    ):
        add(records, relative)

    for arm in ("S-EDGE-FREEZE-P", "S-EDGE-FREEZE-R"):
        train = f"training/{arm}/seed13"
        for name in (
            "EXECUTION.json",
            "RESULTS.md",
            "RUN_IDENTITY.json",
            "active_relation_epoch_summary.json",
            "candidate_manifest.json",
            "history.jsonl",
            "order_hashes.json",
            "gradients/fixed_batch.json",
            "exact_rankings/metrics.json",
            "exact_rankings/rankings.jsonl.gz",
            "exact_rankings_epoch1/metrics.json",
            "exact_rankings_epoch1/rankings.jsonl.gz",
        ):
            add(records, f"{train}/{name}")
        add(records, f"training/{arm}-confirm/STATUS.json")

        ranking = f"teacher_evaluation/rankings/{arm}"
        for name in ("INDEX_RECEIPT.json", "RETRIEVAL_RECEIPT.json", "RUNNING.json", "metrics.json"):
            add(records, f"{ranking}/{name}")
        index = f"teacher_evaluation/indexes/{arm}"
        for name in ("image_ids.json", "table_ids.json", "text_ids.json", "manifest.json"):
            add(records, f"{index}/{name}")

        teacher = f"teacher_evaluation/teacher/{arm}"
        for name in ("TEACHER_RECEIPT.json", "latency_benchmarks.jsonl", "metrics.json", "rankings.jsonl.gz"):
            add(records, f"{teacher}/{name}")
    return records


def omitted_data() -> list[dict]:
    paths = [
        ("diagnostics/candidate_universe/natural_reservoir_top256.jsonl.gz", "791MB compressed reservoir; exceeds the package budget by itself"),
        ("teacher_evaluation/teacher/T0_pairs_cuda0.sqlite", "393MB Teacher pair cache; referenced by CACHE_IDENTITY and receipt"),
    ]
    result = []
    for relative, reason in paths:
        matches = sorted(RUN.glob(relative))
        if matches:
            result.extend({"relative_path": str(path.relative_to(RUN)), "bytes": path.stat().st_size, "reason": reason} for path in matches)
        else:
            result.append({"relative_path": relative, "bytes": None, "reason": reason})
    for arm in ("S-EDGE-FREEZE-P", "S-EDGE-FREEZE-R"):
        for name in ("image.hnsw", "text.hnsw", "table.hnsw"):
            path = RUN / "teacher_evaluation" / "indexes" / arm / name
            result.append({"relative_path": str(path.relative_to(RUN)), "bytes": path.stat().st_size if path.is_file() else None,
                           "reason": "large HNSW binary; index manifest and ID maps are included"})
        for relative, reason in (
            (f"teacher_evaluation/rankings/{arm}/rankings.jsonl.gz", "full Student retrieval ranking; metrics and receipts are included"),
        ):
            path = RUN / relative
            result.append({"relative_path": relative, "bytes": path.stat().st_size if path.is_file() else None, "reason": reason})
        for path in sorted((RUN / "training" / arm / "seed13" / "checkpoints").glob("*.pt")):
            result.append({"relative_path": str(path.relative_to(RUN)), "bytes": path.stat().st_size,
                           "reason": "multi-hundred-MB model checkpoint; run metadata and exact rankings are included"})
    result.append({"relative_path": "training_cpu_sandbox_retracted/**", "bytes": sum(p.stat().st_size for p in (RUN / "training_cpu_sandbox_retracted").rglob("*") if p.is_file()),
                   "reason": "retracted CPU reruns, not part of the final GPU evidence package"})
    return result


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n")


def archive(records: list[dict], destination: Path, manifest: dict, root_name: str) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    expected = {record["archive_path"]: (record["bytes"], record["sha256"]) for record in records}
    with tarfile.open(destination, "w:gz", compresslevel=6) as tar:
        for record in records:
            source = safe(Path(record["source_path"]))
            tar.add(source, arcname=f"{root_name}/{record['archive_path']}", recursive=False)
        info = tarfile.TarInfo(f"{root_name}/PACKAGE_MANIFEST.json")
        info.size = len(manifest_bytes)
        import io
        tar.addfile(info, io.BytesIO(manifest_bytes))

    observed: dict[str, tuple[int, str]] = {}
    with tarfile.open(destination, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                raise ValueError(f"archive contains non-file member: {member.name}")
            if member.name == f"{root_name}/PACKAGE_MANIFEST.json":
                continue
            relative = member.name.split("/", 1)[1]
            stream = tar.extractfile(member)
            digest = hashlib.sha256()
            count = 0
            assert stream is not None
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                count += len(block)
                digest.update(block)
            observed[relative] = (count, digest.hexdigest())
    if observed != expected:
        raise ValueError("archive readback differs from package manifest")
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "payload_files": len(records),
        "payload_bytes": sum(record["bytes"] for record in records),
        "limit_bytes": LIMIT,
        "under_limit": destination.stat().st_size < LIMIT,
        "readback_verified": True,
    }


def build() -> dict:
    if not RUN.is_dir():
        raise FileNotFoundError(RUN)
    DELIVERY.mkdir(parents=True, exist_ok=True)
    data_records = selected_data()
    omitted = omitted_data()
    readme = DELIVERY / "README.md"
    write_text(readme, f"""# MMDD R29 交付包（{TAG}）

本目录包含两个 gzip 压缩包：

- `MMDD_R29_experiment_data_reports_{TAG}.tar.gz`：R29 GPU 实验报告、统计、逐 query 诊断、训练回执、Teacher 评分结果和可复核的小型数据。
- `MMDD_src_R29_{TAG}.tar.gz`：仓库 `src/` 源代码快照（排除 `__pycache__` 和 `.pyc`）。

两个包都经过逐文件 SHA-256 回读校验；单包及两包合计均小于 300,000,000 bytes。为满足上限，超大 checkpoint、HNSW、Teacher SQLite、完整 Student 排名和自然候选 reservoir 没有复制进包，省略项及原始大小记录在数据包的 `OMITTED_ARTIFACTS.json` 中，原始文件仍保留在服务器实验目录。

`PACKAGE_VERIFICATION.json` 和 `SHA256SUMS` 是本目录的外部校验回执。""")
    omitted_path = DELIVERY / "OMITTED_ARTIFACTS.json"
    write_json(omitted_path, {"run": str(RUN), "omitted": omitted, "reason": "300MB delivery limit; original artifacts remain in place"})

    data_extra = [
        copy_record(readme, "README.md"),
        copy_record(omitted_path, "OMITTED_ARTIFACTS.json"),
    ]
    data_records.extend(data_extra)
    data_manifest = {
        "format_version": 1,
        "scope": "R29 experiment data and reports, size-bounded delivery",
        "source_run": str(RUN),
        "maximum_archive_bytes": LIMIT,
        "included_files": data_records,
        "omitted_artifacts": omitted,
    }
    data_archive = DELIVERY / f"MMDD_R29_experiment_data_reports_{TAG}.tar.gz"
    data_receipt = archive(data_records, data_archive, data_manifest, f"MMDD_R29_DATA_{TAG}")

    source_records: list[dict] = []
    for path in sorted((ROOT / "src").rglob("*")):
        if not path.is_file() or path.suffix == ".pyc" or "__pycache__" in path.parts:
            continue
        if ".env.openai" in path.parts:
            continue
        source_records.append(copy_record(path, f"src/{path.relative_to(ROOT / 'src')}"))
    source_scope = DELIVERY / "SOURCE_SCOPE.md"
    write_text(source_scope, """# Source snapshot

This archive contains the repository `src/` tree as plain source files. Python bytecode and `__pycache__` directories are excluded. The R29 entrypoints and packaging helper are included alongside the existing research utilities and packages.""")
    source_records.append(copy_record(source_scope, "SOURCE_SCOPE.md"))
    source_manifest = {
        "format_version": 1,
        "scope": "repository src snapshot",
        "source_root": str(ROOT / "src"),
        "included_files": source_records,
    }
    source_archive = DELIVERY / f"MMDD_src_R29_{TAG}.tar.gz"
    source_receipt = archive(source_records, source_archive, source_manifest, f"MMDD_SRC_R29_{TAG}")

    combined = data_receipt["bytes"] + source_receipt["bytes"]
    if data_receipt["bytes"] >= LIMIT or source_receipt["bytes"] >= LIMIT or combined >= LIMIT:
        raise ValueError(f"delivery exceeds 300MB: data={data_receipt['bytes']} source={source_receipt['bytes']} combined={combined}")
    verification = {
        "status": "verified",
        "maximum_bytes": LIMIT,
        "archives": [data_receipt, source_receipt],
        "combined_archive_bytes": combined,
        "combined_under_limit": True,
        "run": str(RUN),
    }
    write_json(DELIVERY / "PACKAGE_VERIFICATION.json", verification)
    (DELIVERY / "SHA256SUMS").write_text(
        f"{data_receipt['sha256']}  {Path(data_receipt['path']).name}\n"
        f"{source_receipt['sha256']}  {Path(source_receipt['path']).name}\n"
    )
    return verification


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
