"""Build and verify the requested R27 review archive below 300 decimal MB."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact"
META = ROOT / "work/r27_review_package_20260915"
ARCHIVE = ROOT / "R27_results_review_under300MB_20260915.tar.gz"
LIMIT = 300_000_000


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def package() -> dict:
    META.mkdir(parents=True, exist_ok=True)
    original_manifest = RUN / "PACKAGE_MANIFEST.json"
    original = json.loads(original_manifest.read_text())
    included, omitted = [], []
    priority = {
        "score_handoff/B13/raw_candidates_paths.jsonl.gz",
        "score_handoff/B13/teacher_scores_refs.jsonl.gz",
        "historical_replay/consumed_feature_files.jsonl.gz",
    }
    for rec in original["included_files"]:
        rel = rec["relative_path"]
        keep = (rec["bytes"] <= 2_000_000
                or rel.endswith(("/A0_rankings.jsonl.gz", "/A1_rankings.jsonl.gz"))
                or rel in priority)
        if keep:
            assert sha(RUN / rel) == rec["sha256"], rel
            name = "FULL_PACKAGE_OMISSIONS.json" if rel == "OMITTED_SERVER_FILES.json" else rel
            included.append({**rec, "relative_path": name, "source_path": str(RUN / rel)})
        else:
            omitted.append({**rec, "reason": "omitted to meet the requested 300 MB review-package limit; original retained on server"})

    readme = META / "READ_ME_FIRST.md"
    readme.write_text("""# R27：300MB 内审阅包

这是按用户要求缩减的审阅包，不是原 2.35GB 全量结果包。**本文及本包清单优先解释打包范围。** RESULTS.md、DELIVERY_NOTES.md 等原报告保留原文；其中“全部原始数据随包交付”的表述针对原包，不适用于此包。

保留：全部报告、统计、逐 query 指标与对照、测试及审计回执；九个模型的 A0/A1 完整排名；B13 原始候选/路径 scalar 与 T0 分数；H 训练逐步数值 trace、checkpoint/parent/parity 回执、消费特征文件 SHA；B 全部病例、生成回复、truth、图像裁剪和删除干预；源码快照。

省略：H 大型实际 batch/order 内容及完整原始检索/Teacher 排名；其他八个模型的原始候选/路径和 T0 字典；大型逐行 support 向量及部分索引 ID 表。H 的状态匹配可审阅回执，全部训练消费内容及所有模型原始分数的独立重算仍需服务器原文件。没有重新采样病例、修改指标或重跑实验。

PACKAGE_MANIFEST.json 是此包的实际文件清单及 SHA；OMITTED_SERVER_FILES.json 列出此次新增省略项。FULL_PACKAGE_MANIFEST.json 与 FULL_PACKAGE_OMISSIONS.json 保存原包范围及原先省略的大权重/索引清单。所有原始文件及旧包均保留。
""")
    write_json(META / "OMITTED_SERVER_FILES.json", {
        "additional_omissions": omitted,
        "original_large_file_omissions": "FULL_PACKAGE_OMISSIONS.json",
        "original_archive": str(ROOT / "R27_results_compact_20260915.tar.gz"),
    })
    for path, name in (
        (readme, readme.name),
        (META / "OMITTED_SERVER_FILES.json", "OMITTED_SERVER_FILES.json"),
        (original_manifest, "FULL_PACKAGE_MANIFEST.json"),
        (Path(__file__).resolve(), "source_snapshot/src/package_stage1_r27_review.py"),
    ):
        included.append({"relative_path": name, "source_path": str(path),
                         "bytes": path.stat().st_size, "sha256": sha(path)})
    manifest = META / "PACKAGE_MANIFEST.json"
    write_json(manifest, {
        "format_version": 1, "scope": "review_under_300_decimal_MB",
        "maximum_archive_bytes": LIMIT, "included_files": included,
        "included_bytes": sum(r["bytes"] for r in included),
        "additional_omitted_files": len(omitted),
        "original_manifest_sha256": sha(original_manifest),
    })
    assert sum(r["bytes"] for r in included) + manifest.stat().st_size < LIMIT
    expected = {"R27/" + r["relative_path"]: r["sha256"] for r in included}
    expected["R27/PACKAGE_MANIFEST.json"] = sha(manifest)
    assert len(expected) == len(included) + 1
    with tarfile.open(ARCHIVE, "w:gz", compresslevel=6) as archive:
        for rec in included:
            archive.add(rec["source_path"], arcname="R27/" + rec["relative_path"], recursive=False)
        archive.add(manifest, arcname="R27/PACKAGE_MANIFEST.json", recursive=False)
    assert ARCHIVE.stat().st_size < LIMIT
    verified = set()
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive:
            assert member.isfile() and member.name in expected and member.name not in verified
            digest = hashlib.sha256()
            with archive.extractfile(member) as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            assert digest.hexdigest() == expected[member.name], member.name
            verified.add(member.name)
    assert verified == set(expected)
    receipt = {"status": "verified", "archive": str(ARCHIVE),
               "bytes": ARCHIVE.stat().st_size, "sha256": sha(ARCHIVE),
               "maximum_bytes": LIMIT, "verified_archive_members": len(verified),
               "additional_omitted_files": len(omitted)}
    write_json(ROOT / "R27_results_review_under300MB_20260915.DELIVERY.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(package()))
