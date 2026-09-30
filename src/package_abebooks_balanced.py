"""Package AbeBooks results, data and source under a strict 300 MB limit."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import posixpath
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 300_000_000
NAME = "abebooks_no4_balanced_20260930_bundle"


def digest(path: Path) -> str:
    if ".env.openai" in path.parts or ".env.openai" in path.resolve().parts:
        raise ValueError("Protected file cannot be hashed")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def files(root: Path):
    """Walk only explicit input trees, excluding secrets before inspecting files."""
    for base, directories, names in os.walk(root, followlinks=False):
        directories[:] = sorted(d for d in directories if d not in
            {"__pycache__", ".pytest_cache", ".git", ".env.openai"}
            and not (Path(base) / d).is_symlink())
        for name in sorted(names):
            if name == ".env.openai" or name.endswith(".pyc"):
                continue
            path = Path(base) / name
            if not path.is_symlink():
                yield path


def build(run: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{NAME}.tar.xz"
    if archive.exists():
        raise FileExistsError(archive)
    selected, excluded, links = {}, [], {}
    metadata = output / "package_metadata"
    metadata.mkdir(exist_ok=True)

    def add(path: Path, destination: str) -> None:
        if ".env.openai" in path.parts or ".env.openai" in Path(destination).parts:
            raise ValueError("Protected file cannot be packaged")
        if path.is_symlink():
            raise ValueError(f"Unexpected file symlink: {path}")
        selected[destination] = path

    def extra(name: str, value: object, *, text: bool = False) -> None:
        path = metadata / name
        path.write_text(str(value) if text else json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        add(path, name)

    prefix = f"work/{run.name}"
    for path in files(run):
        rel = path.relative_to(run)
        large_tensor = path.name.endswith(".tokens.npy")
        checkpoint = path.suffix == ".pt" and not (
            path.name == "teacher_c2_logits.pt" or rel.parts[-2:] == ("pca", "basis.pt"))
        if large_tensor or checkpoint:
            excluded.append({"path": f"{prefix}/{rel}", "bytes": path.stat().st_size,
                "reason": "large token feature tensor" if large_tensor else "checkpoint or per-object feature tensor"})
        else:
            add(path, f"{prefix}/{rel}")
    for schedule in ("one_epoch_b64", "ten_epochs_b64", "ten_epochs_b16"):
        for shared in ("dataset_view", "features", "encoder", "data"):
            links[f"{prefix}/{schedule}/{shared}"] = f"../{shared}"
    links["dataset/abebooks_joinability_no4_balanced_20260930"] = f"../{prefix}/dataset_view"

    previous = ROOT / "work/abebooks_data_ablation_20260930"
    for path in files(previous / "columns/dataset_view"):
        add(path, f"inputs/previous_columns_dataset/{path.relative_to(previous / 'columns/dataset_view')}")
    for path in previous.iterdir():
        if path.name != ".env.openai" and path.is_file() and path.suffix in {".json", ".md"}:
            add(path, f"background/previous_ablation/{path.name}")
    for arm in ("baseline", "hubs", "columns", "both"):
        for name in ("recall_summary.json", "recall_per_query.jsonl", "TRAINING_COMPLETE.json", "SELECTION_FREEZE.json"):
            path = previous / arm / name
            if path.is_file():
                add(path, f"background/previous_ablation/{arm}/{name}")
    for path in files(previous / "column_noise_audit"):
        add(path, f"background/column_noise_audit/{path.name}")

    for directory in ("src", "tests"):
        for path in files(ROOT / directory):
            add(path, str(path.relative_to(ROOT)))
    for name in ("AGENTS.md", "README.md", "requirements.txt", "pytest.ini", "方案.md"):
        path = ROOT / name
        if path.is_file():
            add(path, name if name != "README.md" else "PROJECT_README.md")
    audit = ROOT / "audit/MMDD_S1_V4_AUDIT_AND_V4_1_PACKAGE"
    for directory in ("tools", "tests"):
        for path in files(audit / directory):
            if path.suffix == ".py":
                add(path, str(path.relative_to(ROOT)))
    for path in (audit / "next_round/protocol.json", audit / "next_round/reference_contracts.py",
                 ROOT / "hf_models/Qwen3-VL-Embedding-8B/scripts/qwen3_vl_embedding.py",
                 ROOT / "scripts/run_v4_1_gpu0.sh"):
        if path.is_file():
            add(path, str(path.relative_to(ROOT)))

    assets = [json.loads(line) for line in (run / "dataset_view/bridge_assets/part-00000.jsonl").open()]
    image_map = {}
    for asset in assets:
        if asset["asset_type"] != "image":
            continue
        path = Path(asset["local_path"])
        relative = f"assets/images/{digest(path)}{path.suffix.lower()}"
        add(path, relative)
        image_map[asset["asset_id"]] = {"original_local_path": str(path), "package_path": relative}
    extra("ASSET_PATHS.json", image_map)
    extra("OMITTED_ARTIFACTS.json", {"items": excluded,
        "logical_bytes": sum(row["bytes"] for row in excluded),
        "note": "Size sums count hard-linked files by path; originals remain on the host. Pretrained backbone weights are also external prerequisites."})
    environment = subprocess.check_output([os.sys.executable, "-c",
        "import importlib.metadata,json,platform; print(json.dumps({'python':platform.python_version(),"
        "'packages':sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name'])},indent=2))"], text=True)
    extra("PYTHON_ENVIRONMENT.json", environment, text=True)
    patch = subprocess.check_output(["git", "diff", "--", "src", "tests"], cwd=ROOT, text=True)
    extra("TRACKED_CODE_CHANGES.patch", patch, text=True)
    complete = json.loads((run / "COMPLETE.json").read_text())
    for path, sha in complete["source_sha256"].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f"Executed source changed since experiment completion: {path}")

    extra("make_local_dataset.py", '''"""Copy a packaged dataset and rebase image references; preserve archived originals."""
import argparse
import json
import shutil
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--dataset", default="work/abebooks_no4_balanced_20260930/dataset_view")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parent
source = (root / a.dataset).resolve()
if not source.is_relative_to(root):
    raise ValueError("Dataset must be inside the package")
shutil.copytree(source, a.output)
mapping = json.loads((root / "ASSET_PATHS.json").read_text())
for path in (a.output / "bridge_assets").glob("*.jsonl"):
    rows = [json.loads(line) for line in path.open()]
    for row in rows:
        if row["asset_type"] == "image":
            image = (root / mapping[row["asset_id"]]["package_path"]).resolve()
            row["local_path"] = str(image)
            row["relative_path"] = str(image)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\\n" for row in rows))
print(a.output.resolve())
''', text=True)
    extra("README.zh-CN.md", f'''# AbeBooks 本轮实验交付包

范围：删除四列、删除相关 Query/Target、implicit/explicit 平衡与重新划分，以及 1 轮/b64、10 轮/b64、10 轮/b16 三组 Teacher → C1 → SUP/KD 实验。

优先阅读 `work/{run.name}/REPORT.zh-CN.md`。数据为 118 个 Query（59/59），train/dev/test=94/12/12，137 个 Target，120 个正例关系。

## 包内内容

- `work/{run.name}/`：本轮全部 JSON/JSONL/CSV/Markdown/日志，包含完整训练列表、候选池、逐 Query 排名、证据路径、Teacher logits 轨迹、模型选择与核验记录；不是仅抽取摘要。
- 同目录 `dataset_view/`：完整新数据集；`dataset/abebooks_joinability_no4_balanced_20260930` 为内部相对链接。
- 本轮原始 embedding（`features/z`）、PCA、query row embedding、C2 Teacher 蒸馏 logits 也保留。
- `assets/images/` 和 `ASSET_PATHS.json`：全部图片素材，按文件内容去重；文本素材完整保留在数据集的 bridge_assets 中。
- `inputs/previous_columns_dataset/`：本轮构建所用的上一轮删列版输入，可重建平衡数据。
- `src/`、`tests/`：当前完整源码与测试快照，包括所有新增且尚未提交的文件；`.patch` 只补充展示已有受 Git 跟踪文件的改动，不能替代源码快照。
- `background/`：前序四组数据消融和列干扰审计的报告、汇总及逐 Query 指标，提供上下文；前序完整训练张量和全部轨迹不在本包范围。
- `PACKAGE_MANIFEST.json`：每个文件的 SHA-256、大小、源路径；完全相同文件以 tar 内部硬链接去重，解压后仍保留预期路径。

## 为满足 300 MB 上限而省略

模型检查点（包含优化器状态）、逐对象 feature `.pt` 和大型 token feature 张量未放入；文件级清单与大小见 `OMITTED_ARTIFACTS.json`。预训练 Qwen3-VL-Embedding-8B 权重也需另行准备。原始训练目录里的这些文件没有删除。

因此本包适合完整审阅数据、指标、轨迹与代码，**不能不经重建特征/重新训练就加载省略的模型进行推理或断点续训**。历史报告里的 COMPLETE/缓存清单仍保留原始哈希和路径，可能引用上述省略文件；包内完整性以 PACKAGE_MANIFEST.json 为准。

## 解压与核验

```bash
tar -xJf {NAME}.tar.xz
cd {NAME}
python verify_package.py
```

`SHA256SUMS.txt` 提供常规 SHA-256 校验列表。运行训练应使用 MMDD 环境；原环境软件版本见 `PYTHON_ENVIRONMENT.json`，没有导出环境变量或密钥。

## 在另一台机器使用数据

为了保留原始数据和报告哈希，归档中的原始绝对路径未直接改写。以下命令创建图片路径适配当前机器的数据副本：

```bash
python make_local_dataset.py --output /absolute/path/abebooks_no4_local
python make_local_dataset.py --dataset inputs/previous_columns_dataset --output /absolute/path/previous_columns_local
```

重新构建当前数据的入口为 `src/build_abebooks_balanced.py --source ... --output ... --seed 13`。
重新训练按 `src/run_abebooks_fresh.py prepare-data`、`encode`、`src/run_abebooks_balanced.py run` 执行，使用一个新的 `--run-root`；需先将 backbone 放到包根目录 `hf_models/Qwen3-VL-Embedding-8B/`。
`src/summarize_abebooks_balanced.py` 还包含一个回顾性 Raw 对照，依赖前序实验的旧 embedding；该旧缓存未打包，对照结果和逐 Query gold 名次已完整保存在本轮 `COMPARISON.json`。

安全范围：未读取、复制、哈希或打包受保护的秘密配置文件；没有打包 `.git`、Python 缓存或整个用户环境。
''', text=True)
    extra("verify_package.py", '''"""Verify every packaged file and internal link against the package manifest."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text())
for row in manifest["files"]:
    path = root / row["path"]
    assert path.resolve().is_relative_to(root), row["path"]
    assert path.stat().st_size == row["bytes"], row["path"]
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    assert h.hexdigest() == row["sha256"], row["path"]
for name, target in manifest["links"].items():
    path = root / name
    assert path.is_symlink() and path.readlink().as_posix() == target, name
    assert path.resolve().is_relative_to(root) and path.exists(), name
print(f"PASS: {len(manifest['files'])} files and {len(manifest['links'])} internal links")
''', text=True)
    records = [{"path": name, "source_path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
               for name, path in selected.items()]
    # Keep similarly named gzip traces adjacent so xz can compress common streams.
    records.sort(key=lambda row: (Path(row["path"]).name, row["path"]))
    checksum_text = "".join(f"{r['sha256']}  {r['path']}\n" for r in records)
    extra("SHA256SUMS.txt", checksum_text, text=True)
    checksum_path = selected["SHA256SUMS.txt"]
    records.append({"path": "SHA256SUMS.txt", "source_path": str(checksum_path),
                    "bytes": checksum_path.stat().st_size, "sha256": digest(checksum_path)})
    manifest = {"schema": 1, "scope": "balanced no-four-column experiment with background reports",
                "limit_bytes": LIMIT, "files": records, "links": links,
                "image_assets": len(image_map), "unique_image_files": len({v["package_path"] for v in image_map.values()}),
                "excluded_files": len(excluded)}
    (metadata / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selected_files": len(records), "logical_bytes": sum(r["bytes"] for r in records),
                      "unique_content_bytes": sum({r['sha256']: r['bytes'] for r in records}.values()),
                      "excluded_files": len(excluded)}, ensure_ascii=False), flush=True)
    with archive.open("wb") as destination:
        compressor = subprocess.Popen(["xz", "-T1", "-6", "--lzma2=dict=64MiB", "-c"], stdin=subprocess.PIPE, stdout=destination)
        assert compressor.stdin is not None
        with tarfile.open(fileobj=compressor.stdin, mode="w|", format=tarfile.PAX_FORMAT) as tar:
            seen = {}
            for row in records:
                info = tarfile.TarInfo(f"{NAME}/{row['path']}")
                info.mode = 0o644
                sha = row["sha256"]
                if sha in seen:
                    info.type = tarfile.LNKTYPE
                    info.linkname = seen[sha]
                    tar.addfile(info)
                else:
                    seen[sha] = info.name
                    info.size = row["bytes"]
                    with Path(row["source_path"]).open("rb") as source:
                        tar.addfile(info, source)
            for name, target in links.items():
                resolved = posixpath.normpath(posixpath.join(NAME, posixpath.dirname(name), target))
                assert resolved.startswith(NAME + "/")
                info = tarfile.TarInfo(f"{NAME}/{name}")
                info.type, info.linkname, info.mode = tarfile.SYMTYPE, target, 0o755
                tar.addfile(info)
            payload = (metadata / "PACKAGE_MANIFEST.json").read_bytes()
            info = tarfile.TarInfo(f"{NAME}/PACKAGE_MANIFEST.json")
            info.size, info.mode = len(payload), 0o644
            tar.addfile(info, io.BytesIO(payload))
        compressor.stdin.close()
        if compressor.wait() != 0:
            raise RuntimeError("xz compression failed")
    size = archive.stat().st_size
    if size > LIMIT:
        raise ValueError(f"Archive exceeds 300 MB: {size} bytes")
    result = {"archive": str(archive), "bytes": size, "MB_decimal": size / 1e6,
              "sha256": digest(archive), "files": len(records), "links": len(links), "under_300_MB": True}
    (output / "PACKAGE_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / f"{archive.name}.sha256").write_text(f"{result['sha256']}  {archive.name}\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.run_root.resolve(), args.output_dir.resolve())
