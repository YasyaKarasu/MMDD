"""Archive inspectable R26 evidence, with verified references for large binaries.

This command records the experiment's current status; building an archive does
not complete the experiment. Run after final review for the final deliverable.
It never imports experiment/client modules. Tensor schema inspection imports
PyTorch on demand and loads tensors onto the metadata-only device.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile


BINARY_FORMATS = {
    ".pt": "PyTorch serialized checkpoint/tensors; companion receipts describe producer and identity",
    ".hnsw": "hnswlib saved index; companion metadata describes IDs, metric and dimension",
    ".sqlite": "SQLite database; schema is recorded from sqlite_master",
}


def safe_path(path: Path) -> Path:
    """Reject protected names and symbolic links before opening any content."""
    if ".env.openai" in path.parts or path.is_symlink():
        raise ValueError("Protected file or symbolic link cannot be packaged")
    resolved = path.resolve()
    if ".env.openai" in resolved.parts:
        raise ValueError("Protected file cannot be packaged")
    return resolved


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with safe_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def source_closure(root: Path) -> list[Path]:
    """Include R26 entrypoints/tests and their statically imported local helpers."""
    src = root / "src"
    pending = [*src.rglob("*r26*.py"), *root.glob("tests/test_stage1_r25.py"),
               *root.glob("tests/test_stage1_retrieval_aligned.py"), *root.glob("tests/*r26*.py")]
    found = set()
    while pending:
        path = safe_path(pending.pop())
        if path in found:
            continue
        found.add(path)
        tree = ast.parse(path.read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = src if not node.level else path.parent.joinpath(*([".."] * (node.level - 1)))
                module = base.joinpath(*(node.module or "").split("."))
                pending.extend(p for p in (module.with_suffix(".py"), module / "__init__.py") if p.is_file())
                pending.extend(p for a in node.names for p in (module / (a.name + ".py"), module / a.name / "__init__.py") if p.is_file())
        for name in names:
            module = src.joinpath(*name.split("."))
            pending.extend(p for p in (module.with_suffix(".py"), module / "__init__.py") if p.is_file())
    return sorted(found)


class HashReader:
    """Hash the exact bytes sent to tar, rather than a separate earlier read."""

    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        value = self.stream.read(size)
        self.hash.update(value)
        return value


def external_record(path: Path) -> dict:
    path = safe_path(path)
    before = path.stat()
    record = {"path": str(path), "bytes": before.st_size, "sha256": digest(path),
              "format": BINARY_FORMATS[path.suffix],
              "companion_metadata": [str(p) for p in sorted(path.parent.glob("*.json"))]}
    if path.suffix == ".sqlite":
        import sqlite3
        try:
            with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
                record["schema"] = connection.execute(
                    "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name").fetchall()
        except sqlite3.DatabaseError:
            # This exact abandoned backup is intentionally retained as failure
            # evidence. Active caches must still have a readable SQLite schema.
            if path.parts[-2:] != ("feedback","priority_backup_attempt1_incomplete.sqlite"):
                raise
            record.update({"schema":None,"artifact_status":"abandoned_incomplete_backup",
                           "format":"Unreadable partial SQLite backup; raw bytes/hash preserved, never used as a live score cache"})
    elif path.suffix == ".pt":
        import torch

        def schema(value):
            if isinstance(value, torch.Tensor):
                return {"shape": list(value.shape), "dtype": str(value.dtype)}
            if isinstance(value, dict):
                return {str(k): schema(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                # Homogeneous lists retain count and schema without repeating
                # large ID/config sequences; heterogeneous state remains explicit.
                items = [schema(v) for v in value]
                return {"type": type(value).__name__, "length": len(items),
                        "items": items[:1] if items and all(v == items[0] for v in items) else items}
            return {"type": type(value).__name__}

        record["schema"] = schema(torch.load(path, map_location="meta", weights_only=True))
    if path.stat().st_size != before.st_size or path.stat().st_mtime_ns != before.st_mtime_ns:
        raise ValueError(f"Binary changed during packaging: {path}")
    return record


def verify_package(path: Path) -> dict:
    """Read every archived payload back and compare its bytes with the manifest."""
    observed, manifest = {}, None
    with tarfile.open(safe_path(path), "r|gz") as archive:
        for member in archive:
            if not member.isfile() or member.name in observed:
                raise ValueError("Archive has a non-file or duplicate member")
            stream = archive.extractfile(member)
            if member.name == "PACKAGE_MANIFEST.json":
                if manifest is not None:
                    raise ValueError("Duplicate package manifest")
                manifest = json.load(stream)
                continue
            checksum = hashlib.sha256()
            count = 0
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                count += len(chunk)
                checksum.update(chunk)
            observed[member.name] = (count,checksum.hexdigest())
    if manifest is None:
        raise ValueError("Package manifest missing")
    expected = {r["archive_path"]:(r["bytes"],r["sha256"]) for r in manifest["included"]}
    if len(expected) != len(manifest["included"]) or observed != expected:
        raise ValueError("Actual archive payload differs from its manifest")
    return {"verified_payload_files":len(observed),"verified_uncompressed_bytes":sum(v[0] for v in observed.values()),
            "scope":"Every archived payload read back and hashed; external binaries are referenced, not included."}


def build_package(root: Path, out: Path, destination: Path) -> dict:
    """Save every raw evidence file; never externalize rankings/lists/generations."""
    root, out = safe_path(root), safe_path(out)
    destination = destination.absolute()
    safe_path(destination)
    if destination.is_relative_to(out):
        raise ValueError("Archive must be outside the experiment tree")
    receipt_path = destination.with_name(destination.name + ".receipt.json")
    if destination.exists() or receipt_path.exists():
        raise ValueError("Choose a new archive name; preserve previous deliverables")
    matrix = json.loads((out / "EXECUTION_MATRIX.json").read_text())
    sources = source_closure(root)
    payload = [(p, "experiment/" + str(p.relative_to(out))) for p in sorted(out.rglob("*"))
               if p.name != ".env.openai" and (p.is_file() or p.is_symlink())]
    payload.extend((p, "repository/" + str(p.relative_to(root))) for p in sources)
    for relative in ("AGENTS.md", "方案.md", "mmdd_r25_review/R26_EXPERIMENT_PLAN.md"):
        path = root / relative
        if path.is_file():
            payload.append((path, "repository/" + relative))
    # Preserve external scientific inputs (graph, closure registry, native cache,
    # feature manifest) as inspectable data, not just their summary receipts.
    inputs_path = out / "RESOLVED_INPUTS.json"
    if inputs_path.exists():
        for role, record in json.loads(inputs_path.read_text()).items():
            if not record.get("exists"):
                continue
            path = safe_path(Path(record["path"]))
            if not path.is_relative_to(out):
                if digest(path) != record["sha256"]:
                    raise ValueError(f"Resolved scientific input changed: {role}")
                payload.append((path, f"resolved_inputs/{role}/{path.name}"))
    # Historical model endpoints are outside OUT; include actual binary references.
    inventory_path = out / "MODEL_INVENTORY.json"
    historical_training = set()
    if inventory_path.exists():
        for row in json.loads(inventory_path.read_text()):
            if row.get("checkpoint"):
                path = safe_path(Path(row["checkpoint"]))
                if not path.is_relative_to(out):
                    payload.append((path, "external_checkpoints/" + row["generator_id"]))
                    if path.parent.name == "checkpoints":
                        historical_training.add(path.parent.parent)
    for directory in sorted(historical_training):
        # Original histories/failed receipts are part of the provenance of reused
        # endpoints. Other checkpoints remain individually referenced above.
        for path in sorted(directory.iterdir()):
            if path.name != ".env.openai" and path.is_file() and path.suffix not in BINARY_FORMATS:
                payload.append((path, "historical_training/" + str(path.relative_to(root))))
    # Reused C1 and Teacher inputs must retain their actual candidate lists,
    # not just the histories/hashes of consuming those lists.
    reused_inputs = [
        "work/stage1_optimization_r25_final_20260914/common/c1_selective_hard_seed13.jsonl.gz",
        "work/stage1_optimization_r25_final_20260914/common/c1_selective_hard_seed29.jsonl.gz",
        "work/stage1_optimization_r11_20260908/taskC_clean/teacher/teacher_edge.pt",
        "work/stage1_optimization_r22_20260911/manifests/full_natural.jsonl",
        "work/stage1_optimization_r22_20260911/fresh_lineage/S0_mining/seed13/hard_negatives.jsonl.gz",
    ]
    for relative in reused_inputs:
        path = root / relative
        if path.is_file():
            payload.append((path,"reused_inputs/"+relative))
    manifest = {"experiment_execution_status": matrix["execution_status"],
                "experiment_scientific_validity": matrix["scientific_validity"],
                "completion_note": "Archive creation does not certify experimental completion.",
                "included": [], "external_binaries": [],
                "external_scope": "Local experiment binaries, resolved binary inputs, and inventory checkpoints. external_inputs/EXTERNAL_INPUTS_RECEIPT.json and feature_files.jsonl.gz separately record individual frozen-feature/backbone hashes, full backbone headers, and actual small feature-schema samples when present."}
    seen_external = set()
    with tarfile.open(destination, "x:gz", compresslevel=1, dereference=True) as archive:
        for path, name in payload:
            path = safe_path(path)
            if path.suffix in BINARY_FORMATS and not path.is_relative_to(out / "external_inputs/tensor_samples"):
                if path not in seen_external:
                    manifest["external_binaries"].append(external_record(path))
                    seen_external.add(path)
                continue
            before = path.stat()
            info = archive.gettarinfo(str(path), arcname=name)
            with path.open("rb") as stream:
                reader = HashReader(stream)
                archive.addfile(info, reader)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                raise ValueError(f"Evidence changed during packaging: {path}")
            manifest["included"].append({"archive_path": name, "source_path": str(path),
                                         "bytes": info.size, "sha256": reader.hash.hexdigest()})
        content = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
        info = tarfile.TarInfo("PACKAGE_MANIFEST.json")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    verification = verify_package(destination)
    receipt = {"execution_status": "archive_created", "experiment_status": matrix["execution_status"],
               "archive": {"path": str(destination), "bytes": destination.stat().st_size, "sha256": digest(destination)},
               "manifest_sha256": hashlib.sha256(content).hexdigest(),
               "included_files": len(manifest["included"]), "referenced_binaries": len(seen_external),
               "readback_verification":verification}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true", help="Read back an existing archive without modifying it")
    args = parser.parse_args()
    print(json.dumps(verify_package(args.destination) if args.verify_only else
                     build_package(args.root, args.out or args.root / "work/stage1_optimization_r26_20260914", args.destination)))
