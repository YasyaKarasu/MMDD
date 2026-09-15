"""Exercise evidence preservation without any model, GPU, or client imports."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from package_stage1_r26 import build_package, verify_package


def fixture_tree(tmp_path):
    root = tmp_path / "repo"
    out = root / "work/r26"
    out.mkdir(parents=True)
    (root / "src").mkdir()
    (out / "EXECUTION_MATRIX.json").write_text(json.dumps({"execution_status": "in_progress", "scientific_validity": "partial"}))
    return root, out


def test_package_retains_actual_evidence_and_does_not_claim_completion(tmp_path):
    root, out = fixture_tree(tmp_path)
    files = {"rankings/ranks.jsonl.gz": b"actual compressed rank bytes",
             "feedback/lists.jsonl": b'{"candidate_ids":["a","b"]}\n',
             "stage2/generation.json": b'{"raw_completion":"value"}'}
    for name, content in files.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    torch.save({"projection": torch.zeros(2, 3)}, out / "checkpoint.pt")
    sample = out / "external_inputs/tensor_samples/example.pt"
    sample.parent.mkdir(parents=True)
    torch.save({"feature":torch.ones(3)},sample)
    archive = tmp_path / "evidence.tar.gz"
    receipt = build_package(root, out, archive)
    assert receipt["experiment_status"] == "in_progress"
    assert receipt["referenced_binaries"] == 1
    assert receipt["readback_verification"]["verified_payload_files"] == receipt["included_files"]
    with tarfile.open(archive) as stream:
        manifest = json.load(stream.extractfile("PACKAGE_MANIFEST.json"))
        for name, content in files.items():
            assert stream.extractfile("experiment/" + name).read() == content
            record = next(r for r in manifest["included"] if r["archive_path"] == "experiment/" + name)
            assert record["sha256"] == hashlib.sha256(content).hexdigest()
        assert "experiment/checkpoint.pt" not in stream.getnames()
        assert stream.extractfile("experiment/external_inputs/tensor_samples/example.pt").read() == sample.read_bytes()
        binary = manifest["external_binaries"][0]
        assert binary["sha256"] == hashlib.sha256((out / "checkpoint.pt").read_bytes()).hexdigest()
        assert binary["schema"]["projection"] == {"shape": [2, 3], "dtype": "torch.float32"}
    with pytest.raises(ValueError, match="new archive name"):
        build_package(root, out, archive)


def test_package_rejects_symbolic_links_without_reading_target(tmp_path):
    root, out = fixture_tree(tmp_path)
    (out / "pretend_rank.json").symlink_to(tmp_path / "nonexistent-secret")
    with pytest.raises(ValueError, match="symbolic link"):
        build_package(root, out, tmp_path / "rejected.tar.gz")


def test_package_verifies_external_input_before_archiving(tmp_path):
    root, out = fixture_tree(tmp_path)
    graph = root / "graph.jsonl"
    graph.write_bytes(b"changed")
    (out / "RESOLVED_INPUTS.json").write_text(json.dumps({"graph": {
        "path": str(graph), "exists": True, "sha256": hashlib.sha256(b"original").hexdigest()}}))
    archive = tmp_path / "rejected.tar.gz"
    with pytest.raises(ValueError, match="input changed"):
        build_package(root, out, archive)
    assert not archive.exists()


def test_package_includes_transitive_local_source(tmp_path):
    root, out = fixture_tree(tmp_path)
    (root / "src/run_r26.py").write_text("from local_helper import function\n")
    (root / "src/local_helper.py").write_text("def function(): return 1\n")
    archive = tmp_path / "sources.tar.gz"
    build_package(root, out, archive)
    with tarfile.open(archive) as stream:
        assert "repository/src/local_helper.py" in stream.getnames()


def test_package_preserves_reused_training_history_and_database_schema(tmp_path):
    import sqlite3

    root, out = fixture_tree(tmp_path)
    training = root / "old_training/seed13"
    (training / "checkpoints").mkdir(parents=True)
    checkpoint = training / "checkpoints/step_000178.pt"
    torch.save({"weight": torch.ones(1)}, checkpoint)
    history = training / "train_history.jsonl"
    history.write_text('{"step":178,"loss":0.2}\n')
    (out / "MODEL_INVENTORY.json").write_text(json.dumps([
        {"generator_id": "old/seed13", "checkpoint": str(checkpoint)}]))
    with sqlite3.connect(out / "pairs.sqlite") as connection:
        connection.execute("CREATE TABLE scores (query_id TEXT, target_id TEXT, score REAL)")
    archive = tmp_path / "historical.tar.gz"
    build_package(root, out, archive)
    with tarfile.open(archive) as stream:
        assert stream.extractfile("historical_training/old_training/seed13/train_history.jsonl").read() == history.read_bytes()
        manifest = json.load(stream.extractfile("PACKAGE_MANIFEST.json"))
        database = next(r for r in manifest["external_binaries"] if r["path"].endswith("pairs.sqlite"))
        assert database["schema"][0][1] == "scores"


def test_package_readback_rejects_payload_manifest_mismatch(tmp_path):
    import io

    archive = tmp_path / "wrong.tar.gz"
    manifest = {"included":[{"archive_path":"ranks.jsonl","bytes":3,
                              "sha256":hashlib.sha256(b"old").hexdigest()}]}
    with tarfile.open(archive,"w:gz") as stream:
        for name,body in [("ranks.jsonl",b"new"),("PACKAGE_MANIFEST.json",json.dumps(manifest).encode())]:
            info = tarfile.TarInfo(name)
            info.size = len(body)
            stream.addfile(info,io.BytesIO(body))
    with pytest.raises(ValueError,match="differs from its manifest"):
        verify_package(archive)


def test_package_preserves_failed_backup_but_rejects_unreadable_active_cache(tmp_path):
    import sqlite3

    root,out = fixture_tree(tmp_path)
    backup = out / "feedback/priority_backup_attempt1_incomplete.sqlite"
    backup.parent.mkdir()
    backup.write_bytes(b"partial backup bytes")
    archive = tmp_path / "failed_evidence.tar.gz"
    build_package(root,out,archive)
    with tarfile.open(archive) as stream:
        manifest = json.load(stream.extractfile("PACKAGE_MANIFEST.json"))
        record = manifest["external_binaries"][0]
        assert record["artifact_status"] == "abandoned_incomplete_backup"
        assert record["schema"] is None
        assert record["sha256"] == hashlib.sha256(backup.read_bytes()).hexdigest()
    active = out / "teacher/T0_pairs.sqlite"
    active.parent.mkdir()
    active.write_bytes(b"bad active cache")
    with pytest.raises(sqlite3.DatabaseError):
        build_package(root,out,tmp_path / "must_reject.tar.gz")
