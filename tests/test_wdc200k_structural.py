from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_old"))

import wdc200k_io as wdc200k_io_module  # noqa: E402
import wdc200k_structural as structural_module  # noqa: E402
from build_wdc_mm_joinability_dataset import (  # noqa: E402
    iter_wdc_rows,
    read_wdc_table,
)
from stage1_io import stable_hash  # noqa: E402
from wdc200k_selection import (  # noqa: E402
    ReserveManager,
    SelectionPolicy,
    TableCandidate,
)
from wdc200k_structural import (  # noqa: E402
    StructuralExpansionError,
    expand_selected_shard,
    finalize_validated_selection,
)


def test_streamed_source_table_guard_stops_mid_table_and_keeps_checkpoint(
    tmp_path: Path,
) -> None:
    enabled = False
    checks = 0

    def guard(_path: Path, estimated: int = 0) -> None:
        nonlocal checks
        if enabled and estimated:
            checks += 1
            if checks == 2:
                raise OSError("source reserve")

    checkpoint_writer = wdc200k_io_module.AtomicJsonlShard(
        tmp_path / "part-00000.jsonl",
        pre_write_guard=guard,
        guard_interval_bytes=100,
    )
    checkpoint_writer.write({"source_table_id": "committed"})
    checkpoint = checkpoint_writer.commit()
    enabled = True
    writer = structural_module._AtomicSourceTableShard(
        tmp_path / "part-00001.jsonl",
        pre_write_guard=guard,
        guard_interval_bytes=100,
    )
    source_table = {
        "source_table_id": "streamed",
        "source_file": "source.json.gz",
        "page_title": "Streamed",
        "caption": "",
        "section_title": "",
        "num_rows": 10,
        "num_cols": 1,
        "columns": [],
        "provenance_builder": "test",
        "metadata": {},
    }

    with pytest.raises(OSError, match="source reserve"):
        writer.write_source_table(
            source_table,
            (
                {"row_id": index, "values": ["x" * 30]}
                for index in range(10)
            ),
        )
    writer.abort()

    assert wdc200k_io_module.validate_completed_shard(checkpoint, tmp_path)
    assert not (tmp_path / "part-00001.jsonl").exists()


def write_wdc_gzip(
    root: Path,
    *,
    schema_class: str = "Product",
    host: str = "shop.test",
    rows: list[Any],
) -> Path:
    path = (
        root
        / schema_class
        / f"{schema_class}_{host}_October2023.json.gz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row)
            else:
                handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return path


def selection_record(
    path: Path,
    root: Path,
    *,
    rows: int,
    columns: int = 4,
    subset: str = "minimum3",
) -> dict[str, Any]:
    schema_class = path.parent.name
    return {
        "schema_class": schema_class,
        "subset": subset,
        "host": path.name.removeprefix(f"{schema_class}_").removesuffix(
            "_October2023.json.gz"
        ),
        "relative_path": path.relative_to(root).as_posix(),
        "rows": rows,
        "columns": columns,
        "rank": "rank",
        "selection_seed": 13,
    }


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def refresh_manifest_shard(
    manifest: dict[str, Any],
    *,
    relative_path: str,
    absolute_path: Path,
) -> None:
    shard = next(
        record
        for record in manifest["completed_shards"]
        if record["path"] == relative_path
    )
    content = absolute_path.read_bytes()
    shard["records"] = sum(
        1 for line in content.splitlines() if line.strip()
    )
    shard["bytes"] = len(content)
    shard["sha256"] = hashlib.sha256(content).hexdigest()


def candidate(
    path: Path,
    root: Path,
    *,
    rows: int,
    columns: int = 2,
    subset: str = "minimum3",
) -> TableCandidate:
    record = selection_record(
        path,
        root,
        rows=rows,
        columns=columns,
        subset=subset,
    )
    return TableCandidate(
        schema_class=str(record["schema_class"]),
        subset=str(record["subset"]),
        host=str(record["host"]),
        relative_path=str(record["relative_path"]),
        rows=int(record["rows"]),
        columns=int(record["columns"]),
    )


def stable_operation_key(shard_id: str, relative_path: str) -> str:
    return stable_hash(
        "wdc200k-structural-replacement",
        shard_id,
        relative_path,
        length=40,
    )


def invalid_json_reason(text: str) -> str:
    try:
        json.loads(text + "\n")
    except json.JSONDecodeError as error:
        return f"JSONDecodeError: {error}"
    raise AssertionError("fixture must be invalid JSON")


def test_structural_expansion_preserves_all_rows_and_removes_image(
    tmp_path: Path,
) -> None:
    table_path = write_wdc_gzip(
        tmp_path,
        rows=[
            {
                "row_id": index,
                "name": f"n{index}",
                "page_url": f"https://e.test/{index}",
                "image": f"/{index}.jpg",
            }
            for index in range(37)
        ],
    )

    result = expand_selected_shard(
        [selection_record(table_path, tmp_path, rows=37)],
        output_root=tmp_path / "structural",
        input_root=tmp_path,
    )
    assert read_manifest(result.manifest)["schema_version"] == (
        "wdc200k-structural-v2"
    )

    source = read_records(result.source_tables)[0]
    assert len(source["rows"]) == 37
    assert "image" not in {
        column["column_name"] for column in source["columns"]
    }
    assert all(
        "image" not in {cell["column_name"] for cell in row["cells"]}
        for row in source["rows"]
    )
    entities = read_records(result.entities)
    assert len(entities) == 37
    assert len(read_records(result.page_refs)) == 37
    assert len(read_records(result.direct_image_refs)) == 37

    baseline = read_wdc_table(
        table_path,
        tmp_path,
        min_rows=2,
        min_cols=2,
        max_rows=0,
    )
    assert baseline.source_table == source
    assert baseline.entities == entities


def test_structural_progress_reports_each_table_and_shard_commit(
    tmp_path: Path,
) -> None:
    table_path = write_wdc_gzip(
        tmp_path,
        rows=[
            {"name": "A", "page_url": "https://example.test/a"},
            {"name": "B", "page_url": "https://example.test/b"},
        ],
    )
    events: list[dict[str, Any]] = []

    result = expand_selected_shard(
        [selection_record(table_path, tmp_path, rows=2, columns=2)],
        output_root=tmp_path / "structural",
        input_root=tmp_path,
        progress_callback=events.append,
    )

    assert result.tables == 1
    assert [event["phase"] for event in events] == [
        "expand_table",
        "table_complete",
        "commit_structural_shard",
        "structural_shard_complete",
    ]
    assert [event["completed"] for event in events] == [0, 1, 1, 1]
    assert events[0]["relative_path"] == table_path.relative_to(
        tmp_path
    ).as_posix()
    assert events[1]["entities"] == 2


def test_structural_progress_reports_completed_shard_resume(
    tmp_path: Path,
) -> None:
    table_path = write_wdc_gzip(
        tmp_path,
        rows=[{"name": "A", "page_url": "https://example.test/a"}],
    )
    records = [selection_record(table_path, tmp_path, rows=1, columns=2)]
    output_root = tmp_path / "structural"
    expand_selected_shard(
        records,
        output_root=output_root,
        input_root=tmp_path,
    )
    events: list[dict[str, Any]] = []

    result = expand_selected_shard(
        records,
        output_root=output_root,
        input_root=tmp_path,
        progress_callback=events.append,
    )

    assert result.tables == 1
    assert [event["phase"] for event in events] == [
        "validate_completed_shard",
        "structural_shard_complete",
    ]
    assert [event["completed"] for event in events] == [0, 1]
    assert events[-1]["resumed"] is True


def test_iter_wdc_rows_has_no_implicit_row_limit(tmp_path: Path) -> None:
    path = write_wdc_gzip(
        tmp_path,
        rows=[{"name": f"n{index}"} for index in range(113)],
    )

    assert len(list(iter_wdc_rows(path))) == 113


def test_streamed_selection_requires_an_explicit_resume_fingerprint(
    tmp_path: Path,
) -> None:
    path = write_wdc_gzip(
        tmp_path,
        rows=[{"name": "A", "page_url": "https://example.test/a"}],
    )
    records = iter(
        [selection_record(path, tmp_path, rows=1, columns=2)]
    )

    with pytest.raises(ValueError, match="input_fingerprint"):
        expand_selected_shard(
            records,
            output_root=tmp_path / "structural",
            input_root=tmp_path,
        )


def test_selection_seed_changes_structural_resume_fingerprint(
    tmp_path: Path,
) -> None:
    path = write_wdc_gzip(
        tmp_path,
        rows=[{"name": "A", "page_url": "https://example.test/a"}],
    )
    record = selection_record(path, tmp_path, rows=1, columns=2)
    output_root = tmp_path / "structural"
    expand_selected_shard(
        [record],
        output_root=output_root,
        input_root=tmp_path,
        input_fingerprint="same-upstream-fingerprint",
    )

    changed = {**record, "selection_seed": 99}
    with pytest.raises(ValueError, match="fingerprint"):
        expand_selected_shard(
            [changed],
            output_root=output_root,
            input_root=tmp_path,
            input_fingerprint="same-upstream-fingerprint",
        )


def test_streamed_selection_seed_changes_resume_fingerprint(
    tmp_path: Path,
) -> None:
    path = write_wdc_gzip(
        tmp_path,
        rows=[{"name": "A", "page_url": "https://example.test/a"}],
    )
    record = selection_record(path, tmp_path, rows=1, columns=2)
    output_root = tmp_path / "structural"
    expand_selected_shard(
        iter([record]),
        output_root=output_root,
        input_root=tmp_path,
        input_fingerprint="same-upstream-fingerprint",
    )

    with pytest.raises(ValueError, match="fingerprint"):
        expand_selected_shard(
            iter([{**record, "selection_seed": 99}]),
            output_root=output_root,
            input_root=tmp_path,
            input_fingerprint="same-upstream-fingerprint",
        )


def test_streamed_selection_input_is_iterated_exactly_once(
    tmp_path: Path,
) -> None:
    path = write_wdc_gzip(
        tmp_path,
        rows=[{"name": "A", "page_url": "https://example.test/a"}],
    )
    record = selection_record(path, tmp_path, rows=1, columns=2)

    class OneShotSelection:
        def __init__(self) -> None:
            self.iterations = 0

        def __iter__(self) -> Any:
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("selection iterable was replayed")
            yield record

    selection = OneShotSelection()
    result = expand_selected_shard(
        selection,
        output_root=tmp_path / "structural",
        input_root=tmp_path,
        input_fingerprint="selection-shard-checksum",
    )

    assert result.tables == 1
    assert selection.iterations == 1


def test_structural_module_imports_through_scripts_package() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts_old.wdc200k_structural import expand_selected_shard",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_page_refs_include_only_usable_urls_and_failures_are_terminal(
    tmp_path: Path,
) -> None:
    path = write_wdc_gzip(
        tmp_path,
        rows=[
            {
                "name": "valid",
                "page_url": "HTTPS://Example.Test:443/a#fragment",
            },
            {"name": "missing", "page_url": ""},
            {"name": "invalid", "page_url": "http://["},
        ],
    )

    result = expand_selected_shard(
        [selection_record(path, tmp_path, rows=3, columns=2)],
        output_root=tmp_path / "structural",
        input_root=tmp_path,
    )

    entities = read_records(result.entities)
    assert len(entities) == 3
    refs = read_records(result.page_refs)
    assert refs == [
        {
            "entity_id": entities[0]["entity_id"],
            "page_url": "https://example.test/a",
            "row_id": 0,
            "source_table_id": entities[0]["appears_in"][0][
                "source_table_id"
            ],
            "url_key": hashlib.sha256(
                b"https://example.test/a"
            ).hexdigest(),
        }
    ]
    failures = read_records(result.structural_failures)
    assert [record["error_class"] for record in failures] == [
        "missing_page_url",
        "invalid_page_url",
    ]
    assert all(record["status"] == "terminal" for record in failures)


def test_direct_image_refs_keep_existing_builder_order_and_entity_urls(
    tmp_path: Path,
) -> None:
    path = write_wdc_gzip(
        tmp_path,
        rows=[
            {
                "name": "nested",
                "page_url": "https://example.test/items/1",
                "image": {
                    "primary": "/a.jpg",
                    "more": [
                        "https://cdn.test/b.jpg",
                        "/a.jpg",
                    ],
                },
            }
        ],
    )

    result = expand_selected_shard(
        [selection_record(path, tmp_path, rows=1, columns=3)],
        output_root=tmp_path / "structural",
        input_root=tmp_path,
    )

    entity = read_records(result.entities)[0]
    assert entity["image_urls"] == [
        "https://example.test/a.jpg",
        "https://cdn.test/b.jpg",
    ]
    refs = read_records(result.direct_image_refs)
    assert [record["image_url"] for record in refs] == entity["image_urls"]
    assert [record["ordinal"] for record in refs] == [0, 1]


def test_content_hash_is_uncompressed_jsonl_bytes_from_the_expansion_pass(
    tmp_path: Path,
) -> None:
    path = write_wdc_gzip(
        tmp_path,
        rows=[
            {"name": "A", "page_url": "https://example.test/a"},
            {"name": "B", "page_url": "https://example.test/b"},
        ],
    )

    result = expand_selected_shard(
        [selection_record(path, tmp_path, rows=2, columns=2)],
        output_root=tmp_path / "structural",
        input_root=tmp_path,
    )

    with gzip.open(path, "rb") as handle:
        expected_hash = hashlib.sha256(handle.read()).hexdigest()
    validated = read_records(result.validated_selection)[0]
    assert validated["content_hash"] == expected_hash
    assert (
        validated["content_hash_semantics"]
        == "sha256-uncompressed-jsonl-bytes"
    )
    assert validated["rows"] == 2
    assert validated["columns"] == 2


def test_global_finalize_is_exact_validated_and_idempotent(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "structural"
    manifests: list[Path] = []
    for shard_index in range(2):
        path = write_wdc_gzip(
            tmp_path,
            host=f"shop-{shard_index}.test",
            rows=[
                {
                    "name": f"item-{shard_index}",
                    "page_url": f"https://example.test/{shard_index}",
                }
            ],
        )
        result = expand_selected_shard(
            [selection_record(path, tmp_path, rows=1, columns=2)],
            output_root=output_root,
            input_root=tmp_path,
            shard_id=f"{shard_index:05d}",
        )
        manifests.append(result.manifest)

    finalized = finalize_validated_selection(
        manifests,
        output_root=output_root,
        target_tables=2,
    )
    assert read_manifest(finalized.manifest)["schema_version"] == (
        "wdc200k-structural-v2"
    )
    records = read_records(finalized.validated_selection)
    assert len(records) == 2
    assert len({record["relative_path"] for record in records}) == 2
    assert len({record["source_table_id"] for record in records}) == 2
    checksum = hashlib.sha256(
        finalized.validated_selection.read_bytes()
    ).hexdigest()
    mtime = finalized.validated_selection.stat().st_mtime_ns

    resumed = finalize_validated_selection(
        manifests,
        output_root=output_root,
        target_tables=2,
    )

    assert resumed.tables == 2
    assert (
        hashlib.sha256(resumed.validated_selection.read_bytes()).hexdigest()
        == checksum
    )
    assert resumed.validated_selection.stat().st_mtime_ns == mtime

    damaged_manifest = read_manifest(manifests[0])
    damaged_manifest["completed_shards"] = [
        shard
        for shard in damaged_manifest["completed_shards"]
        if not shard["path"].startswith("direct_image_refs/")
    ]
    write_manifest(manifests[0], damaged_manifest)
    with pytest.raises(
        StructuralExpansionError,
        match="artifact",
    ):
        finalize_validated_selection(
            manifests,
            output_root=output_root,
            target_tables=2,
        )


@pytest.mark.parametrize("target_tables", [1, 3])
def test_global_finalize_rejects_non_exact_target(
    tmp_path: Path,
    target_tables: int,
) -> None:
    output_root = tmp_path / "structural"
    first = write_wdc_gzip(
        tmp_path,
        host="first.test",
        rows=[{"name": "A", "page_url": "https://example.test/a"}],
    )
    second = write_wdc_gzip(
        tmp_path,
        host="second.test",
        rows=[{"name": "B", "page_url": "https://example.test/b"}],
    )
    result = expand_selected_shard(
        [
            selection_record(first, tmp_path, rows=1, columns=2),
            selection_record(second, tmp_path, rows=1, columns=2),
        ],
        output_root=output_root,
        input_root=tmp_path,
    )

    with pytest.raises(StructuralExpansionError, match="target"):
        finalize_validated_selection(
            [result.manifest],
            output_root=output_root,
            target_tables=target_tables,
        )

    assert not (
        output_root / "selection" / "validated-selected-tables.jsonl"
    ).exists()


def test_global_finalize_rejects_duplicate_structural_manifest(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "structural"
    path = write_wdc_gzip(
        tmp_path,
        rows=[{"name": "A", "page_url": "https://example.test/a"}],
    )
    result = expand_selected_shard(
        [selection_record(path, tmp_path, rows=1, columns=2)],
        output_root=output_root,
        input_root=tmp_path,
    )

    with pytest.raises(StructuralExpansionError, match="duplicate"):
        finalize_validated_selection(
            [result.manifest, result.manifest],
            output_root=output_root,
            target_tables=2,
        )


def test_global_finalize_rejects_duplicate_validated_records(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "structural"
    path = write_wdc_gzip(
        tmp_path,
        rows=[{"name": "A", "page_url": "https://example.test/a"}],
    )
    manifests = [
        expand_selected_shard(
            [selection_record(path, tmp_path, rows=1, columns=2)],
            output_root=output_root,
            input_root=tmp_path,
            shard_id=f"{index:05d}",
        ).manifest
        for index in range(2)
    ]

    with pytest.raises(StructuralExpansionError, match="duplicate"):
        finalize_validated_selection(
            manifests,
            output_root=output_root,
            target_tables=2,
        )


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_global_finalize_rejects_missing_or_corrupt_structural_shard(
    tmp_path: Path,
    damage: str,
) -> None:
    output_root = tmp_path / "structural"
    path = write_wdc_gzip(
        tmp_path,
        rows=[{"name": "A", "page_url": "https://example.test/a"}],
    )
    result = expand_selected_shard(
        [selection_record(path, tmp_path, rows=1, columns=2)],
        output_root=output_root,
        input_root=tmp_path,
    )
    if damage == "missing":
        result.entities.unlink()
    else:
        result.entities.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        StructuralExpansionError,
        match="checksum|missing",
    ):
        finalize_validated_selection(
            [result.manifest],
            output_root=output_root,
            target_tables=1,
        )


@pytest.mark.parametrize(
    "missing_prefix",
    [
        "entities/",
        "page_refs/",
        "direct_image_refs/",
        "structural_failures/",
    ],
)
def test_global_finalize_requires_every_structural_artifact(
    tmp_path: Path,
    missing_prefix: str,
) -> None:
    output_root = tmp_path / "structural"
    path = write_wdc_gzip(
        tmp_path,
        rows=[{"name": "A", "page_url": ""}],
    )
    result = expand_selected_shard(
        [selection_record(path, tmp_path, rows=1, columns=1)],
        output_root=output_root,
        input_root=tmp_path,
    )
    manifest = read_manifest(result.manifest)
    manifest["completed_shards"] = [
        shard
        for shard in manifest["completed_shards"]
        if not shard["path"].startswith(missing_prefix)
    ]
    write_manifest(result.manifest, manifest)

    with pytest.raises(StructuralExpansionError, match="artifact"):
        finalize_validated_selection(
            [result.manifest],
            output_root=output_root,
            target_tables=1,
        )


@pytest.mark.parametrize("damage", ["duplicate", "unknown"])
def test_global_finalize_rejects_duplicate_or_unknown_artifact(
    tmp_path: Path,
    damage: str,
) -> None:
    output_root = tmp_path / "structural"
    path = write_wdc_gzip(
        tmp_path,
        rows=[{"name": "A", "page_url": "https://example.test/a"}],
    )
    result = expand_selected_shard(
        [selection_record(path, tmp_path, rows=1, columns=2)],
        output_root=output_root,
        input_root=tmp_path,
    )
    manifest = read_manifest(result.manifest)
    if damage == "duplicate":
        entity_shard = next(
            shard
            for shard in manifest["completed_shards"]
            if shard["path"].startswith("entities/")
        )
        manifest["completed_shards"].append(dict(entity_shard))
    else:
        unknown = output_root / "unknown" / "part-00000.jsonl"
        unknown.parent.mkdir(parents=True)
        unknown.write_text("{}\n", encoding="utf-8")
        content = unknown.read_bytes()
        manifest["completed_shards"].append(
            {
                "path": "unknown/part-00000.jsonl",
                "records": 1,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    write_manifest(result.manifest, manifest)

    with pytest.raises(StructuralExpansionError, match="artifact"):
        finalize_validated_selection(
            [result.manifest],
            output_root=output_root,
            target_tables=1,
        )


def test_global_finalize_checks_entity_count_against_validated_rows(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "structural"
    path = write_wdc_gzip(
        tmp_path,
        rows=[{"name": "A", "page_url": "https://example.test/a"}],
    )
    result = expand_selected_shard(
        [selection_record(path, tmp_path, rows=1, columns=2)],
        output_root=output_root,
        input_root=tmp_path,
    )
    validated = read_records(result.validated_selection)
    validated[0]["rows"] = 2
    result.validated_selection.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in validated
        ),
        encoding="utf-8",
    )
    manifest = read_manifest(result.manifest)
    refresh_manifest_shard(
        manifest,
        relative_path=(
            result.validated_selection.relative_to(output_root).as_posix()
        ),
        absolute_path=result.validated_selection,
    )
    write_manifest(result.manifest, manifest)

    with pytest.raises(StructuralExpansionError, match="entity count"):
        finalize_validated_selection(
            [result.manifest],
            output_root=output_root,
            target_tables=1,
        )


def test_global_finalize_checks_page_ref_or_terminal_failure_per_entity(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "structural"
    path = write_wdc_gzip(
        tmp_path,
        rows=[{"name": "A", "page_url": ""}],
    )
    result = expand_selected_shard(
        [selection_record(path, tmp_path, rows=1, columns=1)],
        output_root=output_root,
        input_root=tmp_path,
    )
    result.structural_failures.write_text("", encoding="utf-8")
    manifest = read_manifest(result.manifest)
    refresh_manifest_shard(
        manifest,
        relative_path=(
            result.structural_failures.relative_to(output_root).as_posix()
        ),
        absolute_path=result.structural_failures,
    )
    write_manifest(result.manifest, manifest)

    with pytest.raises(StructuralExpansionError, match="page coverage"):
        finalize_validated_selection(
            [result.manifest],
            output_root=output_root,
            target_tables=1,
        )


def test_global_finalize_accepts_zero_page_and_direct_image_ref_shards(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "structural"
    path = write_wdc_gzip(
        tmp_path,
        rows=[{"name": "A", "page_url": ""}],
    )
    result = expand_selected_shard(
        [selection_record(path, tmp_path, rows=1, columns=1)],
        output_root=output_root,
        input_root=tmp_path,
    )

    finalized = finalize_validated_selection(
        [result.manifest],
        output_root=output_root,
        target_tables=1,
    )

    assert finalized.tables == 1
    assert result.page_refs.stat().st_size == 0
    assert result.direct_image_refs.stat().st_size == 0


def test_each_selection_shard_requires_its_own_exact_success_count(
    tmp_path: Path,
) -> None:
    first_path = write_wdc_gzip(
        tmp_path,
        host="first.test",
        rows=[{"name": "A", "page_url": "https://example.test/a"}],
    )
    second_path = write_wdc_gzip(
        tmp_path,
        host="second.test",
        rows=[{"name": "B", "page_url": "https://example.test/b"}],
    )
    selected = [
        candidate(first_path, tmp_path, rows=1),
        candidate(second_path, tmp_path, rows=1),
    ]
    manager = ReserveManager.create(
        tmp_path / "reserve.sqlite3",
        reserve=[],
        selected=selected,
        policy=SelectionPolicy(target_tables=2),
    )

    first_result = expand_selected_shard(
        [selection_record(first_path, tmp_path, rows=1, columns=2)],
        output_root=tmp_path / "structural",
        input_root=tmp_path,
        reserve_manager=manager,
        shard_id="00000",
    )
    second_result = expand_selected_shard(
        [selection_record(second_path, tmp_path, rows=1, columns=2)],
        output_root=tmp_path / "structural",
        input_root=tmp_path,
        reserve_manager=manager,
        shard_id="00001",
    )

    assert first_result.tables == 1
    assert second_result.tables == 1


def test_invalid_selected_table_uses_reserve_and_records_provenance(
    tmp_path: Path,
) -> None:
    bad = write_wdc_gzip(tmp_path, host="bad.test", rows=["{bad-json"])
    good = write_wdc_gzip(
        tmp_path,
        host="good.test",
        rows=[
            {"name": "A", "page_url": "https://example.test/a"},
            {"name": "B", "page_url": "https://example.test/b"},
        ],
    )
    selected = candidate(bad, tmp_path, rows=1)
    reserve = candidate(good, tmp_path, rows=2)
    policy = SelectionPolicy(target_tables=1)
    manager = ReserveManager.create(
        tmp_path / "reserve.sqlite3",
        reserve=[reserve],
        selected=[selected],
        policy=policy,
    )

    result = expand_selected_shard(
        [selection_record(bad, tmp_path, rows=1, columns=2)],
        output_root=tmp_path / "structural",
        input_root=tmp_path,
        reserve_manager=manager,
    )

    assert read_records(result.source_tables)[0]["source_file"] == (
        reserve.relative_path
    )
    validated = read_records(result.validated_selection)[0]
    assert validated["relative_path"] == reserve.relative_path
    assert validated["replaces_path"] == selected.relative_path
    assert validated["replacement_reason"].startswith("JSONDecodeError:")
    assert [claim.status for claim in manager.terminal_claims()] == ["acked"]
    assert manager.pending_claims() == []


def test_structural_progress_reports_invalid_candidate_replacement(
    tmp_path: Path,
) -> None:
    bad = write_wdc_gzip(tmp_path, host="bad.test", rows=["{bad-json"])
    good = write_wdc_gzip(
        tmp_path,
        host="good.test",
        rows=[{"name": "A", "page_url": "https://example.test/a"}],
    )
    selected = candidate(bad, tmp_path, rows=1)
    reserve = candidate(good, tmp_path, rows=1)
    manager = ReserveManager.create(
        tmp_path / "reserve.sqlite3",
        reserve=[reserve],
        selected=[selected],
        policy=SelectionPolicy(target_tables=1),
    )
    events: list[dict[str, Any]] = []

    expand_selected_shard(
        [selection_record(bad, tmp_path, rows=1, columns=2)],
        output_root=tmp_path / "structural",
        input_root=tmp_path,
        reserve_manager=manager,
        progress_callback=events.append,
    )

    replacement = next(
        event
        for event in events
        if event["phase"] == "replace_invalid_candidate"
    )
    assert replacement["relative_path"] == selected.relative_path
    assert replacement["replacement_path"] == reserve.relative_path
    assert replacement["completed"] == 0
    assert replacement["reason"].startswith("JSONDecodeError:")


def test_structural_acknowledges_preclaimed_recovery_replacement(
    tmp_path: Path,
) -> None:
    original = write_wdc_gzip(
        tmp_path,
        host="original.test",
        rows=[{"name": "old", "page_url": "https://example.test/old"}],
    )
    replacement = write_wdc_gzip(
        tmp_path,
        host="replacement.test",
        rows=[{"name": "new", "page_url": "https://example.test/new"}],
    )
    selected = candidate(original, tmp_path, rows=1)
    reserve = candidate(replacement, tmp_path, rows=1)
    manager = ReserveManager.create(
        tmp_path / "reserve.sqlite3",
        reserve=[reserve],
        selected=[selected],
        policy=SelectionPolicy(target_tables=1),
    )
    claim = manager.claim_replacement(
        operation_key="recovery-round-1",
        invalid_candidate=selected,
        reason="unrecoverable after auto-check",
        retain_on_exhaustion=True,
    )
    record = selection_record(replacement, tmp_path, rows=1, columns=2)
    record["replacement_operation_key"] = claim.operation_key

    result = expand_selected_shard(
        [record],
        output_root=tmp_path / "structural",
        input_root=tmp_path,
        reserve_manager=manager,
    )

    assert read_records(result.validated_selection)[0][
        "replacement_operation_key"
    ] == claim.operation_key
    assert manager.pending_claims() == []
    assert manager.terminal_claims()[0].status == "acked"


def test_replacement_chain_supersedes_broken_replacement(
    tmp_path: Path,
) -> None:
    first = write_wdc_gzip(tmp_path, host="first.test", rows=["{bad"])
    second = write_wdc_gzip(tmp_path, host="second.test", rows=["[bad"])
    good = write_wdc_gzip(
        tmp_path,
        host="third.test",
        rows=[{"name": "ok", "page_url": "https://example.test/ok"}],
    )
    selected = candidate(first, tmp_path, rows=1)
    reserves = [
        candidate(second, tmp_path, rows=1),
        candidate(good, tmp_path, rows=1),
    ]
    manager = ReserveManager.create(
        tmp_path / "reserve.sqlite3",
        reserve=reserves,
        selected=[selected],
        policy=SelectionPolicy(target_tables=1),
    )

    result = expand_selected_shard(
        [selection_record(first, tmp_path, rows=1, columns=2)],
        output_root=tmp_path / "structural",
        input_root=tmp_path,
        reserve_manager=manager,
    )

    validated = read_records(result.validated_selection)[0]
    assert validated["relative_path"] == reserves[1].relative_path
    assert validated["replaces_path"] == selected.relative_path
    assert validated["replacement_chain"] == [
        selected.relative_path,
        reserves[0].relative_path,
    ]
    assert [claim.status for claim in manager.terminal_claims()] == [
        "superseded",
        "acked",
    ]


def test_pending_replacement_chain_is_reconstructed_after_restart(
    tmp_path: Path,
) -> None:
    first = write_wdc_gzip(tmp_path, host="first.test", rows=["{bad"])
    second = write_wdc_gzip(tmp_path, host="second.test", rows=["[bad"])
    good = write_wdc_gzip(
        tmp_path,
        host="third.test",
        rows=[{"name": "ok", "page_url": "https://example.test/ok"}],
    )
    selected = candidate(first, tmp_path, rows=1)
    reserves = [
        candidate(second, tmp_path, rows=1),
        candidate(good, tmp_path, rows=1),
    ]
    manager = ReserveManager.create(
        tmp_path / "reserve.sqlite3",
        reserve=reserves,
        selected=[selected],
        policy=SelectionPolicy(target_tables=1),
    )
    first_reason = invalid_json_reason("{bad")
    first_claim = manager.claim_replacement(
        operation_key=stable_operation_key("00000", selected.relative_path),
        invalid_candidate=selected,
        reason=first_reason,
    )
    assert first_claim.replacement == reserves[0]
    second_reason = invalid_json_reason("[bad")
    second_claim = manager.claim_replacement(
        operation_key=stable_operation_key(
            "00000", reserves[0].relative_path
        ),
        invalid_candidate=reserves[0],
        reason=second_reason,
    )
    assert second_claim.replacement == reserves[1]

    result = expand_selected_shard(
        [selection_record(first, tmp_path, rows=1, columns=2)],
        output_root=tmp_path / "structural",
        input_root=tmp_path,
        reserve_manager=manager,
    )

    validated = read_records(result.validated_selection)[0]
    assert validated["replacement_chain"] == [
        selected.relative_path,
        reserves[0].relative_path,
    ]
    assert validated["replacement_reasons"] == [
        first_reason,
        second_reason,
    ]


def test_replacement_recovery_reads_only_the_current_successor_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = write_wdc_gzip(tmp_path, host="first.test", rows=["{bad"])
    second = write_wdc_gzip(tmp_path, host="second.test", rows=["[bad"])
    good = write_wdc_gzip(
        tmp_path,
        host="third.test",
        rows=[{"name": "ok", "page_url": "https://example.test/ok"}],
    )
    main_selected = candidate(first, tmp_path, rows=1)
    unrelated_selected = [
        TableCandidate(
            schema_class="Product",
            subset="minimum3",
            host=f"unrelated-{index}.test",
            relative_path=(
                "Product/"
                f"Product_unrelated-{index}.test_October2023.json.gz"
            ),
            rows=1,
            columns=2,
        )
        for index in range(30)
    ]
    unrelated_reserve = [
        TableCandidate(
            schema_class="Product",
            subset="minimum3",
            host=f"reserve-{index}.test",
            relative_path=(
                "Product/"
                f"Product_reserve-{index}.test_October2023.json.gz"
            ),
            rows=1,
            columns=2,
        )
        for index in range(30)
    ]
    chain_reserve = [
        candidate(second, tmp_path, rows=1),
        candidate(good, tmp_path, rows=1),
    ]
    manager = ReserveManager.create(
        tmp_path / "reserve.sqlite3",
        reserve=[*chain_reserve, *unrelated_reserve],
        selected=[main_selected, *unrelated_selected],
        policy=SelectionPolicy(target_tables=31),
    )
    first_claim = manager.claim_replacement(
        operation_key=stable_operation_key(
            "00000", main_selected.relative_path
        ),
        invalid_candidate=main_selected,
        reason=invalid_json_reason("{bad"),
    )
    manager.claim_replacement(
        operation_key=stable_operation_key(
            "00000", chain_reserve[0].relative_path
        ),
        invalid_candidate=chain_reserve[0],
        reason=invalid_json_reason("[bad"),
    )
    assert first_claim.status == "pending"
    for index, selected in enumerate(unrelated_selected):
        claim = manager.claim_replacement(
            operation_key=f"unrelated-{index}",
            invalid_candidate=selected,
            reason="unrelated",
        )
        manager.acknowledge(
            operation_key=claim.operation_key,
            replacement_path=str(claim.replacement_path),
        )

    def forbid_global_history_scan() -> Any:
        raise AssertionError("global replacement history was loaded")

    monkeypatch.setattr(manager, "pending_claims", forbid_global_history_scan)
    monkeypatch.setattr(manager, "terminal_claims", forbid_global_history_scan)
    original_get = manager.get_claim_by_operation
    queried: list[str] = []

    def tracked_get(operation_key: str) -> Any:
        queried.append(operation_key)
        return original_get(operation_key)

    monkeypatch.setattr(manager, "get_claim_by_operation", tracked_get)

    result = expand_selected_shard(
        [selection_record(first, tmp_path, rows=1, columns=2)],
        output_root=tmp_path / "structural",
        input_root=tmp_path,
        reserve_manager=manager,
    )

    assert result.tables == 1
    assert queried == [
        stable_operation_key("00000", chain_reserve[0].relative_path)
    ]


def test_durable_output_before_ack_recovers_pending_claim_without_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = write_wdc_gzip(tmp_path, host="bad.test", rows=["{bad"])
    good = write_wdc_gzip(
        tmp_path,
        host="good.test",
        rows=[{"name": "ok", "page_url": "https://example.test/ok"}],
    )
    selected = candidate(bad, tmp_path, rows=1)
    reserve = candidate(good, tmp_path, rows=1)
    manager = ReserveManager.create(
        tmp_path / "reserve.sqlite3",
        reserve=[reserve],
        selected=[selected],
        policy=SelectionPolicy(target_tables=1),
    )
    records = [selection_record(bad, tmp_path, rows=1, columns=2)]
    output_root = tmp_path / "structural"
    acknowledge = manager.acknowledge

    def crash_before_ack(**_kwargs: str) -> Any:
        raise RuntimeError("simulated crash before ack")

    monkeypatch.setattr(manager, "acknowledge", crash_before_ack)
    with pytest.raises(RuntimeError, match="simulated crash"):
        expand_selected_shard(
            records,
            output_root=output_root,
            input_root=tmp_path,
            reserve_manager=manager,
        )

    source_path = next((output_root / "source_tables").glob("*.jsonl"))
    checksum_before = hashlib.sha256(source_path.read_bytes()).hexdigest()
    mtime_before = source_path.stat().st_mtime_ns
    assert len(manager.pending_claims()) == 1

    monkeypatch.setattr(manager, "acknowledge", acknowledge)
    resumed = expand_selected_shard(
        records,
        output_root=output_root,
        input_root=tmp_path,
        reserve_manager=manager,
    )

    assert resumed.source_tables == source_path
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == checksum_before
    assert source_path.stat().st_mtime_ns == mtime_before
    assert manager.pending_claims() == []


def test_malformed_gzip_does_not_publish_partial_structural_shards(
    tmp_path: Path,
) -> None:
    path = write_wdc_gzip(
        tmp_path,
        rows=[
            {"name": "valid", "page_url": "https://example.test/valid"},
            "{bad-json",
        ],
    )
    output_root = tmp_path / "structural"

    with pytest.raises(StructuralExpansionError, match="JSONDecodeError"):
        expand_selected_shard(
            [selection_record(path, tmp_path, rows=2, columns=2)],
            output_root=output_root,
            input_root=tmp_path,
        )

    assert not list((output_root / "source_tables").glob("*.jsonl"))
    assert not list(output_root.rglob("*.tmp"))


def test_invalid_utf8_and_corrupt_gzip_do_not_publish(
    tmp_path: Path,
) -> None:
    invalid_utf8 = (
        tmp_path / "Product" / "Product_utf8.test_October2023.json.gz"
    )
    invalid_utf8.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(invalid_utf8, "wb") as handle:
        handle.write(b'{"name":"ok","page_url":"https://e.test"}\n')
        handle.write(b"\xff\n")
    corrupt = (
        tmp_path / "Product" / "Product_corrupt.test_October2023.json.gz"
    )
    corrupt.write_bytes(b"not-a-gzip-stream")

    for shard_id, path in (("utf8", invalid_utf8), ("gzip", corrupt)):
        output_root = tmp_path / f"structural-{shard_id}"
        with pytest.raises(StructuralExpansionError):
            expand_selected_shard(
                [selection_record(path, tmp_path, rows=2, columns=2)],
                output_root=output_root,
                input_root=tmp_path,
                shard_id=shard_id,
            )
        assert not list((output_root / "source_tables").glob("*.jsonl"))
        assert not list(output_root.rglob("*.tmp"))


def test_resume_after_mid_commit_failure_rewrites_a_complete_artifact_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_wdc_gzip(
        tmp_path,
        rows=[{"name": "A", "page_url": "https://example.test/a"}],
    )
    records = [selection_record(path, tmp_path, rows=1, columns=2)]
    output_root = tmp_path / "structural"
    original_commit = wdc200k_io_module.AtomicJsonlShard.commit
    calls = 0

    def fail_third_commit(self: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("simulated commit failure")
        return original_commit(self)

    monkeypatch.setattr(
        wdc200k_io_module.AtomicJsonlShard,
        "commit",
        fail_third_commit,
    )
    with pytest.raises(OSError, match="simulated commit failure"):
        expand_selected_shard(
            records,
            output_root=output_root,
            input_root=tmp_path,
        )

    monkeypatch.setattr(
        wdc200k_io_module.AtomicJsonlShard,
        "commit",
        original_commit,
    )
    resumed = expand_selected_shard(
        records,
        output_root=output_root,
        input_root=tmp_path,
    )

    assert resumed.tables == 1
    assert len(read_records(resumed.source_tables)) == 1
    assert len(read_records(resumed.validated_selection)) == 1
    assert json.loads(resumed.manifest.read_text(encoding="utf-8"))[
        "complete"
    ]


def test_non_object_json_row_is_malformed_and_does_not_drop_a_row(
    tmp_path: Path,
) -> None:
    path = write_wdc_gzip(
        tmp_path,
        rows=[
            {"name": "valid", "page_url": "https://example.test/valid"},
            ["not", "an", "object"],
        ],
    )
    output_root = tmp_path / "structural"

    with pytest.raises(
        StructuralExpansionError,
        match="non-object JSON row",
    ):
        expand_selected_shard(
            [selection_record(path, tmp_path, rows=2, columns=2)],
            output_root=output_root,
            input_root=tmp_path,
        )

    assert not list((output_root / "source_tables").glob("*.jsonl"))
    assert not list(output_root.rglob("*.tmp"))


def test_large_source_table_bypasses_whole_record_jsonl_serialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_wdc_gzip(
        tmp_path,
        rows=[
            {
                "row_id": index,
                "name": f"item-{index}",
                "page_url": f"https://example.test/{index}",
                "value": str(index),
            }
            for index in range(2_000)
        ],
    )
    original_write = wdc200k_io_module.write_jsonl_record

    def reject_whole_source(handle: Any, record: dict[str, Any]) -> None:
        if (
            "source_table_id" in record
            and isinstance(record.get("rows"), list)
        ):
            raise AssertionError("whole source table reached json.dumps")
        original_write(handle, record)

    monkeypatch.setattr(
        wdc200k_io_module,
        "write_jsonl_record",
        reject_whole_source,
    )

    result = expand_selected_shard(
        [selection_record(path, tmp_path, rows=2_000, columns=3)],
        output_root=tmp_path / "structural",
        input_root=tmp_path,
    )

    source = read_records(result.source_tables)[0]
    assert source["num_rows"] == 2_000
    assert len(source["rows"]) == 2_000


def test_streamed_source_matches_adapter_with_sparse_late_columns(
    tmp_path: Path,
) -> None:
    path = write_wdc_gzip(
        tmp_path,
        rows=[
            {
                "name": "first",
                "page_url": "https://example.test/first",
                "score": "1",
            },
            {
                "name": "second",
                "page_url": "https://example.test/second",
                "late": {"nested": "value"},
            },
            {
                "name": "",
                "page_url": "https://example.test/third",
                "score": "not numeric",
                "late": ["a", "b"],
            },
        ],
    )
    baseline = read_wdc_table(
        path,
        tmp_path,
        min_rows=1,
        min_cols=1,
        max_rows=0,
    )

    result = expand_selected_shard(
        [selection_record(path, tmp_path, rows=3, columns=4)],
        output_root=tmp_path / "structural",
        input_root=tmp_path,
    )

    assert read_records(result.source_tables)[0] == baseline.source_table
    assert read_records(result.entities) == baseline.entities
