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
sys.path.insert(0, str(ROOT / "scripts"))

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
)


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


def test_structural_module_imports_through_scripts_package() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts.wdc200k_structural import expand_selected_shard",
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
