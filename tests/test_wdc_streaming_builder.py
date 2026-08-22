from __future__ import annotations

import gzip
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mmdd_dataset.joinability as joinability_module  # noqa: E402
import mmdd_dataset.wdc_pipeline as pipeline_module  # noqa: E402
from mmdd_dataset.joinability import (  # noqa: E402
    BuildConfig,
    build_joinability_dataset,
    build_joinability_for_table,
)
from mmdd_dataset.wdc_adapter import WdcCandidate, adapt_table  # noqa: E402
from mmdd_dataset.wdc_evidence import (  # noqa: E402
    ResultCache,
    execute_tasks,
    normalize_public_url,
)
from mmdd_dataset.wdc_pipeline import (  # noqa: E402
    WdcPipelineConfig,
    run_extract,
    run_fetch_evidence,
    run_materialize,
    run_normalize,
    run_select_sample,
    dry_run,
)
from mmdd_dataset.utils import get_cell  # noqa: E402
from mmdd_dataset.wdc_runtime import (  # noqa: E402
    iter_dataset_artifact,
    iter_jsonl,
    iter_manifest_artifact,
)


def write_wdc_table(
    root: Path, index: int, rows: int = 5, *, with_images: bool = False
) -> Path:
    class_dir = root / "Product"
    class_dir.mkdir(parents=True, exist_ok=True)
    path = class_dir / f"Product_shop{index}.test_October2023.json.gz"
    colors = ("red", "green", "blue", "black", "white")
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row_id in range(rows):
            record = {
                "row_id": row_id,
                "name": f"item-{index}-{row_id}",
                "color": colors[row_id % len(colors)],
                "category": "tool",
                "page_url": f"https://shop{index}.example/item/{row_id}",
            }
            if with_images:
                record["image"] = f"https://cdn.example/{index}/{row_id}.jpg"
            handle.write(
                json.dumps(record) + "\n"
            )
    return path


def config_for(
    tmp_path: Path,
    *,
    input_dir: Path,
    work_name: str = "work",
    output_name: str = "output",
    target_tables: int = 3,
    seed: int = 13,
) -> WdcPipelineConfig:
    return WdcPipelineConfig(
        input_dir=input_dir,
        work_dir=tmp_path / work_name,
        output_dir=tmp_path / output_name,
        target_tables=target_tables,
        seed=seed,
        shard_size=1,
        concurrency=2,
        entities_per_table=3,
        min_rows=3,
        min_cols=2,
        max_images_per_entity=1,
        min_free_bytes=0,
        query_rows=3,
        min_target_rows=3,
        min_recovered_rows=2,
    )


def selected_records(config: WdcPipelineConfig, artifact: str) -> list[dict[str, Any]]:
    return list(
        iter_manifest_artifact(
            config.work_dir / "select_sample" / "manifest.json", artifact
        )
    )


def test_selection_streams_catalog_and_seed_is_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    paths = [write_wdc_table(input_dir, index) for index in range(7)]

    class OnePassCatalog:
        def __init__(self) -> None:
            self.iterations = 0

        def __iter__(self) -> Iterator[WdcCandidate]:
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("catalog was materialized or replayed")
            for path in paths:
                yield WdcCandidate(
                    relative_path=path.relative_to(input_dir).as_posix(),
                    schema_class="Product",
                    subset="unstratified",
                    host=path.stem,
                    rows=5,
                    columns=4,
                )

    first_catalog = OnePassCatalog()
    monkeypatch.setattr(pipeline_module, "iter_candidates", lambda _root: iter(first_catalog))
    first = config_for(tmp_path, input_dir=input_dir, work_name="work-a", output_name="out-a")
    run_select_sample(first)
    first_tables = selected_records(first, "selected_tables")
    first_entities = selected_records(first, "sampled_entities")

    second_catalog = OnePassCatalog()
    monkeypatch.setattr(pipeline_module, "iter_candidates", lambda _root: iter(second_catalog))
    second = config_for(tmp_path, input_dir=input_dir, work_name="work-b", output_name="out-b")
    run_select_sample(second)

    assert first_catalog.iterations == second_catalog.iterations == 1
    assert [record["relative_path"] for record in first_tables] == [
        record["relative_path"] for record in selected_records(second, "selected_tables")
    ]
    assert [record["entity_id"] for record in first_entities] == [
        record["entity_id"] for record in selected_records(second, "sampled_entities")
    ]
    first_shard = next((first.work_dir / "select_sample" / "selected_tables").glob("*.jsonl"))
    assert len(first_shard.read_text(encoding="utf-8").splitlines()) == 1


def test_normalize_resume_skips_committed_shard_and_ignores_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    for index in range(3):
        write_wdc_table(input_dir, index)
    config = config_for(tmp_path, input_dir=input_dir)
    run_select_sample(config)

    calls: list[str] = []
    original = pipeline_module.adapt_table

    def counted(path: Path, *args: Any, **kwargs: Any) -> Any:
        calls.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pipeline_module, "adapt_table", counted)

    def interrupt(index: int) -> None:
        if index == 0:
            raise RuntimeError("planned interruption")

    with pytest.raises(RuntimeError, match="planned interruption"):
        run_normalize(config, after_shard=interrupt)

    source_root = config.work_dir / "normalize" / "source_tables"
    temporary = source_root / "part-00001.jsonl.tmp"
    temporary.write_text('{"not":"complete"}\n', encoding="utf-8")
    first_name = calls[0]

    manifest = run_normalize(config)

    assert manifest["complete"] is True
    assert calls.count(first_name) == 1
    assert len(manifest["outputs"]["source_tables"]) == 3
    assert json.loads((source_root / "part-00001.jsonl").read_text().splitlines()[0])[
        "source_table_id"
    ].startswith("st_wdc_")


def test_final_dataset_is_self_contained_after_work_directory_is_renamed(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    for index in range(2):
        write_wdc_table(input_dir, index, with_images=True)
    config = config_for(tmp_path, input_dir=input_dir, target_tables=2)
    run_select_sample(config)
    run_normalize(config)

    def page_fetch(url: str) -> dict[str, Any]:
        return {
            "status": "success",
            "url": url,
            "text": "This product is a tool.",
            "image_urls": [],
            "bytes": 23,
        }

    cached_image = tmp_path / "cached-image.jpg"
    cached_image.write_bytes(b"synthetic-image")

    def image_fetch(url: str) -> dict[str, Any]:
        return {
            "status": "success",
            "url": url,
            "cache_path": str(cached_image),
            "content_type": "image/jpeg",
            "bytes": cached_image.stat().st_size,
        }

    run_fetch_evidence(
        config,
        page_fetcher=page_fetch,
        image_fetcher=image_fetch,
    )
    import_path = tmp_path / "imported_extractions.jsonl"
    config = replace(config, import_extractions=import_path)
    normalize_manifest = json.loads(
        (config.work_dir / "normalize" / "manifest.json").read_text(encoding="utf-8")
    )
    evidence_manifest = json.loads(
        (config.work_dir / "fetch_evidence" / "manifest.json").read_text(encoding="utf-8")
    )
    sampled_shards = pipeline_module._sampled_entity_shards(config)
    imported = []
    for index, table_record in enumerate(normalize_manifest["outputs"]["source_tables"]):
        table_path = config.work_dir / "normalize" / table_record["path"]
        assets = pipeline_module._successful_assets_for_shard(
            config, evidence_manifest, index
        )
        tasks = pipeline_module._model_tasks_for_shard(
            config, table_path, sampled_shards[index], assets
        )
        tables = {
            table["source_table_id"]: table for table in iter_jsonl(table_path)
        }
        for task in tasks:
            table = tables[task["source_table_id"]]
            column = next(
                column
                for column in table["columns"]
                if column["column_name"] == task["attribute_name"]
            )
            row = next(
                row for row in table["rows"] if row["row_id"] == task["source_row_id"]
            )
            imported.append(
                {
                    "task_id": task["task_id"],
                    "value": get_cell(row, column["column_index"])["text"],
                    "evidence": "synthetic page evidence",
                }
            )
    import_path.write_text(
        "".join(json.dumps(record) + "\n" for record in imported),
        encoding="utf-8",
    )
    assert run_extract(config)["complete"] is True
    run_materialize(config)

    manifest_path = config.output_dir / "dataset_manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["complete"] is True
    assert str(config.work_dir) not in manifest_text
    assert all(
        not Path(shard["path"]).is_absolute()
        for artifact in manifest["artifacts"].values()
        for shard in artifact["shards"]
    )

    renamed_work = tmp_path / "renamed-work"
    config.work_dir.rename(renamed_work)
    source_tables = list(iter_dataset_artifact(config.output_dir, "source_tables"))
    decisions = list(
        iter_dataset_artifact(config.output_dir, "table_queryability_decisions")
    )
    image_assets = [
        asset
        for asset in iter_dataset_artifact(config.output_dir, "bridge_assets")
        if asset["asset_type"] == "image"
    ]
    assert len(source_tables) == len(decisions) == 2
    assert len(list(iter_dataset_artifact(config.output_dir, "query_tables"))) == 2
    assert image_assets
    assert all(not Path(asset["local_path"]).is_absolute() for asset in image_assets)
    assert all((config.output_dir / asset["local_path"]).is_file() for asset in image_assets)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/a",
        "http://127.0.0.1/a",
        "http://10.0.0.1/a",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/a",
        "https://user:password@example.com/a",
        "javascript:alert(1)",
    ],
)
def test_url_safety_rejects_non_public_targets(url: str) -> None:
    assert normalize_public_url(url) is None


def test_url_safety_accepts_public_http_url() -> None:
    assert normalize_public_url("HTTPS://Example.COM:443/a?q=1#fragment") == (
        "https://example.com/a?q=1"
    )


def test_dry_run_estimates_without_creating_work_or_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    write_wdc_table(input_dir, 0)
    config = config_for(tmp_path, input_dir=input_dir, target_tables=1)

    estimate = dry_run(config)

    assert estimate["selected_tables"] == 1
    assert estimate["network_or_model_calls"] == 0
    assert not config.work_dir.exists()
    assert not config.output_dir.exists()


def test_evidence_cache_reuses_duplicate_url_across_runs(tmp_path: Path) -> None:
    calls = 0

    def fetch(url: str) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"status": "success", "url": url, "text": "cached"}

    tasks = [
        {
            "task_id": f"task-{index}",
            "entity_id": f"entity-{index}",
            "url": "https://example.com/same",
        }
        for index in range(2)
    ]
    cache = ResultCache(tmp_path / "cache.sqlite3")
    first = execute_tasks(
        tasks,
        kind="page",
        cache=cache,
        policy="test-v1",
        workers=2,
        fetch=fetch,
    )
    second = execute_tasks(
        tasks,
        kind="page",
        cache=cache,
        policy="test-v1",
        workers=2,
        fetch=fetch,
    )

    assert calls == 1
    assert first == second


def test_wdc_and_entitables_entrypoints_share_one_joinability_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    path = write_wdc_table(input_dir, 0)
    adapted = adapt_table(path, input_dir, min_rows=3, min_cols=2, max_rows=None)
    table = adapted.table
    assets = []
    extractions = []
    for entity in adapted.entities[:3]:
        asset_id = "asset_" + entity["entity_id"]
        assets.append(
            {
                "asset_id": asset_id,
                "entity_id": entity["entity_id"],
                "asset_type": "text",
            }
        )
        row_id = int(entity["source_row_id"])
        extractions.append(
            {
                "source_table_id": table["source_table_id"],
                "source_row_id": row_id,
                "entity_id": entity["entity_id"],
                "asset_id": asset_id,
                "asset_type": "text",
                "attribute_name": "color",
                "value": get_cell(table["rows"][row_id], 1)["text"],
                "evidence": "synthetic",
            }
        )
    calls = 0
    original = joinability_module._build_joinability_for_table

    def observed(*args: Any, **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(joinability_module, "_build_joinability_for_table", observed)
    build_config = BuildConfig(
        query_rows=3,
        min_target_rows=3,
        min_recovered_rows=2,
        min_recovered_ratio=0.6,
    )
    batch = build_joinability_dataset(
        [table], assets, extractions, {table["source_table_id"]: "train"}, build_config
    )
    streamed = build_joinability_for_table(
        table, assets, extractions, "train", build_config
    )

    assert calls == 2
    assert batch == streamed
    assert len(streamed["query_tables"]) == 1
