import hashlib
import json
import multiprocessing
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from wdc200k_assets import (  # noqa: E402
    ImageBudget,
    ImageOutcomeStore,
    asset_materialization_input_fingerprint,
    asset_planning_input_fingerprint,
    build_unique_image_jobs,
    fetch_unique_images,
    iter_entity_page_join,
    iter_image_failures,
    iter_image_outcomes,
    materialize_asset_shards,
    materialize_entity_assets,
    persist_entity_asset_plans,
    plan_entity_assets,
    validate_asset_plan_shards,
    validate_complete_image_fetch,
    validate_materialized_asset_shards,
    validate_unique_image_jobs,
)
from wdc200k_fetch import (  # noqa: E402
    FetchPolicy,
    fetch_unique_pages,
    iter_page_fanout,
)
from wdc200k_io import AtomicJsonlShard, SqliteJobStore  # noqa: E402
from build_wdc_mm_joinability_dataset import WdcWebClient  # noqa: E402
import build_wdc_mm_joinability_dataset as legacy_wdc_builder  # noqa: E402
from build_wdc200k_mm_joinability_dataset import (  # noqa: E402
    parse_args as parse_pipeline_args,
)
import wdc200k_assets as assets_module  # noqa: E402
from wdc200k_eta import DurableUrlCounts, UrlProgressSnapshot  # noqa: E402


def _open_image_outcome_store_process(
    path: str,
    results,
) -> None:
    try:
        ImageOutcomeStore(Path(path))
        results.put(("ok",))
    except BaseException as error:
        results.put(("error", type(error).__name__, str(error)))


def test_assets_module_supports_package_import() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts.wdc200k_assets import ImageBudget",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def entity(
    entity_id: str = "e1",
    *,
    image_urls: list[str] | None = None,
) -> dict:
    return {
        "entity_id": entity_id,
        "wiki_title": f"wdc_{entity_id}",
        "display_texts": [entity_id],
        "context_terms": [],
        "appears_in": [
            {
                "source_table_id": "st1",
                "query_view_id": None,
                "row_id": 7,
                "column_index": 0,
                "column_name": "name",
            }
        ],
        "page_url": f"https://e.test/{entity_id}",
        "image_urls": list(image_urls or []),
    }


def page(
    *,
    image_urls: list[str] | None = None,
    text: str = "",
) -> dict:
    return {
        "status": "success",
        "page_url": "https://e.test/e1",
        "final_url": "https://e.test/e1",
        "text": text,
        "image_urls": list(image_urls or []),
    }


def test_every_entity_uses_direct_first_then_page_images() -> None:
    plan = plan_entity_assets(
        entity=entity(
            image_urls=["https://i.test/direct.jpg"],
        ),
        page=page(
            image_urls=[
                "https://i.test/a.jpg",
                "https://i.test/b.jpg",
                "https://i.test/c.jpg",
            ]
        ),
        budget=ImageBudget(
            attempts_per_entity=3,
            retained_per_entity=3,
        ),
    )

    assert [job.image_url for job in plan.image_refs] == [
        "https://i.test/direct.jpg",
        "https://i.test/a.jpg",
        "https://i.test/b.jpg",
        "https://i.test/c.jpg",
    ]
    assert plan.page_was_required is True


def test_image_failure_does_not_remove_entity_or_table() -> None:
    result = materialize_entity_assets(
        entity("e1"),
        page=None,
        image_outcomes={},
    )

    assert result.bridge_assets == []
    assert result.entity_id == "e1"
    assert len(result.table_asset_links) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attempts_per_entity", -1),
        ("retained_per_entity", -1),
    ],
)
def test_image_budget_rejects_each_negative_limit(
    field: str,
    value: int,
) -> None:
    with pytest.raises(ValueError, match=field):
        replace(ImageBudget(), **{field: value})


def test_attempt_and_retention_budgets_are_independent() -> None:
    planned = plan_entity_assets(
        entity(
            image_urls=[
                "https://i.test/1.jpg",
                "https://i.test/2.jpg",
                "https://i.test/3.jpg",
            ]
        ),
        page(),
        ImageBudget(attempts_per_entity=3, retained_per_entity=1),
    )

    assert len(planned.image_refs) == 3


def test_candidate_urls_are_normalized_and_deduplicated_in_order() -> None:
    planned = plan_entity_assets(
        entity(
            image_urls=[
                "HTTPS://I.TEST/a.jpg",
                "https://i.test/a.jpg",
            ]
        ),
        page(
            image_urls=[
                "https://i.test/a.jpg",
                "https://i.test/b.jpg",
            ]
        ),
        ImageBudget(attempts_per_entity=4, retained_per_entity=4),
    )

    assert [
        (reference.image_url, reference.source)
        for reference in planned.image_refs
    ] == [
        ("https://i.test/a.jpg", "wdc_image_column"),
        ("https://i.test/b.jpg", "wdc_page_image"),
    ]


def test_successful_direct_image_never_disables_page_assets() -> None:
    result = materialize_entity_assets(
        entity(image_urls=["https://i.test/direct.jpg"]),
        page(
            text="e1 has page text that must still be consumed.",
            image_urls=["https://i.test/page.jpg"],
        ),
        image_outcomes={
            "https://i.test/direct.jpg": image_outcome(
                "https://i.test/direct.jpg",
                sha256="direct",
            ),
            "https://i.test/page.jpg": image_outcome(
                "https://i.test/page.jpg",
                sha256="page",
            ),
        },
        budget=ImageBudget(attempts_per_entity=2, retained_per_entity=2),
    )

    assert [asset["asset_type"] for asset in result.bridge_assets] == [
        "text",
        "image",
        "image",
    ]


def test_direct_failure_does_not_consume_page_image_success_quota() -> None:
    direct = "https://i.test/direct.jpg"
    page_images = [f"https://i.test/page-{index}.jpg" for index in range(3)]
    result = materialize_entity_assets(
        entity(image_urls=[direct]),
        page(image_urls=page_images),
        image_outcomes={
            direct: image_outcome(direct, status="terminal"),
            **{
                url: image_outcome(url, sha256=f"page-{index}")
                for index, url in enumerate(page_images)
            },
        },
        budget=ImageBudget(attempts_per_entity=3, retained_per_entity=3),
    )

    images = [
        asset for asset in result.bridge_assets if asset["asset_type"] == "image"
    ]
    assert [asset["image_url"] for asset in images] == page_images


def test_one_direct_success_keeps_page_text_and_two_remaining_page_images() -> None:
    direct = "https://i.test/direct.jpg"
    page_images = [f"https://i.test/page-{index}.jpg" for index in range(3)]
    result = materialize_entity_assets(
        entity(image_urls=[direct]),
        page(
            text="e1 page text remains available after a direct image succeeds.",
            image_urls=page_images,
        ),
        image_outcomes={
            direct: image_outcome(direct, sha256="direct"),
            **{
                url: image_outcome(url, sha256=f"page-{index}")
                for index, url in enumerate(page_images)
            },
        },
        budget=ImageBudget(attempts_per_entity=3, retained_per_entity=3),
    )

    images = [
        asset for asset in result.bridge_assets if asset["asset_type"] == "image"
    ]
    assert any(asset["asset_type"] == "text" for asset in result.bridge_assets)
    assert [asset["image_url"] for asset in images] == [direct, *page_images[:2]]


def test_cli_max_images_override_controls_final_retention(tmp_path: Path) -> None:
    args = parse_pipeline_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
            "--max_images_per_entity",
            "1",
        ]
    )
    urls = ["https://i.test/a.jpg", "https://i.test/b.jpg"]
    result = materialize_entity_assets(
        entity(image_urls=urls),
        page(),
        {
            url: image_outcome(url, sha256=str(index))
            for index, url in enumerate(urls)
        },
        ImageBudget(
            attempts_per_entity=3,
            retained_per_entity=args.max_images_per_entity,
        ),
    )
    assert sum(
        asset["asset_type"] == "image" for asset in result.bridge_assets
    ) == 1


def image_outcome(
    image_url: str,
    *,
    status: str = "success",
    sha256: str = "sha",
    file_name: str = "image.png",
) -> dict:
    if status != "success":
        return {
            "status": "terminal",
            "image_url": image_url,
            "error_class": "download_failed",
        }
    return {
        "status": "success",
        "image_url": image_url,
        "original_url": image_url,
        "final_url": image_url + "?final=1",
        "file_name": file_name,
        "local_path": f"/cache/images/{file_name}",
        "relative_path": f"images/{file_name}",
        "sha256": sha256,
        "width": 64,
        "height": 48,
        "mime_type": "image/png",
        "bytes": 123,
        "downloaded": True,
    }


def test_failed_early_candidate_does_not_hide_later_successes() -> None:
    urls = [f"https://i.test/{index}.jpg" for index in range(4)]
    result = materialize_entity_assets(
        entity(image_urls=urls),
        page(),
        image_outcomes={
            urls[0]: image_outcome(urls[0], status="terminal"),
            urls[1]: image_outcome(urls[1], sha256="s1"),
            urls[2]: image_outcome(urls[2], sha256="s2"),
            urls[3]: image_outcome(urls[3], sha256="s3"),
        },
        budget=ImageBudget(attempts_per_entity=4, retained_per_entity=2),
    )

    images = [
        asset
        for asset in result.bridge_assets
        if asset["asset_type"] == "image"
    ]
    assert [asset["image_url"] for asset in images] == urls[1:3]


def test_duplicate_image_content_is_retained_once_per_entity() -> None:
    urls = ["https://i.test/a.jpg", "https://i.test/b.jpg"]
    result = materialize_entity_assets(
        entity(image_urls=urls),
        page(),
        image_outcomes={
            url: image_outcome(
                url,
                sha256="same-content",
                file_name="same-content.png",
            )
            for url in urls
        },
        budget=ImageBudget(attempts_per_entity=2, retained_per_entity=2),
    )

    images = [
        asset
        for asset in result.bridge_assets
        if asset["asset_type"] == "image"
    ]
    assert len(images) == 1


def test_text_chunks_and_bridge_schema_match_current_wdc_builder() -> None:
    result = materialize_entity_assets(
        entity("e1"),
        page(
            text=(
                "e1 first paragraph with useful evidence.\n\n"
                + "x" * 900
            )
        ),
        image_outcomes={},
    )

    text_assets = [
        asset
        for asset in result.bridge_assets
        if asset["asset_type"] == "text"
    ]
    assert 1 <= len(text_assets) <= 3
    assert all(len(asset["content"]) <= 800 for asset in text_assets)
    assert set(text_assets[0]) == {
        "asset_id",
        "source_asset_id",
        "entity_id",
        "entity_wiki_title",
        "asset_type",
        "content",
        "text_chunk_index",
        "text_chunk_count",
        "selected_text_chunk_count",
        "text_chunk_relevance_score",
        "source",
        "url",
        "page_url",
        "final_url",
    }
    assert text_assets[0]["source"] == "wdc_page_text_chunk"


def test_text_asset_records_are_exactly_equal_to_current_wdc_builder() -> None:
    current_entity = entity("e1")
    current_page = page(
        text=(
            "e1 has canonical bridge evidence. "
            + "This sentence verifies stable text assets."
        )
    )

    class LegacyClient:
        def fetch_page(self, _page_url: str) -> dict:
            return {
                "final_url": current_page["final_url"],
                "text": current_page["text"],
                "image_urls": [],
            }

        def download_image(self, *_args, **_kwargs):
            raise AssertionError("zero image budget must not download")

    expected = legacy_wdc_builder.build_wdc_bridge_assets_for_entity(
        current_entity,
        LegacyClient(),
        max_images_per_entity=0,
    )
    actual = materialize_entity_assets(
        current_entity,
        current_page,
        {},
        ImageBudget(attempts_per_entity=0, retained_per_entity=0),
    ).bridge_assets

    assert actual == expected


def test_multi_chunk_text_assets_are_exactly_equal_to_current_wdc_builder() -> None:
    current_entity = entity("e1")
    current_page = page(
        text="\n\n".join(
            (
                f"section {index} e1 "
                + chr(ord("a") + index) * 390
            )
            for index in range(7)
        )
    )

    class LegacyClient:
        def fetch_page(self, _page_url: str) -> dict:
            return {
                "final_url": current_page["final_url"],
                "text": current_page["text"],
                "image_urls": [],
            }

        def download_image(self, *_args, **_kwargs):
            raise AssertionError("zero image budget must not download")

    expected = legacy_wdc_builder.build_wdc_bridge_assets_for_entity(
        current_entity,
        LegacyClient(),
        max_images_per_entity=0,
        text_asset_chunk_chars=800,
        min_text_asset_chunk_chars=120,
        max_text_asset_chunks_per_entity=3,
    )
    actual = materialize_entity_assets(
        current_entity,
        current_page,
        {},
        ImageBudget(attempts_per_entity=0, retained_per_entity=0),
        text_asset_chunk_chars=800,
        min_text_asset_chunk_chars=120,
        max_text_asset_chunks_per_entity=3,
    ).bridge_assets

    assert expected[0]["text_chunk_count"] > 3
    assert len(expected) == 3
    assert actual == expected


def test_image_and_link_schemas_match_current_reader() -> None:
    image_url = "https://i.test/a.jpg"
    result = materialize_entity_assets(
        entity("e1", image_urls=[image_url]),
        page(),
        image_outcomes={image_url: image_outcome(image_url)},
    )

    image_asset = next(
        asset
        for asset in result.bridge_assets
        if asset["asset_type"] == "image"
    )
    assert set(image_asset) == {
        "asset_id",
        "entity_id",
        "asset_type",
        "source",
        "image_url",
        "original_url",
        "final_url",
        "page_url",
        "local_path",
        "relative_path",
        "file_name",
        "bytes",
        "sha256",
        "width",
        "height",
        "mime_type",
        "downloaded",
    }
    link = result.table_asset_links[0]
    assert set(link) == {
        "link_id",
        "source_table_id",
        "query_view_id",
        "row_id",
        "column_index",
        "column_name",
        "cell_text",
        "entity_id",
        "entity_wiki_title",
        "asset_ids",
    }
    assert link["entity_wiki_title"] == "Wdc e1"
    assert link["asset_ids"] == [image_asset["asset_id"]]


def test_failed_page_and_empty_page_text_leave_entity_and_link() -> None:
    for failed_page in (
        {"status": "terminal", "image_urls": [], "text": "ignored"},
        page(text=""),
    ):
        result = materialize_entity_assets(
            entity("e1"),
            failed_page,
            image_outcomes={},
        )
        assert result.entity_id == "e1"
        assert result.bridge_assets == []
        assert len(result.table_asset_links) == 1


def read_jsonl(paths: list[Path] | tuple[Path, ...]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    return records


class FakePageTransport:
    network_policy_fingerprint = "page-v1"

    def __init__(self, outcomes: dict[str, dict | BaseException]) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    def fetch_page(
        self,
        url: str,
        *,
        deadline_seconds: float,
        max_retries: int,
    ) -> dict | None:
        assert deadline_seconds > 0
        assert max_retries == 0
        self.calls.append(url)
        outcome = self.outcomes[url]
        if isinstance(outcome, BaseException):
            raise outcome
        return {
            "page_url": url,
            "final_url": url,
            "text": str(outcome.get("text") or ""),
            "image_urls": list(outcome.get("image_urls") or []),
        }


def test_disk_join_streams_task3_entities_with_task4_page_fanout(
    tmp_path: Path,
) -> None:
    successful = entity("success")
    terminal = entity("terminal")
    missing = entity("missing")
    entity_path = tmp_path / "entities.jsonl"
    entity_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in (successful, terminal, missing)
        ),
        encoding="utf-8",
    )
    transport = FakePageTransport(
        {
            successful["page_url"]: {
                "text": "durable page",
                "image_urls": ["https://i.test/page.jpg"],
            },
            terminal["page_url"]: TimeoutError("timed out"),
        }
    )
    fetched = fetch_unique_pages(
        [
            {
                "entity_id": current["entity_id"],
                "source_table_id": "st1",
                "row_id": 7,
                "page_url": current["page_url"],
                "url_key": hashlib.sha256(
                    current["page_url"].encode("utf-8")
                ).hexdigest(),
            }
            for current in (successful, terminal)
        ],
        SqliteJobStore(tmp_path / "page-jobs.sqlite3"),
        transport,
        FetchPolicy(
            retries=0,
            global_concurrency=1,
            per_host_concurrency=1,
            network_policy_fingerprint="page-v1",
        ),
    )

    joined = list(
        iter_entity_page_join(
            [entity_path],
            iter_page_fanout(
                fetched.outcomes_path,
                fetched.policy_fingerprint,
            ),
            join_path=tmp_path / "entity-page-join.sqlite3",
            commit_every=1,
        )
    )

    assert sorted(transport.calls) == sorted(
        [successful["page_url"], terminal["page_url"]]
    )
    assert [record["entity_id"] for record, _page in joined] == [
        "success",
        "terminal",
        "missing",
    ]
    assert joined[0][1]["status"] == "success"
    assert joined[0][1]["text"] == "durable page"
    assert joined[1][1]["status"] == "terminal"
    assert joined[1][1]["error_class"] == "TimeoutError"
    assert joined[2][1] is None
    with sqlite3.connect(tmp_path / "entity-page-join.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM entity_pages"
        ).fetchone() == (2,)


def test_zero_attempt_and_zero_retained_budgets_are_independent() -> None:
    image_url = "https://i.test/a.jpg"
    assert (
        plan_entity_assets(
            entity(image_urls=[image_url]),
            page(),
            ImageBudget(attempts_per_entity=0, retained_per_entity=3),
        ).image_refs
        == ()
    )
    result = materialize_entity_assets(
        entity(image_urls=[image_url]),
        page(),
        {image_url: image_outcome(image_url)},
        ImageBudget(attempts_per_entity=1, retained_per_entity=0),
    )
    assert result.bridge_assets == []


def test_planning_persists_all_mappings_before_global_dedup_and_resumes(
    tmp_path: Path,
) -> None:
    shared = "https://i.test/shared.jpg"
    entity_pages = (
        (
            entity(f"e{index}", image_urls=[shared]),
            page(image_urls=[f"https://i.test/{index}.jpg"]),
        )
        for index in range(7)
    )
    planned = persist_entity_asset_plans(
        entity_pages,
        output_root=tmp_path / "plans",
        input_fingerprint="entities-v1",
        budget=ImageBudget(2, 2),
        records_per_shard=3,
    )

    mappings = read_jsonl(planned.image_mapping_paths)
    assert len(mappings) == 14
    assert max(
        len(read_jsonl([path]))
        for path in planned.image_mapping_paths
    ) <= 3
    mtimes = {
        path: path.stat().st_mtime_ns
        for path in (
            *planned.entity_plan_paths,
            *planned.image_mapping_paths,
        )
    }

    resumed = persist_entity_asset_plans(
        (),
        output_root=tmp_path / "plans",
        input_fingerprint="entities-v1",
        budget=ImageBudget(2, 2),
        records_per_shard=3,
    )

    assert resumed == planned
    assert {
        path: path.stat().st_mtime_ns for path in mtimes
    } == mtimes

    unique_path = tmp_path / "unique" / "images.jsonl"
    completed = build_unique_image_jobs(
        planned,
        unique_path,
        chunk_records=2,
    )
    assert completed.records == 8
    assert completed.complete is True
    assert completed.manifest_path.is_file()
    unique = read_jsonl([unique_path])
    assert len(
        {
            record["url_key"]
            for record in unique
        }
    ) == 8

    mtime = unique_path.stat().st_mtime_ns
    resumed_unique = build_unique_image_jobs(
        planned,
        unique_path,
        chunk_records=2,
    )
    assert resumed_unique == completed
    assert unique_path.stat().st_mtime_ns == mtime


def test_unique_image_job_stage_rejects_corruption_and_parameter_change(
    tmp_path: Path,
) -> None:
    planned = persist_entity_asset_plans(
        [(entity(image_urls=["https://i.test/a.jpg"]), page())],
        output_root=tmp_path / "plans",
        input_fingerprint="entities-v1",
    )
    unique_path = tmp_path / "unique" / "images.jsonl"
    build_unique_image_jobs(planned, unique_path, chunk_records=2)
    unique_path.write_text('{"corrupt":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        build_unique_image_jobs(planned, unique_path, chunk_records=2)
    with pytest.raises(ValueError, match="fingerprint"):
        build_unique_image_jobs(planned, unique_path, chunk_records=3)


def test_plan_resume_rejects_a_corrupt_mapping_shard(
    tmp_path: Path,
) -> None:
    planned = persist_entity_asset_plans(
        [(entity(image_urls=["https://i.test/a.jpg"]), page())],
        output_root=tmp_path / "plans",
        input_fingerprint="entities-v1",
    )
    planned.image_mapping_paths[0].write_text(
        '{"corrupt":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum"):
        persist_entity_asset_plans(
            (),
            output_root=tmp_path / "plans",
            input_fingerprint="entities-v1",
        )


class FakeImageTransport:
    network_policy_fingerprint = "image-v1"
    max_retries = 0
    max_response_seconds = 8.0

    def __init__(
        self,
        root: Path,
        outcomes: dict[str, str | Exception],
    ) -> None:
        self.root = root
        self.outcomes = outcomes
        self.calls: list[str] = []
        self.cache: dict[str, dict] = {}

    def cached_image_outcome(self, image_url: str) -> dict | None:
        return self.cache.get(image_url)

    def download_image(
        self,
        image_url: str,
        *,
        page_url: str,
        source: str,
        entity_id: str,
    ) -> dict | None:
        del page_url, source, entity_id
        self.calls.append(image_url)
        outcome = self.outcomes[image_url]
        if isinstance(outcome, Exception):
            terminal = {
                "status": "terminal",
                "image_url": image_url,
                "error_class": type(outcome).__name__,
                "policy_fingerprint": self.network_policy_fingerprint,
            }
            self.cache[image_url] = terminal
            raise outcome
        file_path = self.root / (
            str(len(self.calls)) + ".png"
        )
        color = (10, 20, 30) if outcome == "shared" else (50, 60, 70)
        Image.new("RGB", (64, 48), color=color).save(file_path)
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        success = {
            "status": "success",
            "image_url": image_url,
            "original_url": image_url,
            "final_url": image_url + "?final=1",
            "file_name": file_path.name,
            "local_path": str(file_path),
            "relative_path": file_path.name,
            "sha256": digest,
            "width": 64,
            "height": 48,
            "mime_type": "image/png",
            "bytes": file_path.stat().st_size,
            "downloaded": True,
            "policy_fingerprint": self.network_policy_fingerprint,
        }
        self.cache[image_url] = success
        return success


class ColdStartBarrierImageTransport(FakeImageTransport):
    def __init__(self, root: Path, image_url: str) -> None:
        super().__init__(root, {image_url: "unique"})
        self.network_barrier = threading.Barrier(2)
        self.call_lock = threading.Lock()

    def download_image(
        self,
        image_url: str,
        *,
        page_url: str,
        source: str,
        entity_id: str,
    ) -> dict | None:
        with self.call_lock:
            call_index = len(self.calls)
            self.calls.append(image_url)
        try:
            self.network_barrier.wait(timeout=0.3)
        except threading.BrokenBarrierError:
            pass
        file_path = self.root / f"cold-{call_index}.png"
        Image.new("RGB", (64, 48), color=(31, 41, 59)).save(file_path)
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        return {
            "status": "success",
            "image_url": image_url,
            "original_url": image_url,
            "final_url": image_url,
            "file_name": file_path.name,
            "local_path": str(file_path),
            "relative_path": file_path.name,
            "sha256": digest,
            "width": 64,
            "height": 48,
            "mime_type": "image/png",
            "bytes": file_path.stat().st_size,
            "downloaded": True,
            "policy_fingerprint": self.network_policy_fingerprint,
        }


def write_unique_jobs(path: Path, urls: list[str]):
    planned = persist_entity_asset_plans(
        [
            (entity(f"e{index}", image_urls=[url]), page())
            for index, url in enumerate(urls)
        ],
        output_root=path.parent / f"{path.stem}-plans",
        input_fingerprint=hashlib.sha256(
            "\n".join(urls).encode("utf-8")
        ).hexdigest(),
    )
    return build_unique_image_jobs(planned, path, chunk_records=17)


def write_named_job_set(
    root: Path,
    name: str,
    urls: list[str],
) -> tuple:
    planned = persist_entity_asset_plans(
        [
            (
                entity(f"{name}-{index}", image_urls=[url]),
                page(),
            )
            for index, url in enumerate(urls)
        ],
        output_root=root / f"{name}-plans",
        input_fingerprint=f"{name}-entities-v1",
    )
    jobs = build_unique_image_jobs(
        planned,
        root / f"{name}-unique.jsonl",
        chunk_records=17,
    )
    return planned, jobs


def test_image_outcome_store_cold_open_is_process_safe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cold-outcomes.sqlite3"
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(
            target=_open_image_outcome_store_process,
            args=(str(path), results),
        )
        for _index in range(6)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert all(process.exitcode == 0 for process in processes)
    assert [results.get(timeout=1) for _process in processes] == [
        ("ok",)
    ] * len(processes)
    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }
    assert {"image_outcomes", "image_url_claims"} <= tables


def test_image_outcome_store_rolls_back_incompatible_schema_init(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incompatible-outcomes.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE image_url_claims (
                policy_fingerprint TEXT NOT NULL,
                url_key TEXT NOT NULL,
                PRIMARY KEY (policy_fingerprint, url_key)
            )
            """
        )

    with pytest.raises(ValueError, match="claim schema"):
        ImageOutcomeStore(path)

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }
    assert tables == {"image_url_claims"}


def test_cold_concurrent_job_sets_make_one_shared_url_request(
    tmp_path: Path,
) -> None:
    image_url = "https://i.test/cold-shared.jpg"
    _planned_left, jobs_left = write_named_job_set(
        tmp_path,
        "cold-left",
        [image_url],
    )
    _planned_right, jobs_right = write_named_job_set(
        tmp_path,
        "cold-right",
        [image_url],
    )
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    outcomes_path = tmp_path / "cold-outcomes.sqlite3"
    transport = ColdStartBarrierImageTransport(tmp_path, image_url)
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )
    start_barrier = threading.Barrier(2)
    results = []
    failures: list[BaseException] = []
    snapshots: dict[str, list[UrlProgressSnapshot]] = {
        "cold-left": [],
        "cold-right": [],
    }
    result_lock = threading.Lock()

    def run(name: str, current_jobs) -> None:
        try:
            start_barrier.wait(timeout=2)
            result = fetch_unique_images(
                current_jobs,
                store,
                transport,
                policy,
                outcomes_path=outcomes_path,
                image_dir=tmp_path / "content",
                progress_callback=snapshots[name].append,
                progress_callback_every=1,
            )
            with result_lock:
                results.append(result)
        except BaseException as error:
            with result_lock:
                failures.append(error)

    threads = [
        threading.Thread(target=run, args=(name, current_jobs))
        for name, current_jobs in (
            ("cold-left", jobs_left),
            ("cold-right", jobs_right),
        )
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not failures
    assert len(results) == 2
    assert all(result.complete for result in results)
    assert transport.calls == [image_url]
    final_transport_events = sorted(
        sum(items[-1].transport_event_histogram)
        for items in snapshots.values()
    )
    assert final_transport_events == [0, 1]
    assert all(items[-1].completed_durable == 1 for items in snapshots.values())
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            """
            SELECT status, COUNT(*), COUNT(DISTINCT kind)
            FROM jobs
            GROUP BY status
            """
        ).fetchall() == [("success", 2, 2)]


def test_unique_image_fetch_requests_each_url_once_and_content_addresses(
    tmp_path: Path,
) -> None:
    urls = [
        "https://i.test/shared.jpg",
        "https://i.test/shared.jpg",
        "https://i.test/other.jpg",
    ]
    unique_path = tmp_path / "unique.jsonl"
    unique_jobs = write_unique_jobs(
        unique_path,
        list(dict.fromkeys(urls)),
    )
    transport = FakeImageTransport(
        tmp_path,
        {
            "https://i.test/shared.jpg": "shared",
            "https://i.test/other.jpg": "shared",
        },
    )
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=2,
        per_host_concurrency=1,
    )

    result = fetch_unique_images(
        unique_jobs,
        SqliteJobStore(tmp_path / "jobs.sqlite3"),
        transport,
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
    )
    outcomes = list(
        iter_image_outcomes(
            result.outcomes_path,
            result.policy_fingerprint,
        )
    )

    assert len(transport.calls) == 2
    assert set(transport.calls) == {
        "https://i.test/shared.jpg",
        "https://i.test/other.jpg",
    }
    assert result.success == 2
    assert len({outcome["local_path"] for outcome in outcomes}) == 1
    assert len(list((tmp_path / "content").iterdir())) == 1
    assert result.transport_attempt_summary["transport_attempts"] == 2
    assert (
        result.transport_attempt_summary["duplicate_physical_requests"] == 0
    )
    assert (
        result.transport_attempt_summary["unfinished_transport_attempts"] == 0
    )
    manifest = json.loads(
        result.fetch_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["transport_attempts"] == result.transport_attempt_summary


def test_image_progress_callback_starts_from_durable_baseline_and_is_bounded(
    tmp_path: Path,
) -> None:
    urls = [f"https://i.test/progress-{index}.jpg" for index in range(5)]
    transport = FakeImageTransport(
        tmp_path,
        {url: f"unique-{index}" for index, url in enumerate(urls)},
    )
    jobs = SqliteJobStore(tmp_path / "jobs.sqlite3")
    outcomes_path = tmp_path / "outcomes.sqlite3"
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )

    fetch_unique_images(
        write_unique_jobs(tmp_path / "first.jsonl", urls[:1]),
        jobs,
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
    )
    updates: list[UrlProgressSnapshot] = []
    result = fetch_unique_images(
        write_unique_jobs(tmp_path / "all.jsonl", urls),
        jobs,
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
        progress_callback=updates.append,
        progress_callback_every=2,
    )

    assert result.complete
    assert [snapshot.completed_durable for snapshot in updates] == [0, 2, 4, 5]
    assert all(snapshot.total == 5 for snapshot in updates)
    assert len({snapshot.execution_epoch for snapshot in updates}) == 1


def test_image_first_completed_batch_coalesces_url_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urls = [f"https://host-{index}.test/batch.jpg" for index in range(6)]
    barrier = threading.Barrier(6)

    class BatchImageTransport(FakeImageTransport):
        def download_image(self, image_url: str, **kwargs):
            barrier.wait(timeout=5)
            return super().download_image(image_url, **kwargs)

    real_wait = assets_module.wait

    def wait_for_whole_batch(fs, *, return_when):
        assert return_when is assets_module.FIRST_COMPLETED
        done, pending = real_wait(fs)
        return done, pending

    monkeypatch.setattr(assets_module, "wait", wait_for_whole_batch)
    snapshots: list[UrlProgressSnapshot] = []
    result = fetch_unique_images(
        write_unique_jobs(tmp_path / "unique.jsonl", urls),
        SqliteJobStore(tmp_path / "jobs.sqlite3"),
        BatchImageTransport(
            tmp_path,
            {url: f"unique-{index}" for index, url in enumerate(urls)},
        ),
        FetchPolicy(
            network_policy_fingerprint="image-v1",
            policy_version="wdc200k-image-v1",
            global_concurrency=6,
            per_host_concurrency=1,
        ),
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
        progress_callback=snapshots.append,
        progress_callback_every=2,
    )

    assert result.complete
    assert [snapshot.completed_durable for snapshot in snapshots] == [0, 6]
    final = snapshots[-1]
    assert final.in_flight_jobs == 0
    assert final.finished_not_durable == 0
    assert sum(final.transport_event_histogram) == 6
    assert sum(final.commit_event_histogram) == 6


def test_image_incomplete_forced_refresh_keeps_off_milestone_for_next_epoch(
    tmp_path: Path,
) -> None:
    urls = [f"https://foreign-{index}.test/image.jpg" for index in range(448)]
    unique_jobs = write_unique_jobs(tmp_path / "unique.jsonl", urls)
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=8,
        per_host_concurrency=1,
    )
    fingerprint = assets_module._image_policy_fingerprint(policy)
    kind = assets_module._image_kind(fingerprint, unique_jobs)
    jobs = SqliteJobStore(tmp_path / "jobs.sqlite3")
    outcomes_path = tmp_path / "outcomes.sqlite3"
    all_epochs: list[list[UrlProgressSnapshot]] = []
    foreign_completed = False

    for _execution in range(32):
        snapshots: list[UrlProgressSnapshot] = []

        def callback(snapshot: UrlProgressSnapshot) -> None:
            nonlocal foreign_completed
            snapshots.append(snapshot)
            if not foreign_completed:
                leased = jobs.claim(kind, 448, "foreign", lease_seconds=3600)
                assert len(leased) == 448
                first = leased[0]
                ImageOutcomeStore(outcomes_path).put(
                    fingerprint,
                    str(first.payload["url_key"]),
                    str(first.payload["image_url"]),
                    {
                        "status": "terminal",
                        "image_url": first.payload["image_url"],
                        "error_class": "foreign",
                    },
                )
                jobs.finish(
                    first.job_id,
                    "terminal",
                    {"url_key": first.payload["url_key"]},
                    owner="foreign",
                    lease_id=first.lease_id,
                )
                foreign_completed = True

        result = fetch_unique_images(
            unique_jobs,
            jobs,
            FakeImageTransport(tmp_path, {}),
            policy,
            outcomes_path=outcomes_path,
            image_dir=tmp_path / "content",
            progress_callback=callback,
            progress_callback_every=2,
        )
        assert not result.complete
        all_epochs.append(snapshots)

    assert [len(epoch) for epoch in all_epochs] == [1] * 32
    assert [epoch[0].completed_durable for epoch in all_epochs] == [0] + [1] * 31
    assert sum(len(epoch) - 1 for epoch in all_epochs) <= 224
    assert len(all_epochs) <= 32
    assert sum(map(len, all_epochs)) <= 256


def test_image_tracker_distinguishes_physical_and_synchronous_cache_paths(
    tmp_path: Path,
) -> None:
    physical = "https://i.test/physical.jpg"
    failed = "https://i.test/failed.jpg"
    cached = "https://i.test/cached.jpg"
    unique_jobs = write_unique_jobs(
        tmp_path / "unique.jsonl",
        [physical, failed, cached],
    )
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=3,
        per_host_concurrency=3,
    )
    fingerprint = assets_module._image_policy_fingerprint(policy)
    outcomes_path = tmp_path / "outcomes.sqlite3"
    ImageOutcomeStore(outcomes_path).put(
        fingerprint,
        hashlib.sha256(cached.encode("utf-8")).hexdigest(),
        cached,
        {
            "status": "terminal",
            "image_url": cached,
            "original_url": cached,
            "error_class": "cached",
        },
    )
    snapshots: list[UrlProgressSnapshot] = []
    result = fetch_unique_images(
        unique_jobs,
        SqliteJobStore(tmp_path / "jobs.sqlite3"),
        FakeImageTransport(
            tmp_path,
            {physical: "unique", failed: TimeoutError("deadline")},
        ),
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
        progress_callback=snapshots.append,
        progress_callback_every=1,
    )

    assert result.complete
    final = snapshots[-1]
    assert final.completed_durable == final.total == 3
    assert sum(final.transport_event_histogram) == 2
    assert sum(final.commit_event_histogram) == 2
    assert final.physical_in_flight == 0


def test_image_claim_loss_reconciles_another_claimants_durable_outcome(
    tmp_path: Path,
) -> None:
    image_url = "https://i.test/claim-loss.jpg"
    unique_jobs = write_unique_jobs(tmp_path / "unique.jsonl", [image_url])
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )
    fingerprint = assets_module._image_policy_fingerprint(policy)
    outcomes_path = tmp_path / "outcomes.sqlite3"
    url_key = hashlib.sha256(image_url.encode("utf-8")).hexdigest()

    def steal_and_finish(lease) -> None:
        store = ImageOutcomeStore(outcomes_path)
        decision = store.claim_url(
            fingerprint,
            url_key,
            owner="other-claimant",
            lease_seconds=60,
            now=lease.lease_until + 1.0,
        )
        assert decision.lease is not None
        store.put_claimed(
            fingerprint,
            url_key,
            image_url,
            {
                "status": "terminal",
                "image_url": image_url,
                "original_url": image_url,
                "error_class": "other_claimant",
            },
            lease=decision.lease,
            now=lease.lease_until + 2.0,
        )
        assert store.finish_claim(
            fingerprint,
            url_key,
            lease=decision.lease,
        )

    snapshots: list[UrlProgressSnapshot] = []
    transport = FakeImageTransport(tmp_path, {})
    result = fetch_unique_images(
        unique_jobs,
        SqliteJobStore(tmp_path / "jobs.sqlite3"),
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
        after_url_claim=steal_and_finish,
        progress_callback=snapshots.append,
        progress_callback_every=1,
    )

    assert result.complete
    assert transport.calls == []
    assert snapshots[-1].completed_durable == 1
    assert sum(snapshots[-1].transport_event_histogram) == 0
    assert sum(snapshots[-1].commit_event_histogram) == 1


def test_image_url_refresh_uses_kind_status_and_outcome_key_indexes(
    tmp_path: Path,
) -> None:
    jobs = SqliteJobStore(tmp_path / "jobs.sqlite3")
    outcomes_path = tmp_path / "outcomes.sqlite3"
    ImageOutcomeStore(outcomes_path)
    kind = "wdc200k-image:policy:set"
    jobs.enqueue(kind, "job", {"url_key": "a" * 64})
    counts = assets_module._refresh_image_url_counts(
        jobs, kind, outcomes_path, "policy"
    )
    assert (counts.completed, counts.pending, counts.leased, counts.total) == (
        0,
        1,
        0,
        1,
    )
    with jobs._connect() as connection:
        connection.execute(
            "ATTACH DATABASE ? AS image_progress_outcomes",
            (str(outcomes_path),),
        )
        status_plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT status, COUNT(*) FROM jobs "
            "WHERE kind = ? GROUP BY status",
            (kind,),
        ).fetchall()
        completed_plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM jobs AS job "
            "JOIN image_progress_outcomes.image_outcomes AS outcome "
            "ON outcome.policy_fingerprint = ? "
            "AND outcome.url_key = json_extract(job.payload_json, '$.url_key') "
            "WHERE job.kind = ? AND job.status IN ('success', 'terminal') "
            "AND outcome.status IN ('success', 'terminal')",
            ("policy", kind),
        ).fetchall()
    details = [str(row[3]) for row in status_plan + completed_plan]
    assert not any("SCAN jobs" in detail for detail in details)
    assert any("jobs_kind_status" in detail for detail in details)
    assert any(
        "sqlite_autoindex_image_outcomes_1" in detail for detail in details
    )


def test_image_url_refresh_uses_one_snapshot_during_external_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = SqliteJobStore(tmp_path / "jobs.sqlite3")
    outcomes_path = tmp_path / "outcomes.sqlite3"
    outcomes = ImageOutcomeStore(outcomes_path)
    kind = "wdc200k-image:policy:set"
    policy = "policy"
    image_url = "https://i.test/concurrent.jpg"
    url_key = hashlib.sha256(image_url.encode("utf-8")).hexdigest()
    jobs.enqueue(kind, "job", {"url_key": url_key})
    claimed = jobs.claim(kind, 1, "other", lease_seconds=60)[0]
    status_query_started = threading.Event()
    writer_committed = threading.Event()
    refresh_closed = threading.Event()
    writer_errors: list[BaseException] = []
    original_connect = jobs._connect

    class BarrierConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def execute(self, sql: str, parameters=()):
            cursor = self.connection.execute(sql, parameters)
            normalized = " ".join(sql.casefold().split())
            if normalized.startswith("select status, count(*) as count"):
                status_query_started.set()
                assert writer_committed.wait(timeout=5)
            return cursor

        def close(self) -> None:
            refresh_closed.set()
            self.connection.close()

        def __getattr__(self, name: str):
            return getattr(self.connection, name)

        def __enter__(self):
            self.connection.__enter__()
            return self

        def __exit__(self, *args):
            return self.connection.__exit__(*args)

    monkeypatch.setattr(
        jobs,
        "_connect",
        lambda: BarrierConnection(original_connect()),
    )

    def finish_elsewhere() -> None:
        try:
            assert status_query_started.wait(timeout=5)
            outcomes.put(
                policy,
                url_key,
                image_url,
                {
                    "status": "terminal",
                    "image_url": image_url,
                    "original_url": image_url,
                    "error_class": "done",
                },
            )
            with original_connect() as connection:
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'terminal', owner = NULL,
                        lease_expires = NULL, lease_id = NULL
                    WHERE job_id = ? AND lease_id = ?
                    """,
                    (claimed.job_id, claimed.lease_id),
                )
                connection.commit()
        except BaseException as error:
            writer_errors.append(error)
        finally:
            writer_committed.set()

    writer = threading.Thread(target=finish_elsewhere)
    writer.start()
    try:
        counts = assets_module._refresh_image_url_counts(
            jobs, kind, outcomes_path, policy
        )
    finally:
        writer.join(timeout=5)

    assert not writer_errors
    assert not writer.is_alive()
    assert refresh_closed.is_set()
    assert (counts.completed, counts.pending, counts.leased, counts.total) == (
        0,
        0,
        1,
        1,
    )


def test_unique_image_fetch_uses_real_wdc_client_cache_contract(
    tmp_path: Path,
) -> None:
    body_path = tmp_path / "body.png"
    Image.new("RGB", (64, 48), color=(9, 19, 29)).save(body_path)
    body = body_path.read_bytes()

    class RasterResponse:
        status_code = 200
        headers = {"Content-Type": "image/png"}
        encoding = "utf-8"

        def __init__(self, url: str) -> None:
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def close(self) -> None:
            return None

        def iter_content(self, chunk_size: int):
            del chunk_size
            yield body

    class RasterSession:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}
            self.calls: list[str] = []

        def get(self, url: str, **_kwargs):
            self.calls.append(url)
            return RasterResponse(url + "?final=1")

    urls = ["https://i.test/a.png", "https://i.test/b.png"]
    unique_path = tmp_path / "unique.jsonl"
    unique_jobs = write_unique_jobs(unique_path, urls)
    session = RasterSession()
    client = WdcWebClient(
        tmp_path / "web",
        session=session,
        host_delay=0,
        max_retries=0,
        max_response_seconds=8,
        min_free_disk_bytes=0,
        network_policy_version="image-v1",
    )
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )

    fetched = fetch_unique_images(
        unique_jobs,
        SqliteJobStore(tmp_path / "jobs.sqlite3"),
        client,
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=client.image_dir,
    )
    outcomes = list(
        iter_image_outcomes(
            fetched.outcomes_path,
            fetched.policy_fingerprint,
        )
    )

    assert fetched.success == 2
    assert len(session.calls) == 2
    assert set(session.calls) == set(urls)
    assert len({outcome["local_path"] for outcome in outcomes}) == 1
    assert {
        outcome["image_url"]: outcome["original_url"]
        for outcome in outcomes
    } == {image_url: image_url for image_url in urls}
    assert len(list(client.image_dir.glob("image_*"))) == 1


def test_terminal_image_outcome_is_not_replayed_on_resume(
    tmp_path: Path,
) -> None:
    image_url = "https://i.test/broken.jpg"
    unique_path = tmp_path / "unique.jsonl"
    unique_jobs = write_unique_jobs(unique_path, [image_url])
    transport = FakeImageTransport(
        tmp_path,
        {image_url: TimeoutError()},
    )
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")

    first = fetch_unique_images(
        unique_jobs,
        store,
        transport,
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
    )
    second = fetch_unique_images(
        unique_jobs,
        store,
        transport,
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
    )

    assert first.terminal == second.terminal == 1
    assert transport.calls == [image_url]


def test_image_fetch_isolates_overlapping_job_sets_and_scopes_outcomes(
    tmp_path: Path,
) -> None:
    a = "https://i.test/a.jpg"
    b = "https://i.test/b.jpg"
    c = "https://other.test/c.jpg"
    planned_ab, jobs_ab = write_named_job_set(
        tmp_path,
        "ab",
        [a, b],
    )
    planned_a, jobs_a = write_named_job_set(tmp_path, "a", [a])
    planned_c, jobs_c = write_named_job_set(tmp_path, "c", [c])
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    outcomes_path = tmp_path / "outcomes.sqlite3"
    transport = FakeImageTransport(
        tmp_path,
        {a: "unique", b: "unique", c: "unique"},
    )
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=2,
        per_host_concurrency=1,
    )

    fetched_ab = fetch_unique_images(
        jobs_ab,
        store,
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
    )
    ab_manifest_mtime = fetched_ab.fetch_manifest_path.stat().st_mtime_ns
    fetched_a = fetch_unique_images(
        jobs_a,
        store,
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
    )
    a_manifest_mtime = fetched_a.fetch_manifest_path.stat().st_mtime_ns
    resumed_ab = fetch_unique_images(
        jobs_ab,
        store,
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
    )
    fetched_c = fetch_unique_images(
        jobs_c,
        store,
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
    )
    final_ab = fetch_unique_images(
        jobs_ab,
        store,
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
    )

    assert transport.calls.count(a) == 1
    assert transport.calls.count(b) == 1
    assert transport.calls.count(c) == 1
    assert fetched_ab.complete is fetched_a.complete is True
    assert fetched_c.complete is final_ab.complete is True
    assert (fetched_ab.unique, fetched_ab.outcomes_count) == (2, 2)
    assert (fetched_a.unique, fetched_a.outcomes_count) == (1, 1)
    assert (fetched_c.unique, fetched_c.outcomes_count) == (1, 1)
    assert final_ab.outcome_digest == fetched_ab.outcome_digest
    assert len(
        {
            fetched_ab.fetch_manifest_path,
            fetched_a.fetch_manifest_path,
            fetched_c.fetch_manifest_path,
        }
    ) == 3
    assert fetched_ab.fetch_manifest_path.stat().st_mtime_ns == (
        ab_manifest_mtime
    )
    assert fetched_a.fetch_manifest_path.stat().st_mtime_ns == (
        a_manifest_mtime
    )
    assert resumed_ab.fetch_manifest_sha256 == (
        fetched_ab.fetch_manifest_sha256
    )
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM jobs"
        ).fetchone() == (4,)
        assert connection.execute(
            """
            SELECT COUNT(DISTINCT kind), COUNT(DISTINCT job_id)
            FROM jobs
            """
        ).fetchone() == (3, 4)

    for name, planned, fetched, expected_urls in (
        ("ab", planned_ab, final_ab, {a, b}),
        ("a", planned_a, fetched_a, {a}),
        ("c", planned_c, fetched_c, {c}),
    ):
        materialized = materialize_asset_shards(
            planned,
            fetch_result=fetched,
            output_root=tmp_path / f"{name}-materialized",
            input_fingerprint=f"{name}-assets-v1",
        )
        assets = read_jsonl(materialized.bridge_asset_paths)
        assert {
            record["image_url"]
            for record in assets
            if record["asset_type"] == "image"
        } == expected_urls

    c_key = hashlib.sha256(c.encode("utf-8")).hexdigest()
    with sqlite3.connect(outcomes_path) as connection:
        connection.execute(
            """
            DELETE FROM image_outcomes
            WHERE policy_fingerprint = ? AND url_key = ?
            """,
            (fetched_c.policy_fingerprint, c_key),
        )
    incomplete_c = fetch_unique_images(
        jobs_c,
        store,
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
    )
    assert incomplete_c.complete is False
    assert incomplete_c.outcomes_count == 0
    assert incomplete_c.remaining == 1
    assert transport.calls.count(c) == 1
    assert materialize_asset_shards(
        planned_ab,
        fetch_result=final_ab,
        output_root=tmp_path / "ab-materialized",
        input_fingerprint="ab-assets-v1",
    )
    with pytest.raises(ValueError, match="not complete"):
        materialize_asset_shards(
            planned_c,
            fetch_result=incomplete_c,
            output_root=tmp_path / "c-incomplete-materialized",
            input_fingerprint="c-assets-incomplete",
        )


def test_image_fetch_expands_to_overlapping_larger_job_set(
    tmp_path: Path,
) -> None:
    a = "https://i.test/grow-a.jpg"
    b = "https://i.test/grow-b.jpg"
    planned_a, jobs_a = write_named_job_set(tmp_path, "small", [a])
    planned_ab, jobs_ab = write_named_job_set(
        tmp_path,
        "large",
        [a, b],
    )
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    transport = FakeImageTransport(
        tmp_path,
        {a: "unique", b: "unique"},
    )
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
    )
    outcomes_path = tmp_path / "outcomes.sqlite3"

    fetched_a = fetch_unique_images(
        jobs_a,
        store,
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
    )
    fetched_ab = fetch_unique_images(
        jobs_ab,
        store,
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
    )

    assert fetched_a.complete is fetched_ab.complete is True
    assert (fetched_a.unique, fetched_a.outcomes_count) == (1, 1)
    assert (fetched_ab.unique, fetched_ab.outcomes_count) == (2, 2)
    assert transport.calls.count(a) == 1
    assert transport.calls.count(b) == 1
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT kind) FROM jobs"
        ).fetchone() == (3, 2)
    materialized = materialize_asset_shards(
        planned_ab,
        fetch_result=fetched_ab,
        output_root=tmp_path / "large-materialized",
        input_fingerprint="large-assets-v1",
    )
    assert {
        record["image_url"]
        for record in read_jsonl(materialized.bridge_asset_paths)
        if record["asset_type"] == "image"
    } == {a, b}
    assert materialize_asset_shards(
        planned_a,
        fetch_result=fetched_a,
        output_root=tmp_path / "small-materialized",
        input_fingerprint="small-assets-v1",
    )


def test_terminal_outcome_repairs_each_overlapping_job_set_without_network(
    tmp_path: Path,
) -> None:
    broken = "https://i.test/broken.jpg"
    _planned_first, jobs_first = write_named_job_set(
        tmp_path,
        "first",
        [broken],
    )
    _planned_second, jobs_second = write_named_job_set(
        tmp_path,
        "second",
        [broken],
    )
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    transport = FakeImageTransport(
        tmp_path,
        {broken: TimeoutError()},
    )
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
    )

    first = fetch_unique_images(
        jobs_first,
        store,
        transport,
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
    )
    second = fetch_unique_images(
        jobs_second,
        store,
        transport,
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
    )

    assert first.complete is second.complete is True
    assert first.terminal == second.terminal == 1
    assert first.job_kind != second.job_kind
    assert first.fetch_manifest_path != second.fetch_manifest_path
    assert transport.calls == [broken]
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            """
            SELECT status, COUNT(*)
            FROM jobs
            GROUP BY status
            """
        ).fetchall() == [("terminal", 2)]


def test_concurrent_overlapping_job_sets_repair_independent_jobs(
    tmp_path: Path,
) -> None:
    image_url = "https://i.test/shared-concurrent.jpg"
    _planned_seed, jobs_seed = write_named_job_set(
        tmp_path,
        "seed",
        [image_url],
    )
    _planned_left, jobs_left = write_named_job_set(
        tmp_path,
        "left",
        [image_url],
    )
    _planned_right, jobs_right = write_named_job_set(
        tmp_path,
        "right",
        [image_url],
    )
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    transport = FakeImageTransport(
        tmp_path,
        {image_url: "unique"},
    )
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )
    outcomes_path = tmp_path / "outcomes.sqlite3"
    fetch_unique_images(
        jobs_seed,
        store,
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
    )
    barrier = threading.Barrier(2)
    results: list = []
    failures: list[BaseException] = []
    result_lock = threading.Lock()

    def run(current_jobs) -> None:
        try:
            barrier.wait(timeout=2)
            result = fetch_unique_images(
                current_jobs,
                store,
                transport,
                policy,
                outcomes_path=outcomes_path,
                image_dir=tmp_path / "content",
            )
            with result_lock:
                results.append(result)
        except BaseException as error:
            with result_lock:
                failures.append(error)

    threads = [
        threading.Thread(target=run, args=(current_jobs,))
        for current_jobs in (jobs_left, jobs_right)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not failures
    assert len(results) == 2
    assert all(result.complete for result in results)
    assert len({result.job_kind for result in results}) == 2
    assert len({result.fetch_manifest_path for result in results}) == 2
    assert transport.calls == [image_url]
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT kind) FROM jobs"
        ).fetchone() == (3, 3)


def test_image_outcome_stream_detects_payload_checksum_corruption(
    tmp_path: Path,
) -> None:
    image_url = "https://i.test/broken.jpg"
    unique_path = tmp_path / "unique.jsonl"
    unique_jobs = write_unique_jobs(unique_path, [image_url])
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )
    fetched = fetch_unique_images(
        unique_jobs,
        SqliteJobStore(tmp_path / "jobs.sqlite3"),
        FakeImageTransport(tmp_path, {image_url: TimeoutError()}),
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
    )
    with sqlite3.connect(fetched.outcomes_path) as connection:
        connection.execute(
            """
            UPDATE image_outcomes
            SET outcome_json = '{"status":"terminal"}'
            """
        )

    with pytest.raises(ValueError, match="checksum"):
        list(
            iter_image_outcomes(
                fetched.outcomes_path,
                fetched.policy_fingerprint,
            )
        )


def test_crash_after_image_outcome_write_repairs_job_without_download(
    tmp_path: Path,
) -> None:
    image_url = "https://i.test/durable.jpg"
    unique_path = tmp_path / "unique.jsonl"
    unique_jobs = write_unique_jobs(unique_path, [image_url])
    transport = FakeImageTransport(tmp_path, {image_url: "unique"})
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")

    with pytest.raises(RuntimeError, match="simulated crash"):
        fetch_unique_images(
            unique_jobs,
            store,
            transport,
            policy,
            outcomes_path=tmp_path / "outcomes.sqlite3",
            image_dir=tmp_path / "content",
            lease_seconds=-1,
            after_cache_write=lambda _outcome: (
                _ for _ in ()
            ).throw(RuntimeError("simulated crash")),
        )

    resumed = fetch_unique_images(
        unique_jobs,
        store,
        transport,
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
    )
    assert resumed.success == 1
    assert transport.calls == [image_url]
    with sqlite3.connect(resumed.outcomes_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM image_url_claims"
        ).fetchone() == (0,)


def test_keyboard_interrupt_drains_finished_future_and_immediately_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_urls = [
        "https://i.test/interrupt-a.jpg",
        "https://i.test/interrupt-b.jpg",
    ]
    unique_jobs = write_unique_jobs(tmp_path / "unique.jsonl", image_urls)
    jobs = SqliteJobStore(tmp_path / "jobs.sqlite3")
    outcomes_path = tmp_path / "outcomes.sqlite3"
    transport = FakeImageTransport(
        tmp_path,
        {image_url: image_url for image_url in image_urls},
    )
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )
    real_wait = assets_module.wait
    interrupted = False

    def interrupt_after_worker_finishes(fs, *, return_when):
        nonlocal interrupted
        completed, pending = real_wait(fs)
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("synthetic image scheduler interrupt")
        return completed, pending

    monkeypatch.setattr(assets_module, "wait", interrupt_after_worker_finishes)
    with pytest.raises(KeyboardInterrupt, match="scheduler interrupt"):
        fetch_unique_images(
            unique_jobs,
            jobs,
            transport,
            policy,
            outcomes_path=outcomes_path,
            image_dir=tmp_path / "content",
            claim_buffer=2,
            lease_seconds=3600,
            url_claim_lease_seconds=3600,
        )

    assert len(transport.calls) == 1
    with sqlite3.connect(jobs.path) as connection:
        assert connection.execute(
            "SELECT status, COUNT(*) FROM jobs GROUP BY status ORDER BY status"
        ).fetchall() == [("retryable", 1), ("success", 1)]
    with sqlite3.connect(outcomes_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM image_url_claims"
        ).fetchone() == (0,)

    monkeypatch.setattr(assets_module, "wait", real_wait)
    resumed = fetch_unique_images(
        unique_jobs,
        jobs,
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
        claim_buffer=2,
        lease_seconds=3600,
        url_claim_lease_seconds=3600,
    )

    assert resumed.complete
    assert sorted(transport.calls) == sorted(image_urls)
    assert resumed.transport_attempt_summary["transport_attempts"] == 2
    assert resumed.transport_attempt_summary["duplicate_physical_requests"] == 0
    assert resumed.transport_attempt_summary["unfinished_transport_attempts"] == 0


def test_keyboard_interrupt_drains_worker_completed_during_executor_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_urls = [
        "https://i.test/shutdown-a.jpg",
        "https://i.test/shutdown-b.jpg",
    ]
    worker_started = threading.Event()
    shutdown_started = threading.Event()
    release_worker = threading.Event()

    class ShutdownTransport(FakeImageTransport):
        def download_image(self, image_url: str, **kwargs):
            worker_started.set()
            assert release_worker.wait(timeout=5)
            assert shutdown_started.is_set()
            return super().download_image(image_url, **kwargs)

    executor_type = assets_module.ThreadPoolExecutor

    class ShutdownObservedExecutor(executor_type):
        def shutdown(self, wait=True, *, cancel_futures=False):
            shutdown_started.set()
            release_worker.set()
            return super().shutdown(
                wait=wait,
                cancel_futures=cancel_futures,
            )

    unique_jobs = write_unique_jobs(tmp_path / "unique.jsonl", image_urls)
    jobs = SqliteJobStore(tmp_path / "jobs.sqlite3")
    outcomes_path = tmp_path / "outcomes.sqlite3"
    transport = ShutdownTransport(
        tmp_path,
        {image_url: image_url for image_url in image_urls},
    )
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )
    real_wait = assets_module.wait

    def interrupt_before_worker_finishes(fs, *, return_when):
        assert worker_started.wait(timeout=5)
        assert not shutdown_started.is_set()
        assert all(not future.done() for future in fs)
        raise KeyboardInterrupt("interrupt before executor shutdown")

    monkeypatch.setattr(assets_module, "wait", interrupt_before_worker_finishes)
    monkeypatch.setattr(
        assets_module,
        "ThreadPoolExecutor",
        ShutdownObservedExecutor,
    )
    with pytest.raises(KeyboardInterrupt, match="before executor shutdown"):
        fetch_unique_images(
            unique_jobs,
            jobs,
            transport,
            policy,
            outcomes_path=outcomes_path,
            image_dir=tmp_path / "content",
            claim_buffer=2,
            lease_seconds=3600,
            url_claim_lease_seconds=3600,
    )

    assert shutdown_started.is_set()
    assert len(transport.calls) == 1
    assert transport.calls[0] in image_urls
    with sqlite3.connect(jobs.path) as connection:
        assert connection.execute(
            "SELECT status, COUNT(*) FROM jobs GROUP BY status ORDER BY status"
        ).fetchall() == [("retryable", 1), ("success", 1)]
    with sqlite3.connect(outcomes_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM image_url_claims"
        ).fetchone() == (0,)

    monkeypatch.setattr(assets_module, "wait", real_wait)
    monkeypatch.setattr(assets_module, "ThreadPoolExecutor", executor_type)
    resumed = fetch_unique_images(
        unique_jobs,
        jobs,
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
        claim_buffer=2,
        lease_seconds=3600,
        url_claim_lease_seconds=3600,
    )

    assert resumed.complete
    assert sorted(transport.calls) == sorted(image_urls)
    assert resumed.transport_attempt_summary["duplicate_physical_requests"] == 0
    assert resumed.transport_attempt_summary["unfinished_transport_attempts"] == 0


@pytest.mark.parametrize("cleanup_target", ["job", "url"])
def test_image_cleanup_failure_preserves_root_and_foreign_leases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_target: str,
) -> None:
    image_url = "https://i.test/cleanup-root.jpg"
    unique_jobs = write_unique_jobs(tmp_path / "unique.jsonl", [image_url])
    jobs = SqliteJobStore(tmp_path / "jobs.sqlite3")
    jobs.enqueue("foreign-kind", "foreign-job", {"url": "foreign"})
    foreign_job = jobs.claim(
        "foreign-kind",
        limit=1,
        owner="foreign-execution",
        lease_seconds=3600,
    )[0]
    outcomes_path = tmp_path / "outcomes.sqlite3"
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )
    fingerprint = assets_module._image_policy_fingerprint(policy)
    kind = assets_module._image_kind(fingerprint, unique_jobs)
    foreign_url_key = "f" * 64
    outcome_store = ImageOutcomeStore(outcomes_path)
    foreign_url_lease = outcome_store.claim_url(
        fingerprint,
        foreign_url_key,
        owner="foreign-execution:foreign-job:foreign-lease",
        lease_seconds=3600,
    ).lease
    assert foreign_url_lease is not None

    if cleanup_target == "job":
        original_release = SqliteJobStore.release_owner_leases

        def fail_current_job_release(self, release_kind, *, owner):
            if self.path == jobs.path and release_kind == kind:
                with self._connect() as connection:
                    assert connection.execute(
                        "SELECT COUNT(*) FROM jobs WHERE kind = ? "
                        "AND status = 'leased' AND owner = ?",
                        (release_kind, owner),
                    ).fetchone() == (1,)
                raise OSError("current job cleanup sentinel")
            return original_release(self, release_kind, owner=owner)

        monkeypatch.setattr(
            SqliteJobStore,
            "release_owner_leases",
            fail_current_job_release,
        )
    else:
        original_release = ImageOutcomeStore.release_claims_by_owner_prefix

        def fail_current_url_release(
            self,
            policy_fingerprint,
            *,
            owner_prefix,
        ):
            if self.path == outcomes_path and policy_fingerprint == fingerprint:
                with self._connect() as connection:
                    assert connection.execute(
                        "SELECT COUNT(*) FROM image_url_claims "
                        "WHERE policy_fingerprint = ? "
                        "AND substr(owner, 1, length(?)) = ?",
                        (policy_fingerprint, owner_prefix, owner_prefix),
                    ).fetchone() == (1,)
                raise OSError("current URL cleanup sentinel")
            return original_release(
                self,
                policy_fingerprint,
                owner_prefix=owner_prefix,
            )

        monkeypatch.setattr(
            ImageOutcomeStore,
            "release_claims_by_owner_prefix",
            fail_current_url_release,
        )

    transport = FakeImageTransport(tmp_path, {image_url: "unused"})
    with pytest.raises(RuntimeError, match="root scheduler sentinel") as caught:
        fetch_unique_images(
            unique_jobs,
            jobs,
            transport,
            policy,
            outcomes_path=outcomes_path,
            image_dir=tmp_path / "content",
            lease_seconds=3600,
            url_claim_lease_seconds=3600,
            after_url_claim=lambda _lease: (
                _ for _ in ()
            ).throw(RuntimeError("root scheduler sentinel")),
        )

    assert "cleanup sentinel" not in str(caught.value)
    assert transport.calls == []
    with sqlite3.connect(jobs.path) as connection:
        assert connection.execute(
            "SELECT status, owner, lease_id FROM jobs WHERE job_id = ?",
            (foreign_job.job_id,),
        ).fetchone() == (
            "leased",
            "foreign-execution",
            foreign_job.lease_id,
        )
        assert connection.execute(
            "SELECT status FROM jobs WHERE kind = ?",
            (kind,),
        ).fetchone() == (
            "leased" if cleanup_target == "job" else "retryable",
        )
    with sqlite3.connect(outcomes_path) as connection:
        claims = connection.execute(
            "SELECT url_key, owner FROM image_url_claims ORDER BY url_key"
        ).fetchall()
    assert (foreign_url_key, foreign_url_lease.owner) in claims
    assert len(claims) == (2 if cleanup_target == "url" else 1)


def test_image_commit_failure_drains_other_completed_future_without_refetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_commit_url = "https://failed.test/commit.jpg"
    other_url = "https://other.test/commit.jpg"
    image_urls = [failed_commit_url, other_url]
    unique_jobs = write_unique_jobs(tmp_path / "unique.jsonl", image_urls)
    jobs = SqliteJobStore(tmp_path / "jobs.sqlite3")
    outcomes_path = tmp_path / "outcomes.sqlite3"
    transport = FakeImageTransport(
        tmp_path,
        {image_url: image_url for image_url in image_urls},
    )
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=2,
        per_host_concurrency=1,
    )
    real_wait = assets_module.wait

    def both_done_failed_commit_first(fs, *, return_when):
        completed, pending = real_wait(fs)
        return (
            sorted(
                completed,
                key=lambda future: (
                    future.result().outcome["image_url"]
                    != failed_commit_url
                ),
            ),
            pending,
        )

    def fail_selected_cache_callback(outcome: dict) -> None:
        if outcome["image_url"] == failed_commit_url:
            raise RuntimeError("selected image commit sentinel")

    def fail_progress_after_completion(snapshot: UrlProgressSnapshot) -> None:
        if snapshot.completed_durable:
            raise ValueError("later image progress sentinel")

    monkeypatch.setattr(assets_module, "wait", both_done_failed_commit_first)
    with pytest.raises(RuntimeError, match="selected image commit sentinel"):
        fetch_unique_images(
            unique_jobs,
            jobs,
            transport,
            policy,
            outcomes_path=outcomes_path,
            image_dir=tmp_path / "content",
            claim_buffer=2,
            lease_seconds=3600,
            url_claim_lease_seconds=3600,
            after_cache_write=fail_selected_cache_callback,
            progress_callback=fail_progress_after_completion,
            progress_callback_every=1,
        )

    fingerprint = assets_module._image_policy_fingerprint(policy)
    outcome_store = ImageOutcomeStore(outcomes_path)
    assert len(list(outcome_store.iter(fingerprint))) == 2

    monkeypatch.setattr(assets_module, "wait", real_wait)
    resumed = fetch_unique_images(
        unique_jobs,
        jobs,
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
        claim_buffer=2,
        lease_seconds=3600,
        url_claim_lease_seconds=3600,
    )

    assert resumed.complete
    assert sorted(transport.calls) == sorted(image_urls)
    assert resumed.transport_attempt_summary["duplicate_physical_requests"] == 0
    assert resumed.transport_attempt_summary["unfinished_transport_attempts"] == 0


def test_completed_future_error_precedence_uses_submission_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_urls = [
        "https://first.test/precedence.jpg",
        "https://second.test/precedence.jpg",
    ]
    unique_jobs = write_unique_jobs(tmp_path / "unique.jsonl", image_urls)
    jobs = SqliteJobStore(tmp_path / "jobs.sqlite3")
    outcomes_path = tmp_path / "outcomes.sqlite3"
    transport = FakeImageTransport(
        tmp_path,
        {image_url: image_url for image_url in image_urls},
    )
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=2,
        per_host_concurrency=1,
    )
    real_wait = assets_module.wait
    submitted_urls: list[str] = []

    def reverse_completed_submission_order(fs, *, return_when):
        submitted = list(fs)
        completed, pending = real_wait(submitted)
        assert not pending
        submitted_urls.extend(
            str(future.result().outcome["image_url"])
            for future in submitted
        )
        return list(reversed(submitted)), pending

    def fail_by_submission_order(outcome: dict) -> None:
        submission_index = submitted_urls.index(str(outcome["image_url"]))
        if submission_index == 0:
            raise RuntimeError("earlier submitted image sentinel")
        raise ValueError("later submitted image sentinel")

    monkeypatch.setattr(
        assets_module,
        "wait",
        reverse_completed_submission_order,
    )
    with pytest.raises(RuntimeError, match="earlier submitted image sentinel"):
        fetch_unique_images(
            unique_jobs,
            jobs,
            transport,
            policy,
            outcomes_path=outcomes_path,
            image_dir=tmp_path / "content",
            claim_buffer=2,
            lease_seconds=3600,
            url_claim_lease_seconds=3600,
            after_cache_write=fail_by_submission_order,
        )

    assert len(submitted_urls) == 2
    fingerprint = assets_module._image_policy_fingerprint(policy)
    assert len(list(ImageOutcomeStore(outcomes_path).iter(fingerprint))) == 2

    monkeypatch.setattr(assets_module, "wait", real_wait)
    resumed = fetch_unique_images(
        unique_jobs,
        jobs,
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
        claim_buffer=2,
        lease_seconds=3600,
        url_claim_lease_seconds=3600,
    )

    assert resumed.complete
    assert sorted(transport.calls) == sorted(image_urls)
    assert resumed.transport_attempt_summary["duplicate_physical_requests"] == 0
    assert resumed.transport_attempt_summary["unfinished_transport_attempts"] == 0


def test_url_claim_crash_before_request_expires_and_reclaims(
    tmp_path: Path,
) -> None:
    image_url = "https://i.test/claim-crash.jpg"
    unique_jobs = write_unique_jobs(
        tmp_path / "unique.jsonl",
        [image_url],
    )
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    transport = FakeImageTransport(
        tmp_path,
        {image_url: "unique"},
    )
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
    )

    with pytest.raises(RuntimeError, match="after claim"):
        fetch_unique_images(
            unique_jobs,
            store,
            transport,
            policy,
            outcomes_path=tmp_path / "outcomes.sqlite3",
            image_dir=tmp_path / "content",
            lease_seconds=-1,
            url_claim_lease_seconds=0.05,
            url_claim_poll_seconds=0.005,
            after_url_claim=lambda _lease: (
                _ for _ in ()
            ).throw(RuntimeError("after claim")),
        )
    assert transport.calls == []
    time.sleep(0.06)

    resumed = fetch_unique_images(
        unique_jobs,
        store,
        transport,
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
        url_claim_lease_seconds=0.05,
        url_claim_poll_seconds=0.005,
    )

    assert resumed.complete is True
    assert resumed.success == 1
    assert transport.calls == [image_url]


def test_terminal_outcome_before_claim_finish_repairs_without_request(
    tmp_path: Path,
) -> None:
    image_url = "https://i.test/terminal-claim-crash.jpg"
    unique_jobs = write_unique_jobs(
        tmp_path / "unique.jsonl",
        [image_url],
    )
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    transport = FakeImageTransport(
        tmp_path,
        {image_url: TimeoutError()},
    )
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
    )

    with pytest.raises(RuntimeError, match="after outcome"):
        fetch_unique_images(
            unique_jobs,
            store,
            transport,
            policy,
            outcomes_path=tmp_path / "outcomes.sqlite3",
            image_dir=tmp_path / "content",
            lease_seconds=-1,
            url_claim_lease_seconds=60,
            after_cache_write=lambda _outcome: (
                _ for _ in ()
            ).throw(RuntimeError("after outcome")),
        )

    resumed = fetch_unique_images(
        unique_jobs,
        store,
        transport,
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
    )

    assert resumed.complete is True
    assert resumed.terminal == 1
    assert transport.calls == [image_url]
    with sqlite3.connect(resumed.outcomes_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM image_url_claims"
        ).fetchone() == (0,)


def test_url_claim_lease_fences_stale_owner_and_is_policy_scoped(
    tmp_path: Path,
) -> None:
    store = ImageOutcomeStore(tmp_path / "outcomes.sqlite3")
    image_url = "https://i.test/fenced.jpg"
    url_key = hashlib.sha256(image_url.encode("utf-8")).hexdigest()
    first = store.claim_url(
        "policy-one",
        url_key,
        owner="first",
        lease_seconds=1,
        now=10,
    ).lease
    assert first is not None
    second = store.claim_url(
        "policy-one",
        url_key,
        owner="second",
        lease_seconds=1,
        now=12,
    ).lease
    assert second is not None
    assert second.lease_id != first.lease_id

    with pytest.raises(ValueError, match="stale or expired"):
        store.put_claimed(
            "policy-one",
            url_key,
            image_url,
            {
                "status": "terminal",
                "error_class": "stale",
            },
            lease=first,
            now=12,
        )
    assert store.finish_claim(
        "policy-one",
        url_key,
        lease=first,
    ) is False
    persisted = store.put_claimed(
        "policy-one",
        url_key,
        image_url,
        {
            "status": "terminal",
            "error_class": "current",
        },
        lease=second,
        now=12.5,
    )
    assert persisted["error_class"] == "current"
    waiter = store.claim_url(
        "policy-one",
        url_key,
        owner="waiter",
        lease_seconds=1,
        now=12.5,
    )
    assert waiter.outcome["error_class"] == "current"
    assert store.finish_claim(
        "policy-one",
        url_key,
        lease=second,
    ) is False

    other_policy = store.claim_url(
        "policy-two",
        url_key,
        owner="other-policy",
        lease_seconds=1,
        now=12.5,
    )
    assert other_policy.lease is not None
    expired_url = "https://i.test/expired-fence.jpg"
    expired_key = hashlib.sha256(
        expired_url.encode("utf-8")
    ).hexdigest()
    expired = store.claim_url(
        "policy-three",
        expired_key,
        owner="expired",
        lease_seconds=1,
        now=20,
    ).lease
    assert expired is not None
    assert store.finish_claim(
        "policy-three",
        expired_key,
        lease=expired,
        now=22,
    ) is False


def test_url_claim_release_is_execution_prefix_and_policy_scoped(
    tmp_path: Path,
) -> None:
    store = ImageOutcomeStore(tmp_path / "outcomes.sqlite3")
    owned_key = "a" * 64
    foreign_key = "b" * 64
    other_policy_key = "c" * 64
    owned = store.claim_url(
        "policy-one",
        owned_key,
        owner="current-execution:job:lease",
        lease_seconds=3600,
        now=10,
    ).lease
    foreign = store.claim_url(
        "policy-one",
        foreign_key,
        owner="current-execution-other:job:lease",
        lease_seconds=3600,
        now=10,
    ).lease
    other_policy = store.claim_url(
        "policy-two",
        other_policy_key,
        owner="current-execution:job:lease",
        lease_seconds=3600,
        now=10,
    ).lease
    assert owned is not None
    assert foreign is not None
    assert other_policy is not None

    assert (
        store.release_claims_by_owner_prefix(
            "policy-one",
            owner_prefix="current-execution:",
        )
        == 1
    )
    reclaimed = store.claim_url(
        "policy-one",
        owned_key,
        owner="resumed-execution:job:lease",
        lease_seconds=3600,
        now=10,
    )
    still_foreign = store.claim_url(
        "policy-one",
        foreign_key,
        owner="waiter",
        lease_seconds=3600,
        now=10,
    )
    still_other_policy = store.claim_url(
        "policy-two",
        other_policy_key,
        owner="waiter",
        lease_seconds=3600,
        now=10,
    )

    assert reclaimed.lease is not None
    assert still_foreign.retry_at == foreign.lease_until
    assert still_other_policy.retry_at == other_policy.lease_until


def test_image_outcome_put_commit_guard_rolls_back_and_resumes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "outcomes.sqlite3"
    ImageOutcomeStore(path)
    zero_checks = 0

    def reject_commit(_path: Path, estimated_bytes: int = 0) -> None:
        nonlocal zero_checks
        if estimated_bytes == 0:
            zero_checks += 1
            if zero_checks == 3:
                raise OSError("image outcome commit reserve exhausted")

    tracker = assets_module.GuardedWriteTracker(
        path,
        reject_commit,
        interval_bytes=1,
    )
    guarded = ImageOutcomeStore(path, write_tracker=tracker)
    image_url = "https://i.test/put.jpg"
    url_key = hashlib.sha256(image_url.encode("utf-8")).hexdigest()
    with pytest.raises(OSError, match="image outcome commit reserve"):
        guarded.put(
            "policy",
            url_key,
            image_url,
            {"status": "terminal", "error_class": "TimeoutError"},
        )

    resumed = ImageOutcomeStore(path)
    assert resumed.get("policy", url_key) is None
    resumed.put(
        "policy",
        url_key,
        image_url,
        {"status": "terminal", "error_class": "TimeoutError"},
    )
    assert resumed.get("policy", url_key)["status"] == "terminal"


def test_image_transport_attempt_summary_counts_resume_anomalies(
    tmp_path: Path,
) -> None:
    store = ImageOutcomeStore(tmp_path / "outcomes.sqlite3")
    policy = "policy"
    image_url = "https://i.test/anomaly.jpg"
    url_key = hashlib.sha256(image_url.encode("utf-8")).hexdigest()

    first = store.begin_transport_attempt(
        execution_id="execution-one",
        policy_fingerprint=policy,
        url_key=url_key,
        url=image_url,
        baseline_outcome_status=None,
        suppressed=False,
        now=1.0,
    )
    store.finish_transport_attempt(first, final_status="success", now=2.0)
    store.put(
        policy,
        url_key,
        image_url,
        {
            "status": "terminal",
            "image_url": image_url,
            "error_class": "prior_terminal",
        },
    )
    second = store.begin_transport_attempt(
        execution_id="execution-two",
        policy_fingerprint=policy,
        url_key=url_key,
        url=image_url,
        baseline_outcome_status=None,
        suppressed=False,
        now=time.time() + 1.0,
    )
    store.finish_transport_attempt(
        second,
        final_status="terminal",
        now=time.time() + 2.0,
    )
    store.begin_transport_attempt(
        execution_id="execution-three",
        policy_fingerprint=policy,
        url_key=url_key,
        url=image_url,
        baseline_outcome_status="terminal",
        suppressed=True,
        now=5.0,
    )
    store.begin_transport_attempt(
        execution_id="execution-four",
        policy_fingerprint=policy,
        url_key="b" * 64,
        url="https://i.test/crash.jpg",
        baseline_outcome_status=None,
        suppressed=False,
        now=6.0,
    )

    summary = ImageOutcomeStore(
        tmp_path / "outcomes.sqlite3"
    ).transport_attempt_summary(policy)
    assert summary["records"] == 4
    assert summary["transport_attempts"] == 3
    assert summary["duplicate_physical_requests"] == 1
    assert summary["terminal_replays"] == 1
    assert summary["unfinished_transport_attempts"] == 1
    assert summary["blocked_durable_replays"] == 1


def test_image_transport_attempt_start_is_disk_guarded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "outcomes.sqlite3"
    ImageOutcomeStore(path)

    def reject(_path: Path, estimated_bytes: int = 0) -> None:
        if estimated_bytes:
            raise OSError("image transport reserve exhausted")

    guarded = ImageOutcomeStore(
        path,
        write_tracker=assets_module.GuardedWriteTracker(
            path,
            reject,
            interval_bytes=1,
        ),
    )
    with pytest.raises(OSError, match="image transport reserve"):
        guarded.begin_transport_attempt(
            execution_id="execution",
            policy_fingerprint="policy",
            url_key="a" * 64,
            url="https://i.test/guard.jpg",
            baseline_outcome_status=None,
            suppressed=False,
        )
    assert ImageOutcomeStore(path).transport_attempt_summary("policy")[
        "records"
    ] == 0


@pytest.mark.parametrize("failure_mode", ["guard", "sqlite"])
def test_image_finish_fence_failure_preserves_root_error_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    image_url = "https://i.test/finish-fence.jpg"
    unique_jobs = write_unique_jobs(tmp_path / "unique.jsonl", [image_url])
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )
    fingerprint = assets_module._image_policy_fingerprint(policy)
    jobs = SqliteJobStore(tmp_path / "jobs.sqlite3")
    outcomes_path = tmp_path / "outcomes.sqlite3"
    ImageOutcomeStore(outcomes_path)
    expected_type: type[BaseException]
    expected_message: str
    guard = None
    if failure_mode == "guard":
        expected_type = OSError
        expected_message = "image finish guard sentinel"

        def reject_finish(path: Path, estimated_bytes: int = 0) -> None:
            if Path(path) != outcomes_path or estimated_bytes != 0:
                return
            with sqlite3.connect(outcomes_path) as connection:
                unfinished = connection.execute(
                    "SELECT COUNT(*) FROM transport_attempts "
                    "WHERE finished_at IS NULL"
                ).fetchone()[0]
            if unfinished:
                raise OSError(expected_message)

        guard = reject_finish
    else:
        expected_type = sqlite3.IntegrityError
        expected_message = "image finish sqlite sentinel"
        with sqlite3.connect(outcomes_path) as connection:
            connection.execute(
                "CREATE TRIGGER fail_image_transport_finish "
                "BEFORE UPDATE OF finished_at ON transport_attempts "
                "WHEN NEW.finished_at IS NOT NULL "
                f"BEGIN SELECT RAISE(ABORT, '{expected_message}'); END"
            )

    trackers = []
    tracker_type = assets_module.UrlProgressTracker

    def capture_tracker(*args, **kwargs):
        tracker = tracker_type(*args, **kwargs)
        trackers.append(tracker)
        return tracker

    monkeypatch.setattr(assets_module, "UrlProgressTracker", capture_tracker)
    transport = FakeImageTransport(tmp_path, {image_url: "unique"})
    with pytest.raises(expected_type, match=expected_message):
        fetch_unique_images(
            unique_jobs,
            jobs,
            transport,
            policy,
            outcomes_path=outcomes_path,
            image_dir=tmp_path / "content",
            lease_seconds=60.0,
            pre_write_guard=guard,
        )

    failed_snapshot = trackers[0].snapshot(
        DurableUrlCounts(completed=0, pending=0, leased=1, total=1)
    )
    assert failed_snapshot.physical_in_flight == 1
    assert failed_snapshot.finished_not_durable == 0
    url_key = hashlib.sha256(image_url.encode("utf-8")).hexdigest()
    outcome_store = ImageOutcomeStore(outcomes_path)
    assert outcome_store.get(fingerprint, url_key) is None
    summary = outcome_store.transport_attempt_summary(fingerprint)
    assert summary["unfinished_transport_attempts"] == 1
    with sqlite3.connect(jobs.path) as connection:
        assert connection.execute(
            "SELECT status, COUNT(*) FROM jobs GROUP BY status"
        ).fetchall() == [("retryable", 1)]
    if failure_mode == "sqlite":
        with sqlite3.connect(outcomes_path) as connection:
            connection.execute("DROP TRIGGER fail_image_transport_finish")

    resumed_transport = FakeImageTransport(tmp_path, {image_url: "unique"})
    resumed = fetch_unique_images(
        unique_jobs,
        jobs,
        resumed_transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
        lease_seconds=60.0,
    )
    assert resumed.complete
    assert resumed_transport.calls == [image_url]


def test_image_mixed_completed_batch_commits_success_before_root_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    success_url = "https://success.test/mixed.jpg"
    failed_url = "https://failed.test/mixed.jpg"
    unique_jobs = write_unique_jobs(
        tmp_path / "unique.jsonl", [success_url, failed_url]
    )
    barrier = threading.Barrier(2)

    class MixedTransport(FakeImageTransport):
        def download_image(self, image_url: str, **kwargs):
            barrier.wait(timeout=5)
            return super().download_image(image_url, **kwargs)

    original_finish = ImageOutcomeStore.finish_transport_attempt

    def fail_selected(self, attempt_id: str, **kwargs) -> None:
        with sqlite3.connect(self.path) as connection:
            url = connection.execute(
                "SELECT url FROM transport_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()[0]
        if url == failed_url:
            raise OSError("mixed image finish sentinel")
        original_finish(self, attempt_id, **kwargs)

    real_wait = assets_module.wait

    def failed_first(fs, *, return_when):
        done, pending = real_wait(fs)
        return sorted(done, key=lambda future: future.exception() is None), pending

    monkeypatch.setattr(ImageOutcomeStore, "finish_transport_attempt", fail_selected)
    monkeypatch.setattr(assets_module, "wait", failed_first)
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=2,
        per_host_concurrency=1,
    )
    fingerprint = assets_module._image_policy_fingerprint(policy)
    jobs = SqliteJobStore(tmp_path / "jobs.sqlite3")
    outcomes_path = tmp_path / "outcomes.sqlite3"
    snapshots: list[UrlProgressSnapshot] = []
    with pytest.raises(OSError, match="mixed image finish sentinel"):
        fetch_unique_images(
            unique_jobs,
            jobs,
            MixedTransport(
                tmp_path,
                {success_url: "success", failed_url: "failed"},
            ),
            policy,
            outcomes_path=outcomes_path,
            image_dir=tmp_path / "content",
            lease_seconds=60.0,
            progress_callback=snapshots.append,
        )

    store = ImageOutcomeStore(outcomes_path)
    success_key = hashlib.sha256(success_url.encode("utf-8")).hexdigest()
    failed_key = hashlib.sha256(failed_url.encode("utf-8")).hexdigest()
    assert store.get(fingerprint, success_key)["status"] == "success"
    assert store.get(fingerprint, failed_key) is None
    assert [snapshot.completed_durable for snapshot in snapshots] == [0, 1]
    with sqlite3.connect(jobs.path) as connection:
        assert connection.execute(
            "SELECT status, COUNT(*) FROM jobs GROUP BY status ORDER BY status"
        ).fetchall() == [("retryable", 1), ("success", 1)]

    monkeypatch.setattr(ImageOutcomeStore, "finish_transport_attempt", original_finish)
    resumed_transport = FakeImageTransport(tmp_path, {failed_url: "retry"})
    resumed = fetch_unique_images(
        unique_jobs,
        jobs,
        resumed_transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
    )
    assert resumed.complete
    assert resumed_transport.calls == [failed_url]


def test_image_outcome_initialization_commit_uses_live_guard(
    tmp_path: Path,
) -> None:
    path = tmp_path / "outcomes.sqlite3"
    zero_checks = 0

    def reject_commit(_path: Path, estimated_bytes: int = 0) -> None:
        nonlocal zero_checks
        if estimated_bytes == 0:
            zero_checks += 1
            if zero_checks == 2:
                raise OSError("image init reserve exhausted")

    tracker = assets_module.GuardedWriteTracker(path, reject_commit)
    with pytest.raises(OSError, match="image init reserve"):
        ImageOutcomeStore(path, write_tracker=tracker)

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE name = 'image_outcomes'"
        ).fetchone() == (0,)


def test_entity_page_join_ddl_commit_uses_live_guard(
    tmp_path: Path,
) -> None:
    path = tmp_path / "join.sqlite3"
    zero_checks = 0

    def reject_commit(_path: Path, estimated_bytes: int = 0) -> None:
        nonlocal zero_checks
        if estimated_bytes == 0:
            zero_checks += 1
            if zero_checks == 2:
                raise OSError("join init reserve exhausted")

    with pytest.raises(OSError, match="join init reserve"):
        list(
            iter_entity_page_join(
                [],
                [],
                join_path=path,
                pre_write_guard=reject_commit,
            )
        )

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE name = 'entity_pages'"
        ).fetchone() == (0,)


def test_image_url_claim_commit_guard_rolls_back_lease(
    tmp_path: Path,
) -> None:
    path = tmp_path / "outcomes.sqlite3"
    ImageOutcomeStore(path)
    zero_checks = 0

    def reject_commit(_path: Path, estimated_bytes: int = 0) -> None:
        nonlocal zero_checks
        if estimated_bytes == 0:
            zero_checks += 1
            if zero_checks == 3:
                raise OSError("image claim commit reserve exhausted")

    tracker = assets_module.GuardedWriteTracker(
        path,
        reject_commit,
        interval_bytes=1,
    )
    guarded = ImageOutcomeStore(path, write_tracker=tracker)
    with pytest.raises(OSError, match="image claim commit reserve"):
        guarded.claim_url(
            "policy",
            "a" * 64,
            owner="worker",
            lease_seconds=60,
        )

    resumed = ImageOutcomeStore(path)
    decision = resumed.claim_url(
        "policy",
        "a" * 64,
        owner="worker",
        lease_seconds=60,
    )
    assert decision.lease is not None


def test_image_outcome_snapshot_uses_guarded_membership_file(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    planned = persist_entity_asset_plans(
        [
            (
                entity(
                    "e1",
                    image_urls=["https://i.test/one.jpg"],
                ),
                page(),
            )
        ],
        output_root=tmp_path / "plans",
        input_fingerprint="snapshot-membership-v1",
    )
    unique = build_unique_image_jobs(
        planned,
        tmp_path / "unique.jsonl",
    )
    store = ImageOutcomeStore(cache_dir / "outcomes.sqlite3")
    calls: list[tuple[Path, int]] = []

    snapshot = store.snapshot_for_jobs(
        "policy",
        unique,
        pre_write_guard=lambda path, size=0: calls.append(
            (Path(path), size)
        ),
    )

    assert snapshot["count"] == 0
    assert snapshot["missing"] == 1
    assert calls
    assert {path.parent for path, _size in calls} == {cache_dir}
    assert any(size > 0 for _path, size in calls)
    assert any(size == 0 for _path, size in calls)
    assert not list(cache_dir.glob(".image-membership-*"))


def test_image_outcome_snapshot_cleans_membership_on_initial_guard_failure(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    planned = persist_entity_asset_plans(
        [
            (
                entity(
                    "e1",
                    image_urls=["https://i.test/one.jpg"],
                ),
                page(),
            )
        ],
        output_root=tmp_path / "plans",
        input_fingerprint="snapshot-guard-cleanup-v1",
    )
    unique = build_unique_image_jobs(
        planned,
        tmp_path / "unique.jsonl",
    )
    store = ImageOutcomeStore(cache_dir / "outcomes.sqlite3")

    def reject_initial_write(
        _path: Path,
        estimated_bytes: int = 0,
    ) -> None:
        if estimated_bytes > 0:
            raise OSError("snapshot membership reserve exhausted")

    with pytest.raises(OSError, match="membership reserve"):
        store.snapshot_for_jobs(
            "policy",
            unique,
            pre_write_guard=reject_initial_write,
        )

    assert not list(cache_dir.glob(".image-membership-*"))


def test_image_outcome_snapshot_cleans_membership_on_connect_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    planned = persist_entity_asset_plans(
        [
            (
                entity(
                    "e1",
                    image_urls=["https://i.test/one.jpg"],
                ),
                page(),
            )
        ],
        output_root=tmp_path / "plans",
        input_fingerprint="snapshot-connect-cleanup-v1",
    )
    unique = build_unique_image_jobs(
        planned,
        tmp_path / "unique.jsonl",
    )
    store = ImageOutcomeStore(cache_dir / "outcomes.sqlite3")
    original_connect = assets_module.sqlite3.connect

    def reject_membership(path: object, *args: object, **kwargs: object):
        if Path(path).name.startswith(".image-membership-"):
            raise sqlite3.OperationalError("synthetic membership connect")
        return original_connect(path, *args, **kwargs)

    monkeypatch.setattr(
        assets_module.sqlite3,
        "connect",
        reject_membership,
    )
    with pytest.raises(sqlite3.OperationalError, match="membership connect"):
        store.snapshot_for_jobs("policy", unique)

    assert not list(cache_dir.glob(".image-membership-*"))


def test_interrupted_cache_write_immediately_repairs_without_refetch(
    tmp_path: Path,
) -> None:
    image_url = "https://i.test/leased.jpg"
    unique_path = tmp_path / "unique.jsonl"
    unique_jobs = write_unique_jobs(unique_path, [image_url])
    transport = FakeImageTransport(tmp_path, {image_url: "unique"})
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")

    with pytest.raises(RuntimeError, match="simulated crash"):
        fetch_unique_images(
            unique_jobs,
            store,
            transport,
            policy,
            outcomes_path=tmp_path / "outcomes.sqlite3",
            image_dir=tmp_path / "content",
            lease_seconds=60,
            after_cache_write=lambda _outcome: (
                _ for _ in ()
            ).throw(RuntimeError("simulated crash")),
        )

    concurrent = fetch_unique_images(
        unique_jobs,
        store,
        transport,
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
    )
    assert concurrent.complete is True
    assert concurrent.leased == 0
    assert concurrent.remaining == 0
    assert transport.calls == [image_url]


class TrackingTerminalTransport:
    network_policy_fingerprint = "image-v1"
    max_retries = 0
    max_response_seconds = 8.0

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0
        self.active_by_host: dict[str, int] = {}
        self.maximum_by_host: dict[str, int] = {}
        self.calls = 0
        self.cache: dict[str, dict] = {}

    def cached_image_outcome(self, image_url: str) -> dict | None:
        return self.cache.get(image_url)

    def download_image(self, image_url: str, **_kwargs) -> None:
        host = image_url.split("/", 3)[2]
        with self.lock:
            self.calls += 1
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            active = self.active_by_host.get(host, 0) + 1
            self.active_by_host[host] = active
            self.maximum_by_host[host] = max(
                self.maximum_by_host.get(host, 0),
                active,
            )
        time.sleep(0.01)
        with self.lock:
            self.active -= 1
            self.active_by_host[host] -= 1
            self.cache[image_url] = {
                "status": "terminal",
                "image_url": image_url,
                "error_class": "not_an_image",
                "policy_fingerprint": self.network_policy_fingerprint,
            }
        return None


def test_image_scheduler_releases_host_state_for_many_unique_hosts(
    tmp_path: Path,
) -> None:
    urls = [
        f"https://h{index}.test/image.png"
        for index in range(2_000)
    ]
    unique_path = tmp_path / "unique.jsonl"
    unique_jobs = write_unique_jobs(unique_path, urls)
    transport = TrackingTerminalTransport()
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=8,
        per_host_concurrency=2,
    )

    fetched = fetch_unique_images(
        unique_jobs,
        SqliteJobStore(tmp_path / "jobs.sqlite3"),
        transport,
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
        claim_buffer=32,
    )

    assert fetched.maximum_claimed <= 32
    assert fetched.maximum_inflight <= 8
    assert fetched.maximum_host_states <= 32
    assert transport.calls == len(urls)


def test_image_scheduler_enforces_global_and_per_host_limits(
    tmp_path: Path,
) -> None:
    urls = [
        f"https://h{index % 3}.test/{index}.png"
        for index in range(90)
    ]
    unique_path = tmp_path / "unique.jsonl"
    unique_jobs = write_unique_jobs(unique_path, urls)
    transport = TrackingTerminalTransport()
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=6,
        per_host_concurrency=2,
    )

    fetched = fetch_unique_images(
        unique_jobs,
        SqliteJobStore(tmp_path / "jobs.sqlite3"),
        transport,
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
        claim_buffer=24,
    )

    assert transport.maximum_active <= 6
    assert max(transport.maximum_by_host.values()) == 2
    assert fetched.maximum_inflight == 6
    assert fetched.maximum_claimed <= 24


def test_two_entity_fanout_keeps_two_links_after_one_shared_request(
    tmp_path: Path,
) -> None:
    shared = "https://i.test/shared.jpg"
    entities = [
        entity("e1", image_urls=[shared]),
        entity("e2", image_urls=[shared]),
    ]
    unique_path = tmp_path / "unique.jsonl"
    unique_jobs = write_unique_jobs(unique_path, [shared])
    transport = FakeImageTransport(tmp_path, {shared: "shared"})
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )
    fetched = fetch_unique_images(
        unique_jobs,
        SqliteJobStore(tmp_path / "jobs.sqlite3"),
        transport,
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
    )
    outcomes = {
        outcome["image_url"]: outcome
        for outcome in iter_image_outcomes(
            fetched.outcomes_path,
            fetched.policy_fingerprint,
        )
    }
    results = [
        materialize_entity_assets(
            current,
            page(),
            outcomes,
        )
        for current in entities
    ]

    assert transport.calls == [shared]
    assert sum(
        len(result.table_asset_links)
        for result in results
    ) == 2
    assert all(
        len(result.table_asset_links[0]["asset_ids"]) == 1
        for result in results
    )


def test_materialization_streams_canonical_shards_and_resumes(
    tmp_path: Path,
) -> None:
    shared = "https://i.test/shared.jpg"
    planned = persist_entity_asset_plans(
        [
            (
                entity("e1", image_urls=[shared]),
                page(text="e1 durable page text"),
            ),
            (
                entity("e2", image_urls=[shared]),
                page(text="e2 durable page text"),
            ),
        ],
        output_root=tmp_path / "plans",
        input_fingerprint="entities-v1",
        records_per_shard=1,
    )
    unique_path = tmp_path / "unique.jsonl"
    unique_jobs = build_unique_image_jobs(
        planned,
        unique_path,
        chunk_records=1,
    )
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )
    fetched = fetch_unique_images(
        unique_jobs,
        SqliteJobStore(tmp_path / "jobs.sqlite3"),
        FakeImageTransport(tmp_path, {shared: "shared"}),
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
    )

    materialized = materialize_asset_shards(
        planned,
        fetch_result=fetched,
        output_root=tmp_path / "materialized",
        input_fingerprint="assets-v1",
        records_per_shard=1,
    )

    assets = read_jsonl(materialized.bridge_asset_paths)
    links = read_jsonl(materialized.table_asset_link_paths)
    assert [asset["asset_type"] for asset in assets] == [
        "text",
        "image",
        "text",
        "image",
    ]
    assert len(links) == 2
    assert all(len(link["asset_ids"]) == 2 for link in links)
    assert max(
        len(read_jsonl([path]))
        for path in (
            *materialized.bridge_asset_paths,
            *materialized.table_asset_link_paths,
        )
    ) == 1
    mtimes = {
        path: path.stat().st_mtime_ns
        for path in (
            *materialized.bridge_asset_paths,
            *materialized.table_asset_link_paths,
        )
    }
    resumed = materialize_asset_shards(
        planned,
        fetch_result=fetched,
        output_root=tmp_path / "materialized",
        input_fingerprint="assets-v1",
        records_per_shard=1,
    )
    assert resumed == materialized
    assert {path: path.stat().st_mtime_ns for path in mtimes} == mtimes

    with pytest.raises(ValueError, match="complete"):
        materialize_asset_shards(
            planned,
            fetch_result=replace(fetched, complete=False),
            output_root=tmp_path / "materialized",
            input_fingerprint="assets-v1",
            records_per_shard=1,
        )

    forged_outcomes_path = tmp_path / "forged-outcomes.sqlite3"
    with (
        sqlite3.connect(fetched.outcomes_path) as source,
        sqlite3.connect(forged_outcomes_path) as target,
    ):
        source.backup(target)
    with pytest.raises(ValueError, match="job-set identity"):
        materialize_asset_shards(
            planned,
            fetch_result=replace(
                fetched,
                outcomes_path=forged_outcomes_path,
            ),
            output_root=tmp_path / "materialized",
            input_fingerprint="assets-v1",
            records_per_shard=1,
        )

    with sqlite3.connect(fetched.outcomes_path) as connection:
        payload = json.dumps(
            {
                "status": "terminal",
                "image_url": "https://i.test/unplanned.jpg",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        connection.execute(
            """
            INSERT INTO image_outcomes (
                policy_fingerprint, url_key, image_url, status,
                outcome_json, payload_sha256, updated_at
            ) VALUES (?, ?, ?, 'terminal', ?, ?, 0)
            """,
            (
                fetched.policy_fingerprint,
                "f" * 64,
                "https://i.test/unplanned.jpg",
                payload,
                hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            ),
        )

    resumed_after_unrelated_outcome = materialize_asset_shards(
        planned,
        fetch_result=fetched,
        output_root=tmp_path / "materialized",
        input_fingerprint="assets-v1",
        records_per_shard=1,
    )
    assert resumed_after_unrelated_outcome == materialized


def test_public_asset_validators_reconstruct_the_complete_producer_chain(
    tmp_path: Path,
) -> None:
    image_url = "https://i.test/asset.jpg"
    structural_identity = hashlib.sha256(b"structural").hexdigest()
    page_identity = hashlib.sha256(b"page-fetch").hexdigest()
    planning_input = asset_planning_input_fingerprint(
        structural_identity,
        page_identity,
    )
    planned = persist_entity_asset_plans(
        [
            (
                entity("e1", image_urls=[image_url]),
                page(text="durable page text"),
            )
        ],
        output_root=tmp_path / "plans",
        input_fingerprint=planning_input,
    )
    validated_plan = validate_asset_plan_shards(
        planned,
        expected_input_fingerprint=planning_input,
    )
    unique = build_unique_image_jobs(
        validated_plan,
        tmp_path / "unique.jsonl",
    )
    validate_unique_image_jobs(
        unique,
        planned=validated_plan,
        validation_database=tmp_path / "validate-unique.sqlite3",
    )
    fetched = fetch_unique_images(
        unique,
        SqliteJobStore(tmp_path / "jobs.sqlite3"),
        FakeImageTransport(tmp_path, {image_url: "asset"}),
        FetchPolicy(
            network_policy_fingerprint="image-v1",
            policy_version="wdc200k-image-v1",
        ),
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
    )
    guarded_paths: list[Path] = []
    fetch_snapshot = validate_complete_image_fetch(
        fetched,
        unique_jobs=unique,
        pre_write_guard=lambda path, _size=0: guarded_paths.append(
            Path(path)
        ),
    )
    materialization_input = asset_materialization_input_fingerprint(
        planned.manifest_path,
        fetched.fetch_manifest_path,
    )
    materialized = materialize_asset_shards(
        planned,
        fetch_result=fetched,
        output_root=tmp_path / "assets",
        input_fingerprint=materialization_input,
    )

    validated, barrier = validate_materialized_asset_shards(
        materialized,
        planned=planned,
        image_fetch_result=fetched,
        expected_input_fingerprint=materialization_input,
    )

    assert validated == materialized
    assert fetch_snapshot["outcomes"]["count"] == 1
    assert fetched.outcomes_path in guarded_paths
    assert fetched.job_store_path in guarded_paths
    assert any(
        path.name.startswith(".image-membership-")
        for path in guarded_paths
    )
    assert barrier.bridge_assets == 2
    assert barrier.table_asset_links == 1
    assert barrier.fingerprint["input_fingerprint"] == materialization_input

    with pytest.raises(ValueError, match="planning input"):
        validate_asset_plan_shards(
            planned,
            expected_input_fingerprint="foreign-planning-run",
        )
    with pytest.raises(ValueError, match="materialization input"):
        validate_materialized_asset_shards(
            materialized,
            planned=planned,
            image_fetch_result=fetched,
            expected_input_fingerprint="foreign-materialization-run",
        )


def test_unique_image_validation_recovers_after_mid_batch_guard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned = persist_entity_asset_plans(
        [
            (
                entity(
                    f"e{index}",
                    image_urls=[f"https://i.test/{index}.jpg"],
                ),
                page(),
            )
            for index in range(24)
        ],
        output_root=tmp_path / "plans",
        input_fingerprint="guarded-planning-v1",
    )
    unique = build_unique_image_jobs(planned, tmp_path / "unique.jsonl")
    validation_database = tmp_path / "validate-unique.sqlite3"
    positive_calls = 0
    monkeypatch.setattr(
        assets_module.GuardedWriteTracker,
        "DEFAULT_INTERVAL_BYTES",
        1,
    )

    def interrupt(path: Path, estimated_bytes: int = 0) -> None:
        nonlocal positive_calls
        assert Path(path) == validation_database
        if estimated_bytes > 0:
            positive_calls += 1
            if positive_calls == 2:
                raise RuntimeError("synthetic image reserve exhausted")

    with pytest.raises(RuntimeError, match="image reserve"):
        validate_unique_image_jobs(
            unique,
            planned=planned,
            validation_database=validation_database,
            pre_write_guard=interrupt,
        )

    resumed = validate_unique_image_jobs(
        unique,
        planned=planned,
        validation_database=validation_database,
    )
    assert positive_calls == 2
    assert resumed == unique


def test_image_failures_fan_out_reference_counts_for_current_jobset(
    tmp_path: Path,
) -> None:
    shared = "https://i.test/shared-failure.jpg"
    planned = persist_entity_asset_plans(
        [
            (entity("e1", image_urls=[shared]), page()),
            (entity("e2", image_urls=[shared]), page()),
        ],
        output_root=tmp_path / "plans-two",
        input_fingerprint="two-entity-plan",
    )
    unique = build_unique_image_jobs(
        planned,
        tmp_path / "unique-two.jsonl",
    )
    policy = FetchPolicy(
        deadline_seconds=8.0,
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
    )
    transport = FakeImageTransport(
        tmp_path,
        {shared: TimeoutError("shared failure")},
    )
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    outcomes_path = tmp_path / "outcomes.sqlite3"
    fetched = fetch_unique_images(
        unique,
        store,
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
    )

    failures = list(
        iter_image_failures(
            fetched,
            planned=planned,
            aggregation_database=tmp_path / "failure-fanout.sqlite3",
        )
    )

    assert len(failures) == 1
    assert failures[0]["affected_reference_count"] == 2
    assert failures[0]["url_key"] == hashlib.sha256(
        shared.encode("utf-8")
    ).hexdigest()

    one_entity_plan = persist_entity_asset_plans(
        [(entity("e3", image_urls=[shared]), page())],
        output_root=tmp_path / "plans-one",
        input_fingerprint="one-entity-plan",
    )
    one_entity_unique = build_unique_image_jobs(
        one_entity_plan,
        tmp_path / "unique-one.jsonl",
    )
    one_entity_fetch = fetch_unique_images(
        one_entity_unique,
        store,
        transport,
        policy,
        outcomes_path=outcomes_path,
        image_dir=tmp_path / "content",
    )
    expected = list(
        iter_image_failures(
            one_entity_fetch,
            planned=one_entity_plan,
            aggregation_database=tmp_path / "failure-fanout.sqlite3",
        )
    )
    resumed = list(
        iter_image_failures(
            one_entity_fetch,
            planned=one_entity_plan,
            aggregation_database=tmp_path / "failure-fanout.sqlite3",
        )
    )

    assert len(expected) == 1
    assert expected[0]["affected_reference_count"] == 1
    assert expected[0]["entity_id"] == "e3"
    assert resumed == expected


@pytest.mark.parametrize("mutation", ["delete", "add", "replace"])
def test_unique_image_validator_rejects_rechecksummed_membership_changes(
    tmp_path: Path,
    mutation: str,
) -> None:
    shared = "https://i.test/shared.jpg"
    planned = persist_entity_asset_plans(
        [
            (entity("e1", image_urls=[shared]), page()),
            (entity("e2", image_urls=[shared]), page()),
        ],
        output_root=tmp_path / "plans",
        input_fingerprint="planning-v1",
    )
    unique = build_unique_image_jobs(
        planned,
        tmp_path / "unique.jsonl",
    )
    records = read_jsonl((unique.output_path,))
    if mutation == "delete":
        forged_records: list[dict] = []
    elif mutation == "add":
        extra_url = "https://i.test/extra.jpg"
        forged_records = [
            *records,
            {
                **records[0],
                "image_url": extra_url,
                "url_key": hashlib.sha256(
                    extra_url.encode("utf-8")
                ).hexdigest(),
            },
        ]
    else:
        forged_records = [{**records[0], "entity_id": "substituted"}]
    writer = AtomicJsonlShard(unique.output_path)
    for record in forged_records:
        writer.write(record)
    completed = writer.commit()
    manifest = json.loads(
        unique.manifest_path.read_text(encoding="utf-8")
    )
    manifest["completed_shards"] = [asdict(completed)]
    unique.manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    forged = replace(unique, completed_shard=completed)

    with pytest.raises(ValueError, match="unique image.*membership"):
        validate_unique_image_jobs(
            forged,
            planned=planned,
            validation_database=tmp_path / "validate-unique.sqlite3",
        )
