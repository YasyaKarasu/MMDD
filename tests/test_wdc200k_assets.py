import hashlib
import json
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from wdc200k_assets import (  # noqa: E402
    ImageBudget,
    build_unique_image_jobs,
    fetch_unique_images,
    iter_image_outcomes,
    materialize_asset_shards,
    materialize_entity_assets,
    persist_entity_asset_plans,
    plan_entity_assets,
)
from wdc200k_fetch import FetchPolicy  # noqa: E402
from wdc200k_io import SqliteJobStore  # noqa: E402
from build_wdc_mm_joinability_dataset import WdcWebClient  # noqa: E402
import build_wdc_mm_joinability_dataset as legacy_wdc_builder  # noqa: E402


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
    unique = read_jsonl([unique_path])
    assert len(
        {
            record["url_key"]
            for record in unique
        }
    ) == 8


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


def write_unique_jobs(path: Path, urls: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                plan_entity_assets(
                    entity(f"e{index}", image_urls=[url]),
                    page(),
                ).image_refs[0].as_record()
            )
            + "\n"
            for index, url in enumerate(urls)
        ),
        encoding="utf-8",
    )


def test_unique_image_fetch_requests_each_url_once_and_content_addresses(
    tmp_path: Path,
) -> None:
    urls = [
        "https://i.test/shared.jpg",
        "https://i.test/shared.jpg",
        "https://i.test/other.jpg",
    ]
    unique_path = tmp_path / "unique.jsonl"
    write_unique_jobs(unique_path, list(dict.fromkeys(urls)))
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
        [unique_path],
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
    write_unique_jobs(unique_path, urls)
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
        [unique_path],
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
    write_unique_jobs(unique_path, [image_url])
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
        [unique_path],
        store,
        transport,
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
    )
    second = fetch_unique_images(
        [unique_path],
        store,
        transport,
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
    )

    assert first.terminal == second.terminal == 1
    assert transport.calls == [image_url]


def test_image_outcome_stream_detects_payload_checksum_corruption(
    tmp_path: Path,
) -> None:
    image_url = "https://i.test/broken.jpg"
    unique_path = tmp_path / "unique.jsonl"
    write_unique_jobs(unique_path, [image_url])
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )
    fetched = fetch_unique_images(
        [unique_path],
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
    write_unique_jobs(unique_path, [image_url])
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
            [unique_path],
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
        [unique_path],
        store,
        transport,
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
    )
    assert resumed.success == 1
    assert transport.calls == [image_url]


def test_two_entity_fanout_keeps_two_links_after_one_shared_request(
    tmp_path: Path,
) -> None:
    shared = "https://i.test/shared.jpg"
    entities = [
        entity("e1", image_urls=[shared]),
        entity("e2", image_urls=[shared]),
    ]
    unique_path = tmp_path / "unique.jsonl"
    write_unique_jobs(unique_path, [shared])
    transport = FakeImageTransport(tmp_path, {shared: "shared"})
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )
    fetched = fetch_unique_images(
        [unique_path],
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
    build_unique_image_jobs(planned, unique_path, chunk_records=1)
    policy = FetchPolicy(
        network_policy_fingerprint="image-v1",
        policy_version="wdc200k-image-v1",
        global_concurrency=1,
        per_host_concurrency=1,
    )
    fetched = fetch_unique_images(
        [unique_path],
        SqliteJobStore(tmp_path / "jobs.sqlite3"),
        FakeImageTransport(tmp_path, {shared: "shared"}),
        policy,
        outcomes_path=tmp_path / "outcomes.sqlite3",
        image_dir=tmp_path / "content",
    )

    materialized = materialize_asset_shards(
        planned,
        outcomes_path=fetched.outcomes_path,
        policy_fingerprint=fetched.policy_fingerprint,
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
        outcomes_path=fetched.outcomes_path,
        policy_fingerprint=fetched.policy_fingerprint,
        output_root=tmp_path / "materialized",
        input_fingerprint="assets-v1",
        records_per_shard=1,
    )
    assert resumed == materialized
    assert {path: path.stat().st_mtime_ns for path in mtimes} == mtimes
