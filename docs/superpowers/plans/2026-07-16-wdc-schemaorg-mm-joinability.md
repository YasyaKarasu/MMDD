# WDC Schema.org Multimodal Joinability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a WDC Schema.org 2023 adapter that always extracts entity webpage text, combines direct and webpage images under one quota, removes the source `image` column, and reuses the current joinability query algorithm.

**Architecture:** A focused WDC module converts gzip JSONL host tables into the existing internal table contract and collects generic web bridge assets with a resumable SQLite cache. The existing joinability module remains the single implementation of model extraction, recovery qualification, query/target projection, qrels, and evidence paths.

**Tech Stack:** Python 3.10, stdlib gzip/HTMLParser/sqlite3/concurrency/http.client/ssl, dnspython for deadline-bounded A/AAAA/CNAME resolution, Pillow, pytest, existing Stage-1 and joinability helpers.

## Global Constraints

- Use the `MMDD` conda environment.
- Use 4-space indentation, public-helper type hints, and `pathlib.Path`.
- Do not add a third-party HTML parser.
- Always fetch `page_url`, even after a successful `image`-column download.
- Direct and page images share `max_images_per_entity`; direct images have priority.
- Reuse the existing 800/120/3 text chunk behavior and query construction code.
- Remove `image` from every emitted table.
- Network failures are per-entity records, not fatal build errors.
- Safe defaults are 100 accepted tables, 1,000 attempted gzip files, 100 rows per table, and 2 images per entity; unlimited modes require `--allow_unbounded`.
- Network connections must be pinned to a validated public DNS answer while retaining the original HTTPS hostname for SNI and certificate verification.
- Page bodies and images share a total cache budget and streaming disk-free guard. External SVG is rejected.

---

### Task 1: WDC gzip table adapter

**Files:**
- Create: `scripts/build_wdc_mm_joinability_dataset.py`
- Create: `tests/test_wdc_mm_joinability_dataset.py`

**Interfaces:**
- Produces: `extract_image_urls(value: Any, base_url: str = "") -> list[str]`
- Produces: `read_wdc_table(path: Path, input_root: Path, min_rows: int, min_cols: int, max_rows: int = 0) -> WdcTableResult`
- Produces: `WdcTableResult(source_table: dict[str, Any] | None, entities: list[dict[str, Any]], image_urls_by_entity: dict[str, list[str]], skip_reason: str | None, malformed_rows: int)`

- [ ] **Step 1: Write failing adapter tests**

```python
def test_read_wdc_table_removes_image_and_preserves_nested_values(tmp_path):
    path = write_gzip_rows(tmp_path, [
        {"row_id": 7, "name": "A", "geo": {"lat": "1"}, "image": ["/a.jpg"], "page_url": "https://x.test/p"},
        {"row_id": 8, "name": "B", "geo": {"lat": "2"}, "image": "https://cdn.test/b.jpg", "page_url": "https://x.test/q"},
    ])
    result = read_wdc_table(path, tmp_path, min_rows=2, min_cols=2)
    assert result.source_table is not None
    assert "image" not in [column["column_name"] for column in result.source_table["columns"]]
    assert get_cell_text(result.source_table["rows"][0], 1) == '{"lat":"1"}'
    assert result.image_urls_by_entity[result.entities[0]["entity_id"]] == ["https://x.test/a.jpg"]
```

- [ ] **Step 2: Run the adapter tests and confirm RED**

Run: `conda run -n MMDD python -m pytest tests/test_wdc_mm_joinability_dataset.py -q`

Expected: collection fails because `build_wdc_mm_joinability_dataset` does not exist.

- [ ] **Step 3: Implement the minimal adapter**

Implement a dataclass result, gzip line iteration with malformed-line isolation, first-seen column union excluding `row_id` and `image`, deterministic nested serialization, entity-column selection, stable IDs, internal cells, and shared profiles. Entity records must contain `entity_id`, compatibility `wiki_title`, `display_texts`, `context_terms`, `appears_in`, `page_url`, and `image_urls`.

- [ ] **Step 4: Run adapter tests and confirm GREEN**

Run: `conda run -n MMDD python -m pytest tests/test_wdc_mm_joinability_dataset.py -q`

Expected: adapter tests pass.

---

### Task 2: Generic HTML extraction and cached web client

**Files:**
- Modify: `scripts/build_wdc_mm_joinability_dataset.py`
- Modify: `tests/test_wdc_mm_joinability_dataset.py`

**Interfaces:**
- Produces: `extract_html_assets(html_text: str, base_url: str) -> tuple[str, list[str]]`
- Produces: `WdcWebClient.fetch_page(page_url: str) -> dict[str, Any] | None`
- Produces: `WdcWebClient.download_image(image_url: str, *, page_url: str, source: str, entity_id: str) -> dict[str, Any] | None`

- [ ] **Step 1: Write failing HTML/cache tests**

```python
def test_extract_html_assets_keeps_content_and_resolves_images():
    text, images = extract_html_assets(
        '<html><head><meta property="og:image" content="/hero.jpg"></head>'
        '<body><nav>menu</nav><h1>Entity A</h1><p>Useful description.</p>'
        '<img data-src="photo.png"><script>ignored()</script></body></html>',
        "https://example.test/path/page",
    )
    assert "Entity A" in text and "Useful description." in text
    assert "menu" not in text and "ignored" not in text
    assert images == ["https://example.test/hero.jpg", "https://example.test/path/photo.png"]
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `conda run -n MMDD python -m pytest tests/test_wdc_mm_joinability_dataset.py -q`

Expected: fails because HTML extraction/client behavior is missing.

- [ ] **Step 3: Implement HTML parsing, SQLite page cache, throttled requests, and bounded image validation**

Use `HTMLParser` with skip-depth tags, metadata/image attribute collection, `urljoin`, and URL-scheme validation. Store page extraction payloads and status in SQLite. Resolve once per redirect hop, reject any non-public address, and connect to the vetted IP while preserving the original HTTPS hostname for SNI/certificate checks. Stream downloads to a temporary path, enforce byte/pixel/deadline/total-cache/disk limits, reject external SVG and tiny/extreme rasters, atomically move valid images into the cache, and return bridge-asset metadata.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `conda run -n MMDD python -m pytest tests/test_wdc_mm_joinability_dataset.py -q`

Expected: HTML, cache, and bounded-download tests pass.

---

### Task 3: Per-entity text and shared image quota

**Files:**
- Modify: `scripts/build_wdc_mm_joinability_dataset.py`
- Modify: `tests/test_wdc_mm_joinability_dataset.py`

**Interfaces:**
- Produces: `build_wdc_bridge_assets_for_entity(entity: dict[str, Any], client: WdcWebClient, max_images_per_entity: int, text_asset_chunk_chars: int, min_text_asset_chunk_chars: int, max_text_asset_chunks_per_entity: int) -> list[dict[str, Any]]`

- [ ] **Step 1: Write a failing behavior test**

```python
def test_direct_images_take_quota_but_page_is_always_fetched():
    client = FakeClient(direct_successes=1, page_images=["https://x.test/1.jpg", "https://x.test/2.jpg"])
    records = build_wdc_bridge_assets_for_entity(entity_with_two_direct_urls(), client, 2, 800, 120, 3)
    assert client.fetch_page_calls == ["https://x.test/entity"]
    images = [record for record in records if record["asset_type"] == "image"]
    assert [record["source"] for record in images] == ["wdc_image_column", "wdc_page_image"]
    assert len(images) == 2
    assert any(record["asset_type"] == "text" for record in records)
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `conda run -n MMDD python -m pytest tests/test_wdc_mm_joinability_dataset.py -q`

Expected: quota/orchestration helper is missing.

- [ ] **Step 3: Implement direct-first image collection and mandatory page text processing**

Fetch the page unconditionally, split all extracted text with `max_chunks=0`, call `select_relevant_text_chunks`, write stable text assets, try direct images first, fill remaining image slots from the page, and deduplicate URL plus returned SHA-256.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run: `conda run -n MMDD python -m pytest tests/test_wdc_mm_joinability_dataset.py -q`

Expected: quota and chunk-limit tests pass.

---

### Task 4: Shared provenance compatibility

**Files:**
- Modify: `scripts/build_mm_joinability_dataset.py`
- Modify: `tests/test_mm_joinability_extraction.py`

**Interfaces:**
- Changes: `table_record()` reads `source_table.get("provenance_builder", "build_mm_joinability_dataset.py")`.

- [ ] **Step 1: Write a failing provenance test**

```python
def test_table_record_accepts_source_provenance_builder():
    record = table_record(source_table={**source_table_fixture(), "provenance_builder": "wdc"}, ...)
    assert record["provenance"]["builder"] == "wdc"
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `conda run -n MMDD python -m pytest tests/test_mm_joinability_extraction.py -q`

Expected: existing hard-coded builder yields the old value.

- [ ] **Step 3: Apply the one-line compatible implementation**

```python
"builder": clean_text(source_table.get("provenance_builder")) or "build_mm_joinability_dataset.py",
```

- [ ] **Step 4: Run the focused test and confirm GREEN**

Run: `conda run -n MMDD python -m pytest tests/test_mm_joinability_extraction.py -q`

Expected: all extraction tests pass.

---

### Task 5: WDC end-to-end builder and CLI

**Files:**
- Modify: `scripts/build_wdc_mm_joinability_dataset.py`
- Modify: `tests/test_wdc_mm_joinability_dataset.py`
- Modify: `README.md`
- Modify: `README.zh-CN.md`

**Interfaces:**
- Produces: `build_dataset(args: argparse.Namespace, *, web_client_factory: Callable[..., WdcWebClient] | None = None, extractor_factory: Callable[..., Any] | None = None) -> dict[str, Any]`
- Produces: `parse_args(argv: list[str] | None = None) -> argparse.Namespace`
- Produces CLI: `conda run -n MMDD python scripts/build_wdc_mm_joinability_dataset.py --input_dir wdc_schemaorg_2023 --output_dir output_wdc_mm_joinability`

- [ ] **Step 1: Write a failing small end-to-end test**

Create one gzip host table with at least six rows and four non-image columns. Inject a fake web client that returns three text chunks and two images, and a fake extractor whose recovered values match one candidate attribute for three rows. Assert one query, one target, one qrel, WDC provenance, exact five-row alignment, and no `image` column in any table artifact.

- [ ] **Step 2: Run the WDC suite and confirm RED**

Run: `conda run -n MMDD python -m pytest tests/test_wdc_mm_joinability_dataset.py -q`

Expected: build orchestration/CLI is incomplete.

- [ ] **Step 3: Implement the sharded pipeline around existing joinability helpers**

Write source/entity/assets/link shards, source splits, model task execution, query/target/qrel/evidence outputs, decisions, stats, and manifest. Preserve existing CLI names for model/query settings and add WDC web controls. Keep test injection private to Python calls and the production CLI straightforward.

- [ ] **Step 4: Document the command and operational constraints**

Add WDC build examples, cache/output paths, webpage-always-fetch rule, shared image quota, and model endpoint prerequisites to both README files.

- [ ] **Step 5: Run the complete relevant suites**

Run: `conda run -n MMDD python -m pytest tests/test_wdc_mm_joinability_dataset.py tests/test_mm_joinability_extraction.py tests/test_stage1_pipeline.py -q`

Expected: zero failures.

#### Post-review safety amendments

- Treat `--max_scanned_files` independently from `--max_source_tables`; count every attempted gzip, including malformed and rejected files.
- Detect per-table row truncation with bounded lookahead and expose safety configuration/truncation counters in startup logs, final logs, and `stats.json`.
- Use conservative defaults (`100` tables, `1000` attempted files, `100` rows, `2` images) and require `--allow_unbounded` for non-positive input bounds.
- Resolve A/AAAA/CNAME records with dnspython using the remaining absolute response deadline and no unbounded stdlib fallback. Pin actual TCP connections to the already validated DNS answer. Keep the original host in the HTTP `Host` header and in HTTPS SNI/certificate verification, and repeat validation/pinning on each manual redirect.
- Count page bodies and images under `--web_max_total_cache_bytes`, enforce the separate image quota, reserve quota and recheck free disk while streaming, and apply `--web_max_response_seconds` to drip responses.
- Backfill legacy image byte counts from verified cache files and charge recovered orphan files to both image and total-cache quotas.
- Reject untrusted SVG deliberately and enforce a decoded raster-pixel bound.

---

### Task 6: Validate extraction and launch the unattended build

**Files:**
- Runtime output: `output_wdc_mm_joinability/`
- Runtime cache: `cache/wdc_mm_joinability/`

- [ ] **Step 1: Verify archive operation state**

Run: `find wdc_schemaorg_2023 -type f \( -name '*_top100.zip' -o -name '*_minimum3.zip' -o -name '*_rest.zip' \) | wc -l`

Expected: `0`, unless the safe extractor stopped and reported insufficient space or an archive error.

- [ ] **Step 2: Run CLI help and a bounded smoke build**

Run: `conda run -n MMDD python scripts/build_wdc_mm_joinability_dataset.py --help`

Run a temporary one-table/five-entity fake-network or cached fixture build and inspect its manifest.

- [ ] **Step 3: Start a named tmux session**

Run: `tmux new-session -d -s wdc-mm-joinability 'conda run -n MMDD python scripts/build_wdc_mm_joinability_dataset.py --input_dir wdc_schemaorg_2023 --output_dir output_wdc_mm_joinability_run_001 --cache_dir cache/wdc_mm_joinability_run_001 --max_source_tables 100 --max_scanned_files 1000 --max_rows_per_source_table 100 --max_images_per_entity 2'`

- [ ] **Step 4: Monitor startup and correct any deterministic failure**

Inspect `tmux capture-pane`, process state, output shards, cache growth, and recent failure records. If the process exits, diagnose with `superpowers:systematic-debugging`, add a failing regression test, fix, rerun verification, and relaunch.
