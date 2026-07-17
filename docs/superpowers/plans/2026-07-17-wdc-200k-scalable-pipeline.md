# WDC 200K Scalable Multimodal Joinability Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable, bounded-memory pipeline that deterministically selects 200,000 WDC Schema.org 2023 tables, preserves every selected row, gives every entity one page-fetch opportunity, and emits the canonical multimodal joinability dataset artifacts.

**Architecture:** Keep final dataset artifacts compatible with `build_mm_joinability_dataset.py`, while moving selection, URL jobs, network cache, model queues, and checkpoints into separate work/cache directories. Process immutable shards stage by stage, globally deduplicate page and image URLs, persist both successes and failures, and materialize query artifacts one source table at a time with the existing query builder.

**Tech Stack:** Python 3.10+, pathlib, gzip/json/csv/zipfile, SQLite WAL, concurrent futures, dnspython, Pillow, pytest, existing MMDD joinability helpers and dynamic vLLM runner.

## Global Constraints

- Select exactly 200,000 valid tables when sufficient valid candidates exist.
- Use all valid `top100` tables, then allocate approximately 90% of remaining slots to `minimum3` and 10% to `rest`.
- Allocate class capacity by square-root weighting after base quotas of 250 `minimum3` and 50 `rest` tables per class.
- Cap one class at 40,000 selected tables.
- Use deterministic stable-hash selection with default seed 13 and deterministic reserve replacement.
- Preserve every row of every selected source table; no implicit or default row limit is allowed.
- Remove the WDC `image` column from all table artifacts.
- Every retained entity gets one `page_url` opportunity, including after a direct-image success.
- Default to one physical request and zero retries per unique URL, an eight-second absolute response deadline, 128 global workers, and two workers per host.
- Default to at most three attempted image URLs and three retained images per entity; both are independently configurable.
- Missing multimodal assets never remove a valid table or entity.
- Keep peak memory dependent on bounded shard size, not total tables/entities/assets.
- Persist successes and failures so resume does not replay completed network or model work.
- Keep final artifacts compatible with `scripts/build_mm_joinability_dataset.py`.
- Keep pipeline work under `work_dir` and reusable cache under `cache_dir`; do not mix them into `output_dir`.
- Maintain a configurable live disk reserve and stop safely before violating it.
- Use the `MMDD` conda environment for every Python and pytest command.

---

## File and Module Map

- Create `scripts/wdc200k_io.py`: atomic JSONL shards, checksummed stage manifests, SQLite job stores, external URL deduplication, and resume validation.
- Create `scripts/wdc200k_selection.py`: statistics-ZIP catalog loading, deterministic mixed stratification, reserve replacement, and selection manifest generation.
- Create `scripts/wdc200k_structural.py`: full-row WDC table expansion, canonical source/entity records, and page/direct-image reference shards.
- Create `scripts/wdc200k_fetch.py`: globally unique page/image job execution, durable outcome caching, host fairness, strict attempt budgets, and live counters.
- Create `scripts/wdc200k_assets.py`: existing-compatible text chunking, direct-image-first candidate planning, content linking, and bridge asset emission.
- Create `scripts/wdc200k_models.py`: persistent text/image model task queues and resumable extraction output.
- Create `scripts/wdc200k_materialize.py`: disk-backed asset/extraction joins and exact existing query/data-lake/qrels/evidence materialization.
- Create `scripts/build_wdc200k_mm_joinability_dataset.py`: consolidated CLI, stage orchestration, progress snapshots, invalidation policy, and final manifest.
- Modify `scripts/build_wdc_mm_joinability_dataset.py`: expose small public cache/result adapters needed by the staged fetcher without changing current behavior.
- Modify `scripts/run_mm_joinability_dynamic_vllm.py`: recognize staged model markers and keep direct stdout output.
- Modify `README.md` and `README.zh-CN.md`: document the 200K pipeline, output compatibility, safe defaults, resume, and tmux workflow.
- Create `tests/test_wdc200k_io.py`.
- Create `tests/test_wdc200k_selection.py`.
- Create `tests/test_wdc200k_structural.py`.
- Create `tests/test_wdc200k_fetch.py`.
- Create `tests/test_wdc200k_assets.py`.
- Create `tests/test_wdc200k_models.py`.
- Create `tests/test_wdc200k_materialize.py`.
- Create `tests/test_wdc200k_pipeline.py`.

---

### Task 1: Atomic Shards, Stage Manifests, and Durable Job Store

**Files:**
- Create: `scripts/wdc200k_io.py`
- Create: `tests/test_wdc200k_io.py`

**Interfaces:**
- Produces: `StageFingerprint`, `AtomicJsonlShard`, `StageManifest`, `SqliteJobStore`, `external_unique_jsonl(...)`, and `validate_completed_shard(...)`.
- Consumes: `stable_hash`, `write_jsonl_record`, and JSON helpers from `scripts/build_mm_table_dataset.py`.

- [ ] **Step 1: Write failing atomic-resume and durable-outcome tests**

```python
def test_atomic_shard_is_visible_only_after_commit(tmp_path: Path) -> None:
    shard = AtomicJsonlShard(tmp_path / "part-00000.jsonl")
    shard.write({"id": "a"})
    assert not shard.path.exists()
    record = shard.commit()
    assert shard.path.exists()
    assert record.records == 1
    assert validate_completed_shard(record, root=tmp_path)


def test_job_store_does_not_reclaim_terminal_outcomes(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue("page", "url-1", {"url": "https://example.test/a"})
    claimed = store.claim("page", limit=1, owner="worker-1")
    assert [job.job_id for job in claimed] == ["url-1"]
    store.finish("url-1", status="terminal", result={"error": "timeout"})
    assert store.claim("page", limit=1, owner="worker-2") == []
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_wdc200k_io.py -q
```

Expected: collection fails because `scripts.wdc200k_io` does not exist.

- [ ] **Step 3: Implement atomic shard commit and manifest records**

```python
@dataclass(frozen=True)
class CompletedShard:
    path: str
    records: int
    bytes: int
    sha256: str


class AtomicJsonlShard:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary_path = path.with_suffix(path.suffix + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.temporary_path.open("w", encoding="utf-8")
        self._records = 0

    def write(self, record: dict[str, Any]) -> None:
        write_jsonl_record(self._handle, record)
        self._records += 1

    def commit(self) -> CompletedShard:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        digest = sha256_path(self.temporary_path)
        size = self.temporary_path.stat().st_size
        self.temporary_path.replace(self.path)
        return CompletedShard(
            path=self.path.name,
            records=self._records,
            bytes=size,
            sha256=digest,
        )
```

Implement `StageManifest` as an atomically written JSON document containing
stage name, input fingerprint, parameter fingerprint, completed shards, totals,
and completion state.

- [ ] **Step 4: Implement the SQLite WAL job store**

Use one table with exact status values `pending`, `leased`, `success`,
`terminal`, and `retryable`. `claim()` must atomically lease only `pending`
records and reclaim expired leases, while `finish()` makes `success` and
`terminal` records ineligible for future claims.

```sql
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT,
    owner TEXT,
    lease_expires REAL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_kind_status
ON jobs(kind, status, updated_at);
```

- [ ] **Step 5: Implement bounded external JSONL uniqueness**

`external_unique_jsonl(input_paths, output_path, key_fn, chunk_records)` must
sort bounded in-memory chunks to temporary files, k-way merge them, and keep the
first record for each key. Tests must use `chunk_records=2` so multiple runs are
forced.

- [ ] **Step 6: Run focused tests**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_wdc200k_io.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/wdc200k_io.py tests/test_wdc200k_io.py
git commit -m "Add durable WDC pipeline storage"
```

---

### Task 2: Statistics Catalog and Deterministic 200K Selection

**Files:**
- Create: `scripts/wdc200k_selection.py`
- Create: `tests/test_wdc200k_selection.py`

**Interfaces:**
- Consumes: `AtomicJsonlShard` and `StageManifest` from Task 1.
- Produces: `TableCandidate`, `SelectionPolicy`, `read_statistics_catalog(...)`, `allocate_strata(...)`, `select_tables(...)`, and `replace_invalid_selection(...)`.

- [ ] **Step 1: Write failing statistics and allocation tests**

```python
def test_statistics_zip_restores_subset_and_filename(tmp_path: Path) -> None:
    archive = make_statistics_zip(
        tmp_path,
        schema_class="Product",
        subsets={"top100": [("shop.test", 10, 4)]},
    )
    records = list(read_statistics_catalog(archive))
    assert records == [
        TableCandidate(
            schema_class="Product",
            subset="top100",
            host="shop.test",
            relative_path="Product/Product_shop.test_October2023.json.gz",
            rows=10,
            columns=4,
        )
    ]


def test_mixed_allocation_is_exact_capped_and_reproducible() -> None:
    catalog = synthetic_catalog(classes=42, per_subset=10_000)
    first = select_tables(catalog, SelectionPolicy(target_tables=200_000, seed=13))
    second = select_tables(reversed(catalog), SelectionPolicy(target_tables=200_000, seed=13))
    assert [item.relative_path for item in first.selected] == [
        item.relative_path for item in second.selected
    ]
    assert len(first.selected) == 200_000
    assert max(Counter(item.schema_class for item in first.selected).values()) <= 40_000
    assert all(item in first.selected for item in catalog if item.subset == "top100")
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_wdc200k_selection.py -q
```

Expected: collection fails because selection interfaces do not exist.

- [ ] **Step 3: Parse the preserved statistics ZIP files**

Read these members with `zipfile.ZipFile.open()`:

```text
table_statistics/<Class>_October2023_statistics_top100.csv
table_statistics/<Class>_October2023_statistics_minimum3.csv
table_statistics/<Class>_October2023_statistics_rest.csv
```

Construct the extracted data filename as
`<Class>/<Class>_<host>_October2023.json.gz`, validate that it exists, and keep
the statistics row/column counts. Do not enumerate or open all 4,985,756 gzip
files during selection.

- [ ] **Step 4: Implement exact quota allocation**

```python
@dataclass(frozen=True)
class SelectionPolicy:
    target_tables: int = 200_000
    seed: int = 13
    minimum3_fraction: float = 0.90
    minimum3_base_per_class: int = 250
    rest_base_per_class: int = 50
    class_cap: int = 40_000
```

Implement base allocation, square-root weighted largest-remainder allocation,
class cap enforcement, cross-subset spill, and deterministic ordering by
`stable_hash(seed, relative_path)`.

- [ ] **Step 5: Implement reserve replacement**

`replace_invalid_selection(...)` must first consume the same class/subset
reserve, then use the deterministic global reserve. Record `replaces_path` and
`replacement_reason` in the manifest.

- [ ] **Step 6: Add a selection-only CLI smoke test**

Invoke:

```bash
conda run -n MMDD python scripts/wdc200k_selection.py \
  --input_dir <fixture> \
  --work_dir <tmp-work> \
  --target_tables 20 \
  --seed 13
```

Assert the manifest has exactly 20 valid records and stable hashes on rerun.

- [ ] **Step 7: Run focused tests**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_wdc200k_selection.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/wdc200k_selection.py tests/test_wdc200k_selection.py
git commit -m "Add stratified WDC table selection"
```

---

### Task 3: Full-Row Structural Expansion and URL Reference Shards

**Files:**
- Create: `scripts/wdc200k_structural.py`
- Create: `tests/test_wdc200k_structural.py`
- Modify: `scripts/build_wdc_mm_joinability_dataset.py`

**Interfaces:**
- Consumes: selection records from Task 2 and `read_wdc_table(...)` semantics from the current WDC builder.
- Produces: `expand_selected_shard(...)`, canonical source/entity shards, `page_refs` shards, and `direct_image_refs` shards.

- [ ] **Step 1: Write failing full-row and image-removal tests**

```python
def test_structural_expansion_preserves_all_rows_and_removes_image(tmp_path: Path) -> None:
    table_path = write_wdc_gzip(
        tmp_path,
        rows=[
            {"row_id": index, "name": f"n{index}", "page_url": f"https://e.test/{index}", "image": f"/{index}.jpg"}
            for index in range(37)
        ],
    )
    result = expand_selected_shard(
        [selection_record(table_path, rows=37)],
        output_root=tmp_path / "structural",
        input_root=tmp_path,
    )
    source = read_only_record(result.source_tables)
    assert len(source["rows"]) == 37
    assert "image" not in source["columns"]
    entities = read_all_records(result.entities)
    assert len(entities) == 37
    assert len(read_all_records(result.page_refs)) == 37
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_wdc200k_structural.py -q
```

Expected: import failure for `expand_selected_shard`.

- [ ] **Step 3: Expose an unlimited-row WDC adapter**

Add a public iterator to `build_wdc_mm_joinability_dataset.py`:

```python
def iter_wdc_rows(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload
```

Keep existing `read_wdc_table(...)` behavior unchanged for current callers.

- [ ] **Step 4: Stream source and entity records**

Process one selected table at a time. Derive the same source table IDs, entity
IDs, cells, row IDs, context terms, and provenance used by the current WDC
adapter. Store direct image URLs only on entity/reference records; remove the
column before building the source table.

- [ ] **Step 5: Emit every entity page reference**

Each entity record with a usable `page_url` emits:

```json
{
  "url_key": "<stable normalized URL hash>",
  "page_url": "https://example.test/entity",
  "entity_id": "ent_...",
  "source_table_id": "st_...",
  "row_id": 7
}
```

Entities with missing/invalid page URLs remain in structural output and emit a
terminal structural failure record.

- [ ] **Step 6: Add shard-resume verification**

Interrupt after one committed structural shard, resume, and assert the first
shard checksum and modification time remain unchanged.

- [ ] **Step 7: Run focused and adapter regression tests**

Run:

```bash
conda run -n MMDD python -m pytest \
  tests/test_wdc200k_structural.py \
  tests/test_wdc_mm_joinability_dataset.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/wdc200k_structural.py scripts/build_wdc_mm_joinability_dataset.py tests/test_wdc200k_structural.py
git commit -m "Stream full WDC source tables"
```

---

### Task 4: Globally Deduplicated Page Fetch Stage

**Files:**
- Create: `scripts/wdc200k_fetch.py`
- Create: `tests/test_wdc200k_fetch.py`
- Modify: `scripts/build_wdc_mm_joinability_dataset.py`

**Interfaces:**
- Consumes: unique page jobs from Tasks 1 and 3.
- Produces: `FetchPolicy`, `fetch_unique_pages(...)`, durable page outcomes, failure shards, and progress snapshots.

- [ ] **Step 1: Write failing request-budget and resume tests**

```python
def test_duplicate_page_urls_make_one_physical_request(tmp_path: Path) -> None:
    transport = CountingTransport({"https://e.test/a": successful_html("hello")})
    result = fetch_unique_pages(
        page_refs=[
            page_ref("e1", "https://e.test/a"),
            page_ref("e2", "https://e.test/a"),
        ],
        store=SqliteJobStore(tmp_path / "pages.sqlite3"),
        transport=transport,
        policy=FetchPolicy(retries=0, deadline_seconds=8),
    )
    assert transport.calls == ["https://e.test/a"]
    assert result.success == 1


def test_terminal_page_failure_is_not_replayed_on_resume(tmp_path: Path) -> None:
    transport = CountingTransport({"https://e.test/a": TimeoutError()})
    store = SqliteJobStore(tmp_path / "pages.sqlite3")
    fetch_unique_pages([page_ref("e1", "https://e.test/a")], store, transport, FetchPolicy())
    fetch_unique_pages([page_ref("e1", "https://e.test/a")], store, transport, FetchPolicy())
    assert transport.calls == ["https://e.test/a"]
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_wdc200k_fetch.py -q
```

Expected: missing fetch-stage interfaces.

- [ ] **Step 3: Add public outcome adapters to `WdcWebClient`**

Expose read-only methods:

```python
def cached_page_outcome(self, page_url: str) -> dict[str, Any] | None: ...
def cached_image_outcome(self, image_url: str) -> dict[str, Any] | None: ...
```

Add explicit negative image-cache records so validation/download failures are
durable for a policy version. Existing current-builder tests must continue to
pass.

- [ ] **Step 4: Implement bounded fair scheduling**

Use a fixed `ThreadPoolExecutor(max_workers=global_concurrency)` around the
existing pinned synchronous transport. Maintain a semaphore of size
`per_host_concurrency` per normalized hostname. Submit replacement work as each
future completes; do not wait for fixed batches.

```python
@dataclass(frozen=True)
class FetchPolicy:
    retries: int = 0
    deadline_seconds: float = 8.0
    global_concurrency: int = 128
    per_host_concurrency: int = 2
```

- [ ] **Step 5: Persist outcomes before releasing jobs**

For each unique URL, write the cache result and mark the job `success` or
`terminal` in one ordered completion path. A process crash between cache write
and job finish must be repaired by checking the cache before a reclaimed job
makes a request.

- [ ] **Step 6: Test real timeout, host limits, and crash repair**

Use local HTTP servers to verify an eight-second policy can interrupt DNS,
headers, and body paths through the existing transport, and a counting barrier
to prove no host exceeds two concurrent requests.

- [ ] **Step 7: Run focused and network regression tests**

Run:

```bash
conda run -n MMDD python -m pytest \
  tests/test_wdc200k_fetch.py \
  tests/test_wdc_mm_joinability_dataset.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/wdc200k_fetch.py scripts/build_wdc_mm_joinability_dataset.py tests/test_wdc200k_fetch.py
git commit -m "Add resumable WDC URL fetching"
```

---

### Task 5: Text Assets, Image Planning, and Unique Image Fetching

**Files:**
- Create: `scripts/wdc200k_assets.py`
- Create: `tests/test_wdc200k_assets.py`

**Interfaces:**
- Consumes: entity/reference shards, page outcomes, and `WdcWebClient.download_image`.
- Produces: `ImageBudget`, `plan_entity_assets(...)`, unique image jobs, canonical bridge assets, and table-asset links.

- [ ] **Step 1: Write failing ordering and budget tests**

```python
def test_every_entity_uses_direct_first_then_page_images() -> None:
    plan = plan_entity_assets(
        entity=entity(image_urls=["https://i.test/direct.jpg"]),
        page=page(image_urls=["https://i.test/a.jpg", "https://i.test/b.jpg", "https://i.test/c.jpg"]),
        budget=ImageBudget(attempts_per_entity=3, retained_per_entity=3),
    )
    assert [job.image_url for job in plan.image_refs] == [
        "https://i.test/direct.jpg",
        "https://i.test/a.jpg",
        "https://i.test/b.jpg",
    ]
    assert plan.page_was_required is True


def test_image_failure_does_not_remove_entity_or_table() -> None:
    result = materialize_entity_assets(entity("e1"), page=None, image_outcomes={})
    assert result.bridge_assets == []
    assert result.entity_id == "e1"
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_wdc200k_assets.py -q
```

Expected: missing asset planner interfaces.

- [ ] **Step 3: Reuse canonical text splitting**

Call the same `chunk_text` and relevance selection helpers currently used by
`build_wdc_bridge_assets_for_entity`, preserving defaults of 800 maximum
characters, 120 minimum characters, and three selected chunks.

- [ ] **Step 4: Emit image mappings before global dedup**

Store one entity-to-image URL mapping for each of the first
`attempts_per_entity` distinct candidates. Then use
`external_unique_jsonl(...)` to create one physical image job per URL.

- [ ] **Step 5: Fetch images with durable negative cache**

Call the Task 4 scheduler with image jobs, zero retries, eight-second deadline,
and existing raster validation. Store successful content in the cache image
directory by SHA-256 and preserve original/final URL provenance.

- [ ] **Step 6: Materialize canonical bridge assets and links**

Retain at most `retained_per_entity` successful images in original candidate
order. Emit bridge asset and table-asset-link records with the exact fields used
by current readers.

- [ ] **Step 7: Run focused tests**

Run:

```bash
conda run -n MMDD python -m pytest \
  tests/test_wdc200k_assets.py \
  tests/test_mm_joinability_extraction.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/wdc200k_assets.py tests/test_wdc200k_assets.py
git commit -m "Plan and materialize WDC assets"
```

---

### Task 6: Persistent Model Task Queues

**Files:**
- Create: `scripts/wdc200k_models.py`
- Create: `tests/test_wdc200k_models.py`
- Modify: `scripts/run_mm_joinability_dynamic_vllm.py`

**Interfaces:**
- Consumes: canonical bridge asset shards and existing `ExtractionTask`, `LocalAttributeExtractor`, and cache helpers.
- Produces: `enqueue_model_tasks(...)`, `run_model_stage(...)`, extraction shards, model failure shards, and staged start/ready/done markers.

- [ ] **Step 1: Write failing model resume test**

```python
def test_model_stage_resumes_without_repeating_success(tmp_path: Path) -> None:
    extractor = CountingExtractor()
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    enqueue_model_tasks([text_asset("a"), image_asset("b")], store)
    run_model_stage(store, extractor, stop_after=1)
    run_model_stage(store, extractor)
    assert Counter(extractor.asset_ids) == {"a": 1, "b": 1}
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_wdc200k_models.py -q
```

Expected: missing model-stage interfaces.

- [ ] **Step 3: Enqueue stable extraction keys**

Use the existing `extraction_cache_key(...)`, prompt version, model identity,
and asset fingerprint. Store the complete extraction task payload in the job
store so a worker does not need all assets in memory.

- [ ] **Step 4: Run bounded task groups and commit each outcome**

Reuse `run_extraction_task_group(...)` in bounded groups. Persist the canonical
extraction record before marking the task successful. Persist model errors with
the same schema as `model_attribute_errors.jsonl`.

- [ ] **Step 5: Extend dynamic markers for staged orchestration**

The model stage writes start counts only after network/assets are complete,
waits for ready, and writes text/image completion markers independently.
`run_mm_joinability_dynamic_vllm.py` must keep direct stdout/stderr and stop all
model process groups on runner exit.

- [ ] **Step 6: Run focused and dynamic-runner tests**

Run:

```bash
conda run -n MMDD python -m pytest \
  tests/test_wdc200k_models.py \
  tests/test_mm_joinability_extraction.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/wdc200k_models.py scripts/run_mm_joinability_dynamic_vllm.py tests/test_wdc200k_models.py
git commit -m "Add resumable WDC model queues"
```

---

### Task 7: Streaming Canonical Dataset Materialization

**Files:**
- Create: `scripts/wdc200k_materialize.py`
- Create: `tests/test_wdc200k_materialize.py`

**Interfaces:**
- Consumes: structural shards, bridge assets, links, extraction outcomes, and existing query-building functions.
- Produces: canonical final output artifacts and `materialize_dataset_shard(...)`.

- [ ] **Step 1: Write failing query-parity test**

```python
def test_materializer_matches_existing_query_builder(tmp_path: Path) -> None:
    fixture = canonical_joinability_fixture()
    expected = join_builder.build_table_join_records(
        fixture.source_table,
        fixture.entity_to_assets,
        fixture.assets,
        fixture.extractions,
        fixture.args,
        split="train",
    )
    actual = materialize_dataset_shard(fixture.to_disk(tmp_path), fixture.args)
    assert actual.query_tables == expected[0]
    assert actual.data_lake_tables == expected[1]
    assert actual.qrels == expected[2]
    assert actual.decision == expected[3]
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_wdc200k_materialize.py -q
```

Expected: missing materializer.

- [ ] **Step 3: Build disk-backed lookup indexes**

Create SQLite tables keyed by entity ID, asset ID, and extraction key. Insert
records by streaming work shards. Do not call the existing `load_assets(...)`
or build global Python dictionaries.

- [ ] **Step 4: Materialize one source table at a time**

Load only one source table plus its referenced entities/assets/extractions,
construct the small dictionaries expected by `build_table_join_records(...)`,
call it directly, and append records to atomic final-output shards.

- [ ] **Step 5: Preserve canonical artifacts**

Write:

```text
source_tables/
query_tables/
data_lake_tables/
entities/
bridge_assets/
table_asset_links/
attribute_extractions/
evidence_recoveries/
qrels.jsonl
splits.json
stats.json
table_queryability_decisions.jsonl
dataset_manifest.json
```

Include WDC-only failure files as additional top-level diagnostics without
changing core artifact schemas.

- [ ] **Step 6: Test empty multimodal entities and tables**

Assert a table with no successful assets remains in source/data-lake output and
has zero asset links rather than being rejected for missing media alone.

- [ ] **Step 7: Run focused and canonical regression tests**

Run:

```bash
conda run -n MMDD python -m pytest \
  tests/test_wdc200k_materialize.py \
  tests/test_mm_joinability_extraction.py \
  tests/test_stage1_pipeline.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/wdc200k_materialize.py tests/test_wdc200k_materialize.py
git commit -m "Stream WDC dataset materialization"
```

---

### Task 8: Consolidated CLI, Invalidation, and Live Progress

**Files:**
- Create: `scripts/build_wdc200k_mm_joinability_dataset.py`
- Create: `tests/test_wdc200k_pipeline.py`

**Interfaces:**
- Consumes: stage entry points from Tasks 2–7.
- Produces: `PipelineConfig`, `run_pipeline(...)`, the complete CLI, stage invalidation, progress snapshots, and dry-run/preflight modes.

- [ ] **Step 1: Write failing CLI/default tests**

```python
def test_cli_defaults_match_approved_policy(tmp_path: Path) -> None:
    args = parse_args(["--input_dir", str(tmp_path), "--output_dir", str(tmp_path / "out")])
    assert args.max_source_tables == 200_000
    assert args.max_rows_per_source_table is None
    assert args.web_max_retries == 0
    assert args.web_max_response_seconds == 8
    assert args.web_global_concurrency == 128
    assert args.web_per_host_concurrency == 2
    assert args.max_image_attempts_per_entity == 3
    assert args.max_images_per_entity == 3
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_wdc200k_pipeline.py -q
```

Expected: consolidated entry point does not exist.

- [ ] **Step 3: Implement stage orchestration**

Stages are exactly:

```python
STAGES = (
    "selection",
    "structural",
    "pages",
    "asset_planning",
    "images",
    "models",
    "materialize",
)
```

`--resume` is the default. `--from_stage` invalidates the named stage and all
downstream stage manifests after checking that upstream fingerprints still
match.

- [ ] **Step 4: Implement preflight and dry-run**

`--stop_after structural` must emit exact table/entity/unique-page/direct-image
counts and request/disk upper bounds without performing network work.

- [ ] **Step 5: Implement progress snapshots**

Write `work_dir/progress.json` atomically every five seconds with stage,
completed/total shards, counters, rolling rates, ETA, and disk usage. Also log
the same summary at bounded intervals to direct stdout.

- [ ] **Step 6: Add end-to-end synthetic interruption test**

Run a small fixture through page fetching, terminate after a committed shard,
resume, and assert physical request counters do not increase for completed
successes or failures.

- [ ] **Step 7: Run complete relevant suite**

Run:

```bash
conda run -n MMDD python -m pytest \
  tests/test_wdc200k_io.py \
  tests/test_wdc200k_selection.py \
  tests/test_wdc200k_structural.py \
  tests/test_wdc200k_fetch.py \
  tests/test_wdc200k_assets.py \
  tests/test_wdc200k_models.py \
  tests/test_wdc200k_materialize.py \
  tests/test_wdc200k_pipeline.py \
  tests/test_wdc_mm_joinability_dataset.py \
  tests/test_mm_joinability_extraction.py \
  tests/test_stage1_pipeline.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/build_wdc200k_mm_joinability_dataset.py tests/test_wdc200k_pipeline.py
git commit -m "Add WDC 200K pipeline CLI"
```

---

### Task 9: Documentation, Scale Gates, Pilot Shutdown, and Formal Launch

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `requirements.txt` only if implementation introduces a new runtime dependency.
- Create: `docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md`

**Interfaces:**
- Consumes: the consolidated CLI and progress snapshot.
- Produces: operator documentation, scale evidence, final verification, and the formal tmux run.

- [ ] **Step 1: Document exact output/work/cache separation**

Include commands for preflight, resume, stage refresh, direct execution, dynamic
vLLM execution, and tmux progress viewing. State that source-table rows are
never capped.

- [ ] **Step 2: Run the 100-table acceptance gate**

Run with fresh work/output/cache directories and local-safe network policy.
Interrupt once, resume, and record table/entity counts, duplicate physical
request count, peak RSS, disk growth, and final artifact validation.

- [ ] **Step 3: Run the 1,000-table acceptance gate**

Require:

- stable or sublinear peak RSS growth;
- zero replay of terminal URL outcomes;
- exact source row preservation;
- valid manifest checksums;
- progress ETA within a factor of two over the final half of the run.

- [ ] **Step 4: Run the 10,000-table structural/preflight gate**

Complete selection and structural stages first. Record exact entities, unique
pages, image candidates, and projected disk/request totals. Continue network
stages only if the live disk reserve remains satisfiable.

- [ ] **Step 5: Run fresh full verification**

Run:

```bash
conda run -n MMDD python -m pytest \
  tests/test_wdc200k_io.py \
  tests/test_wdc200k_selection.py \
  tests/test_wdc200k_structural.py \
  tests/test_wdc200k_fetch.py \
  tests/test_wdc200k_assets.py \
  tests/test_wdc200k_models.py \
  tests/test_wdc200k_materialize.py \
  tests/test_wdc200k_pipeline.py \
  tests/test_wdc_mm_joinability_dataset.py \
  tests/test_mm_joinability_extraction.py \
  tests/test_stage1_pipeline.py -q
conda run -n MMDD python -m py_compile \
  scripts/wdc200k_io.py \
  scripts/wdc200k_selection.py \
  scripts/wdc200k_structural.py \
  scripts/wdc200k_fetch.py \
  scripts/wdc200k_assets.py \
  scripts/wdc200k_models.py \
  scripts/wdc200k_materialize.py \
  scripts/build_wdc200k_mm_joinability_dataset.py
git diff --check
```

Expected: pytest and compilation exit zero; `git diff --check` has no output.

- [ ] **Step 6: Commit documentation and scale report**

```bash
git add README.md README.zh-CN.md requirements.txt docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md
git commit -m "Document WDC 200K pipeline operations"
```

- [ ] **Step 7: Stop the obsolete monolithic pilot**

Gracefully stop `wdc_mm_joinability` only after the new 100-table resume gate
passes. Preserve its cache as diagnostic input; do not merge it into formal
work/cache/output directories.

- [ ] **Step 8: Launch the formal selection and structural preflight**

Create a new tmux session named `wdc_200k` with direct stdout and a `progress`
window reading `work_wdc_200k/progress.json`. Use:

```bash
conda run -n MMDD python scripts/build_wdc200k_mm_joinability_dataset.py \
  --input_dir wdc_schemaorg_2023 \
  --output_dir output_wdc_200k \
  --work_dir work_wdc_200k \
  --cache_dir cache/wdc_200k \
  --max_source_tables 200000 \
  --selection_seed 13 \
  --web_max_retries 0 \
  --web_max_response_seconds 8 \
  --web_global_concurrency 128 \
  --web_per_host_concurrency 2 \
  --max_image_attempts_per_entity 3 \
  --max_images_per_entity 3 \
  --stop_after structural
```

Inspect the exact preflight entity/request/disk totals before resuming network
stages with the same directories and `--resume`.

- [ ] **Step 9: Monitor the first network interval**

Confirm page success/failure counters move, completed terminal outcomes are not
replayed, RSS and disk remain within limits, and the progress ETA stabilizes.
Fix any observed blocker through the systematic-debugging and TDD workflow
before resuming the formal run.

---

## Plan Self-Review Results

- Spec coverage: selection, full-row preservation, page opportunity, image
  budgets, URL deduplication, negative caching, model resume, canonical output,
  disk safety, progress, staged validation, and formal launch are each assigned
  to a task.
- Placeholder scan: the plan contains no deferred implementation placeholders.
- Type consistency: the job store, shard, selection, fetch policy, image budget,
  model queue, materializer, and pipeline interfaces are introduced before
  downstream tasks consume them.
- Scope: all tasks contribute to one coherent staged pipeline; none is an
  independently deployable unrelated subsystem.
