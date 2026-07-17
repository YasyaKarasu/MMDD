# WDC Schema.org 2023 200K Scalable Multimodal Joinability Pipeline

## Objective

Build a reproducible multimodal joinability discovery dataset from exactly
200,000 valid WDC Schema.org Table Corpus 2023 host tables.

The selected source tables retain every row. Every retained entity receives one
opportunity to fetch its `page_url`, but the dataset does not require every
entity to have successful text or image assets. Missing multimodal evidence
never removes an otherwise valid entity or source table.

The final dataset must remain compatible with the artifact layout and record
schemas produced by `scripts/build_mm_joinability_dataset.py`. Large-scale
selection, crawling, and inference state lives outside the final output
directory.

## Constraints

- Select exactly 200,000 valid tables when the input contains enough valid
  candidates.
- Preserve every row of every selected source table. There is no per-table row
  cap.
- Remove the WDC `image` column from all source, query, target, and raw-table
  records.
- Always give every retained entity one `page_url` fetch opportunity, including
  entities whose direct WDC image URL succeeds.
- Default to at most three attempted image URLs and at most three retained
  images per entity. Both limits are configurable.
- Reuse the existing query, target, qrels, evidence recovery, split, and model
  extraction semantics.
- Keep memory usage bounded by shard size rather than by the number of selected
  tables or entities.
- Persist both successful and failed network outcomes so a restart does not
  replay completed requests.
- Stop safely before exhausting configured disk reserves.

## Deterministic Stratified Table Selection

Selection uses the cross-product of the 42 schema.org classes and the WDC
`top100`, `minimum3`, and `rest` subsets.

1. Include every valid `top100` table.
2. Let `R` be the number of slots remaining after the valid `top100` tables are
   included.
3. Initially allocate `round(0.90 * R)` slots to `minimum3` and the remainder
   to `rest`.
4. Within `minimum3`, give each class a base allocation of
   `min(250, available_tables)`.
5. Within `rest`, give each class a base allocation of
   `min(50, available_tables)`.
6. Allocate the remaining slots using weights proportional to the square root
   of each class's remaining available table count.
7. Cap the complete contribution of any one class at 40,000 tables, or 20% of
   the final dataset.
8. Redistribute capacity shortfalls using deterministic largest-remainder
   allocation. Unfilled `rest` capacity spills to `minimum3` first; any
   remaining shortfall is filled from any class/subset that is still below its
   cap.
9. Rank candidates inside each class/subset by a stable hash of the relative
   path and selection seed. The default seed is 13.
10. Keep the candidates immediately after the selected range as an ordered
    reserve pool. Malformed or unreadable selected files are replaced from the
    same class/subset reserve first, then through the deterministic global
    redistribution rule.

The immutable selection manifest records relative path, class, subset, rank,
row count, column count, content hash, selection seed, and replacement
provenance. Repeating selection against unchanged input produces the same
manifest.

## Pipeline Architecture

### Stage 1: Selection and Structural Expansion

The selector produces `selected_tables.jsonl` and `reserve_tables.jsonl` in the
work directory. The structural worker then reads selected tables one at a time,
validates them, preserves all rows, and emits sharded source-table and entity
records.

For each entity it also emits:

- one page job reference containing the normalized `page_url`;
- ordered direct image candidates discovered from the removed WDC `image`
  value;
- table, row, and entity identifiers needed to reconstruct asset links.

No all-entity dictionary is retained in memory.

Before network work starts, the stage reports the exact valid table count,
entity count, unique page URL count, direct image candidate count, structural
output size, and a request/disk upper-bound estimate.

### Stage 2: Unique Page Fetching

Page job shards are externally sorted and globally deduplicated by normalized
URL. Multiple entities referencing the same URL share one physical request and
one cached outcome.

Defaults:

- one physical request per unique page URL;
- zero retries;
- eight-second absolute deadline covering DNS, connect, headers, and body;
- 128 global asynchronous requests;
- at most two simultaneous requests per host;
- existing userinfo, public-IP, redirect, response-byte, wall-clock, and disk
  safety checks.

Both successes and failures are durable terminal outcomes for the run. A
restart does not retry either outcome unless the operator explicitly requests a
cache refresh.

Each successful page result stores the final URL, bounded visible text, and
ordered page image candidates.

### Stage 3: Text Assets and Image Planning

Visible page text is split and selected with the same chunking and quantity
rules as `build_mm_joinability_dataset.py`.

Each entity builds its ordered image candidate list from:

1. direct WDC `image` candidates;
2. image candidates extracted from the entity page.

Duplicate URLs are removed while preserving first occurrence. By default only
the first three distinct candidates enter the image job mapping. The CLI option
`--max_image_attempts_per_entity` controls this limit.

Page fetching is never skipped because a direct image is present or succeeds.

### Stage 4: Unique Image Fetching

Selected image candidates are globally deduplicated by normalized URL and
downloaded asynchronously. Each unique URL receives one request, zero retries,
and the same default eight-second absolute deadline.

Images retain the existing MIME, byte, pixel, aspect-ratio, SSRF, redirect, and
disk safety validation. Valid content is stored by SHA-256 so different URLs
with identical bytes share one file.

Each entity retains at most three successful images by default. The CLI option
`--max_images_per_entity` controls this independent retention limit.

### Stage 5: Model Attribute Extraction

Only successful text and image assets generate model tasks. Text and image
tasks use separate persistent queues and caches. Completed tasks are committed
individually or in bounded batches, so interruption does not repeat successful
model calls.

The existing prompt versions, parsers, evidence requirements, cache semantics,
and dynamic dual-vLLM orchestration remain authoritative.

### Stage 6: Joinability Dataset Materialization

Materialization processes one source-table shard at a time. It joins entities,
assets, asset links, and model extractions through disk-backed indexes, then
calls the existing `build_table_join_records` logic without changing query
construction semantics.

Output shards are written to temporary paths, checksummed, and atomically
published. Only published shards appear in `dataset_manifest.json`.

## Output Compatibility

The final output directory uses the canonical current
`build_mm_joinability_dataset.py` layout:

```text
output_wdc_200k/
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
  media_download_failures.jsonl
  model_attribute_errors.jsonl
  web_fetch_failures.jsonl
```

Core artifact record schemas, shard manifests, qrels, splits, and evidence paths
remain compatible with existing readers. The dataset manifest may add
backward-compatible `wdc_sampling`, `web_crawl`, and source provenance metadata.

Pipeline-internal state must not be placed in the final output:

```text
work_wdc_200k/
  selection/
  structural_jobs/
  page_jobs/
  image_jobs/
  model_jobs/
  checkpoints/
  stage_manifests/
```

Reusable network, media, and inference cache state is also separate:

```text
cache/wdc_200k/
  page_cache/
  image_cache/
  images/
  model_attribute_extractions/
  negative_cache/
```

## Checkpointing and Recovery

- Each work shard has an input fingerprint, parameter fingerprint, record
  count, checksum, and atomic `.done` marker.
- Resume validates the marker and checksum before skipping a shard.
- Page and image outcomes are keyed by normalized URL and include success or
  failure status, timestamp, policy version, and payload checksum.
- Network policy changes invalidate only affected network stages.
- Model prompt or model identity changes invalidate only affected inference and
  downstream materialization stages.
- Selection or structural schema changes invalidate all downstream stages.
- No stage deletes a previously valid checkpoint until its replacement is
  atomically committed.

## Progress and Operations

The tmux session shows direct process output plus a live progress window. The
progress view reports:

- current stage and completed/total shards;
- selected and validated tables;
- emitted entities;
- unique and completed page jobs;
- page success, terminal failure, and timeout counts;
- planned, completed, valid, and failed image jobs;
- completed text and image model tasks;
- current and rolling throughput;
- estimated remaining time;
- cache, work, output, and free disk bytes.

The pipeline continuously checks configured disk reserve. It stops at a shard
boundary when possible and immediately before any write that would violate the
reserve.

## CLI Defaults

The consolidated entry point exposes at least:

- `--max_source_tables 200000`
- `--selection_seed 13`
- `--top100_policy all`
- `--minimum3_fraction 0.90`
- `--class_max_tables 40000`
- `--page_attempts 1`
- `--image_attempts 1`
- `--web_max_retries 0`
- `--web_max_response_seconds 8`
- `--web_global_concurrency 128`
- `--web_per_host_concurrency 2`
- `--max_image_attempts_per_entity 3`
- `--max_images_per_entity 3`
- `--work_dir`
- `--cache_dir`
- `--resume`
- `--refresh_page_cache`
- `--refresh_image_cache`

There is no default or implicit source-table row limit.

## Failure Semantics

- A malformed candidate table is replaced deterministically during selection.
- Page or image failure does not remove an entity or table.
- An entity with no successful multimodal assets remains in structural
  artifacts and may have no table-asset links.
- A table with no successful multimodal entities remains in the dataset.
- Failed URLs are negative-cached for the run and reported with normalized URL,
  stage, error class, HTTP status when available, and affected reference count.
- Unexpected worker exceptions fail only the current job shard; the stage
  records the exception and remains resumable.

## Verification and Acceptance

Automated tests use small synthetic gzip tables and local HTTP servers. They
must verify:

- exact deterministic stratified allocation and reserve replacement;
- exactly 200,000 valid selections when sufficient valid input exists;
- preservation of every row in selected source tables;
- removal of the `image` column from every table artifact;
- one page job per entity reference and one physical request per unique URL;
- direct-image-first ordering and configurable three-attempt/three-retained
  defaults;
- no page skip after direct image success;
- durable success and negative caching across interruption;
- bounded memory with increasing table and entity counts;
- atomic shard recovery and checksum validation;
- disk-reserve shutdown;
- final manifest and record compatibility with the existing builder;
- byte-for-byte equivalent query construction for identical synthetic source,
  entity, asset, and extraction inputs.

Before the formal 200K run, a staged acceptance run must complete at increasing
scales, such as 100, 1,000, and 10,000 selected tables. Each scale reuses the
same pipeline code and validates throughput, memory, disk growth, resume
behavior, and ETA calculation. The existing 100-table monolithic pilot is not a
performance baseline for the new pipeline.
