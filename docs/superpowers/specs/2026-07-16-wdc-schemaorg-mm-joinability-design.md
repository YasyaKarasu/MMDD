# WDC Schema.org 2023 Multimodal Joinability Dataset Design

## Goal

Build a multimodal joinability-discovery dataset from the local WDC Schema.org Table Corpus 2023 while preserving the query/target construction and model-based recovery rules in `scripts/build_mm_joinability_dataset.py`.

The source corpus contains one JSONL-Gzip table per schema.org class/host. Each row has a generated `row_id`, a `page_url`, and schema.org attributes. The input `image` attribute is an asset source, not a dataset column.

## Confirmed requirements

- Read all extracted `*.json.gz` host tables under `wdc_schemaorg_2023`, with `--max_source_tables` available for bounded runs.
- Always fetch every selected entity's `page_url` for visible text and page-image candidates.
- Also try every usable URL found recursively in the row's `image` value.
- Direct `image`-column downloads consume the shared `max_images_per_entity` quota first. Page images only fill the remaining quota.
- A successful direct image must never suppress page text fetching.
- Deduplicate image candidates and downloaded image content.
- Reuse `split_text_asset_content` and `select_relevant_text_chunks` without changing their defaults: 800 maximum characters, 120 minimum characters, and 3 selected chunks per entity.
- Remove the `image` column from source, query, target, and raw data-lake tables.
- Retain `page_url` as a source column and as asset provenance, but never treat it as an image source.
- Reuse the existing query construction implementation, including five aligned rows by default, recovery threshold semantics, best-column selection, context ordering, qrels, and evidence paths.
- Network and per-entity failures are recorded and skipped; they do not abort the full build.

## Input adapter

`scripts/build_wdc_mm_joinability_dataset.py` converts each WDC host file into the existing internal source-table schema.

- Columns are the stable first-seen union of row keys, excluding `image` and `row_id`.
- Nested values are rendered as deterministic compact JSON; scalar values use the shared text cleaner.
- The entity display column is selected in this order: `name`, `headline`, `title`, `identifier`, then `page_url`.
- The selected entity cell receives an internal entity lookup key in its existing `wiki_title` compatibility field. The value is a stable hash of class, source file, source row id, page URL, and display text, so rows sharing a page remain distinct.
- `metadata.candidate_entity_columns` contains only the selected entity column. Column profiles use the shared Stage-1 profiling helper.
- Malformed lines are counted and skipped. A file with fewer than `min_rows` valid rows or fewer than `min_cols` output columns is rejected.
- `--max_rows_per_source_table 0` means unlimited. Bounded operational runs may set a positive cap without changing query construction over the retained source rows.

## Generic webpage and image assets

The generic web client uses `requests`, a descriptive configurable User-Agent, bounded response sizes, retries, connect/read timeouts, and a host-level minimum interval. HTML parsing uses the standard library and extracts:

- document title and description metadata;
- visible paragraph/heading/list text while excluding script, style, template, SVG, navigation, footer, form, and noscript content;
- OpenGraph/Twitter images and ordinary/lazy-loaded image URLs, resolved relative to the final page URL.

For every entity, the client performs the page fetch regardless of direct-image results. Text is split and relevance-selected by the existing helpers. Image download order is:

1. recursively extracted `image`-column URLs;
2. parsed page image URLs;
3. stop when `max_images_per_entity` successful unique images have been saved.

Downloads are streamed with a byte limit. Raster images are verified with Pillow; SVG is rasterized through the repository's existing SVG helper. Tiny or extreme-aspect images and duplicate hashes are rejected. Saved bridge asset records retain source (`wdc_image_column` or `wdc_page_image`), original/final URL, page URL, local/relative path, bytes, SHA-256, width, height, and MIME type.

## Cache, errors, and repeatability

- A SQLite cache under `<cache_dir>/wdc_web.sqlite3` stores successful page extraction results and terminal/retryable failure metadata.
- Images live under `<cache_dir>/wdc_images` using stable URL-derived names; successful files are verified before reuse.
- The build writes `web_fetch_failures.jsonl` and `media_download_failures.jsonl` in the output directory.
- Stable hashes determine source table, entity, asset, split, query, target, and recovery identifiers.
- Output directories follow the existing sharded artifact layout and manifest schema. WDC-specific counts and cache paths are added to `stats.json` and `dataset_manifest.json`.
- Output shards are rewritten on restart, matching the existing builder. Network and model caches make the expensive stages reusable.

## Query construction integration

The WDC builder imports `build_table_join_records`, `source_splits`, `split_map`, model extraction/cache helpers, and shared writers. It supplies the same source-table, entity, and bridge-asset contracts expected by that code. The original query algorithm is not forked.

The only shared-code adjustment is to let `table_record()` take the provenance builder name from a source table, defaulting to the current EntiTables builder name. Existing output remains byte-compatible in that field.

## CLI and operational launch

The CLI keeps the existing query, model, sharding, and asset-limit parameter names. WDC-specific controls include generic HTTP timeouts, retries, response/image byte caps, web workers, host delay, row cap, and cache paths.

The initial unattended tmux run will use a separate output/cache directory and a conservative bounded configuration. It will be monitored through input discovery and the first successful/failing web entities. If startup is stable, the tmux process remains running.

## Testing

Unit tests use temporary gzip JSONL fixtures and fake sessions/downloads. They cover:

- WDC row conversion, stable columns, entity identity, nested values, and removal of `image`;
- recursive image URL extraction and relative URL resolution;
- HTML text/image extraction;
- the shared image quota with direct-image priority plus mandatory page fetch;
- exact reuse of text chunk limits;
- malformed input isolation and source-table limits;
- one small end-to-end build with a fake asset client and fake model extractor that verifies query/target/qrel output and absence of the `image` column.

## Out of scope

- Reconstructing top100/minimum3/rest membership after extraction into a shared class directory.
- Changing the existing model prompt, recovery threshold, row sampling, or context-column behavior.
- Guaranteeing that historical 2023 URLs remain live.
- Circumventing site authentication, bot challenges, or access controls.
