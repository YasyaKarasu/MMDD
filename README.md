# EntiTables Multimodal Table Dataset Builder

This repository contains a builder for creating a multimodal table data lake and a query workload from EntiTables-style JSON files.

The goal is dataset construction only: source tables, projected query views, entity mentions, optional Wikipedia text/image metadata, table-to-asset links, and source-level splits.

It does not construct joinability benchmark labels. It does not generate positive pairs, negative pairs, joinable/not-joinable fields, augmentation targets, or query-target pairs.

## Input Format

Each EntiTables JSON file is expected to be a top-level dictionary:

```json
{
  "table-0001-590": {
    "title": ["Column A", "Column B"],
    "numCols": 2,
    "numericColumns": [],
    "pgTitle": "Wikipedia page title",
    "numDataRows": 4,
    "secondTitle": "Section title",
    "numHeaderRows": 1,
    "caption": "Table caption",
    "data": [
      ["[Page_Title|Display Text]", "plain text"],
      ["[Another_Page]", "123"]
    ]
  }
}
```

Cells may contain Wikipedia links in either `[Page_Title|Display Text]` or `[Page_Title]` form. Plain text cells are also supported.

## Outputs

The script writes these artifacts to `--output_dir`:

- `source_tables/part-*.jsonl`: cleaned and standardized source tables, including normalized columns, parsed cells, column profiles, and candidate entity columns.
- `query_views/part-*.jsonl`: query workload tables produced by projecting one source table into one or more smaller views.
- `entities/part-*.jsonl`: unique Wikipedia entities found in source tables and query views, with cell-level appearances.
- `bridge_assets/part-*.jsonl`: optional Wikipedia text extracts and downloaded image assets with source URLs and metadata.
- `table_asset_links/part-*.jsonl`: links from entity cells in source tables/query views to entity assets. Empty `asset_ids` are retained when assets are unavailable.
- `splits.json`: train/dev/test split at source-table or page-title level.
- `stats.json`: processing counts, skipped reasons, asset counts, and workload statistics.
- `dataset_manifest.json`: the shard manifest listing every artifact directory, part file, and record count.

Large JSONL outputs are streamed and sharded. The builder writes and flushes part files incrementally instead of keeping the full dataset in memory until the end.

## Query Views

`query_views` are part of the dataset and represent query workload tables projected from source tables. A single source table may produce multiple query views.

Supported view strategies:

- `entity_plus_one_attr`
- `entity_plus_two_attrs`
- `entity_only`
- `random_projection`

These views are not positive samples, negative samples, labels, or training pairs. They are query tables for later system experiments.

## Hidden Columns

`hidden_column_indices` and `hidden_column_names` only record which original source-table columns were omitted when constructing a projected query view.

They are provenance fields only. They are not augmentation ground truth, not target columns, not positive labels, and not joinability labels.

## Splits

Splits are source-level splits. With `--split_by page_title`, all source tables from the same Wikipedia page are kept in the same split when possible. With `--split_by source_table_id`, all query views derived from a source table stay with that source table.

This avoids leakage where different projected views from the same source table appear in different train/dev/test partitions.

## Setup

Use the existing conda environment:

```bash
conda run -n MMDD python -m pip install -r requirements.txt
```

`hnswlib` is the default ANN backend for stage-1 retrieval. `pytest` is used for tests. `transformers` and `qwen-vl-utils` are used by the local Qwen3-VL-Embedding-2B encoder wrapper.

## Usage

Build the table dataset and query workload with Wikipedia metadata and downloaded images:

```bash
conda run -n MMDD python scripts/build_mm_table_dataset.py \
  --input_dir dataset/tables_redi2_1 \
  --output_dir output \
  --max_tables 100 \
  --max_query_views_per_source_table 5 \
  --max_entities 1000 \
  --max_images_per_entity 3 \
  --wiki_link_threshold 0.3 \
  --flush_every_records 500 \
  --records_per_shard 50000 \
  --split_by page_title
```

Build only tables, query views, entities, and entity links without calling Wikipedia:

```bash
conda run -n MMDD python scripts/build_mm_table_dataset.py \
  --input_dir dataset/tables_redi2_1 \
  --output_dir output_no_wikipedia \
  --max_tables 100 \
  --max_query_views_per_source_table 5 \
  --wiki_link_threshold 0.3 \
  --flush_every_records 500 \
  --records_per_shard 50000 \
  --split_by page_title \
  --no_wikipedia
```

When `--no_wikipedia` is set, `bridge_assets.jsonl` is empty, but `entities.jsonl` and `table_asset_links.jsonl` are still produced. Entity links remain useful because `asset_ids` are simply empty lists.

## Image Downloads

When Wikipedia fetching is enabled, image assets are downloaded to:

```text
<output_dir>/images/
```

Each image record in `bridge_assets/part-*.jsonl` keeps both provenance and local file information:

- `image_url`: original Wikipedia/Commons image URL.
- `description_url`: image description page URL.
- `local_path`: absolute local path to the downloaded image.
- `relative_path`: path relative to `output_dir`.
- `file_name`: downloaded image file name.
- `bytes`: downloaded file size.
- `sha256`: file digest.
- `metadata`: MediaWiki `imageinfo` metadata, including `extmetadata`.

The script still filters obvious low-value files such as icons, logos, flags, and placeholders before downloading.

## Streaming Writes

Use `--flush_every_records` to control how often JSONL writers flush buffered records to disk. The default is `500`.

Use `--records_per_shard` to control how many JSONL records go into each part file before the writer starts the next shard. The default is `50000`.

This applies to the large streaming outputs:

- `source_tables/part-*.jsonl`
- `query_views/part-*.jsonl`
- `entities/part-*.jsonl`
- `bridge_assets/part-*.jsonl`
- `table_asset_links/part-*.jsonl`

Read `dataset_manifest.json` to discover the current shard set. This is safer than globbing when reusing an output directory, because stale part files from an older run may still exist.

`splits.json` and `stats.json` are written as single JSON files after the needed indexes have been finalized.

## Important Non-Goals

This builder intentionally does not generate:

- `original_joinable`
- `bridged_joinable`
- `negative_type`
- `positive_pairs`
- `negative_pairs`
- `labels`
- query-target pairs
- joinability labels
- augmentation ground truth

The output is a multimodal table dataset plus query workload, not a joinability benchmark label generator.

## WDC Schema.org 200K Joinability Pipeline

`build_wdc200k_mm_joinability_dataset.py` is the staged, resumable entry point
for the 200K-scale WDC Schema.org Table Corpus 2023 build.
`build_wdc_mm_joinability_dataset.py` remains the legacy bounded builder; its
row-limited examples and in-memory behavior are not the 200K operating model.

The staged pipeline keeps three roots separate:

```text
output_wdc_200k/   canonical dataset artifacts only
work_wdc_200k/     selections, shards, durable job stores, checkpoints,
                   stage manifests, runtime markers, and progress.json
cache/wdc_200k/    reusable positive and negative network/media/model outcomes
```

Never nest or reuse these roots as one another. Only `output_dir` has the
canonical source/query/data-lake tables, entities, assets, links, extractions,
qrels, splits, errors, stats, and `dataset_manifest.json`. Source-table rows
are unbounded and every selected row is preserved. Do not pass
`--max_rows_per_source_table`: the compatibility option deliberately rejects
every value, including zero. The WDC `image` column is used only to discover
assets and is removed from table artifacts.

### Preflight, staged execution, and recovery

A read-only configuration preflight validates the input statistics archives,
path separation, runtime paths, disk reserve, and arguments without running a
stage:

```bash
conda run -n MMDD python scripts/build_wdc200k_mm_joinability_dataset.py \
  --input_dir wdc_schemaorg_2023 \
  --output_dir output_wdc_200k \
  --work_dir work_wdc_200k \
  --cache_dir cache/wdc_200k \
  --max_source_tables 200000 \
  --selection_seed 13 \
  --dry_run
```

The structural preflight writes deterministic selection and structural state
under `work_dir`, but performs no network or model work:

```bash
conda run --no-capture-output -n MMDD python \
  scripts/build_wdc200k_mm_joinability_dataset.py \
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

Inspect `work_wdc_200k/progress.json` and the structural manifests before
continuing. Resume with the same four roots and parameters; completed,
checksummed shards and terminal URL outcomes are not replayed:

```bash
conda run --no-capture-output -n MMDD python \
  scripts/build_wdc200k_mm_joinability_dataset.py \
  --input_dir wdc_schemaorg_2023 \
  --output_dir output_wdc_200k \
  --work_dir work_wdc_200k \
  --cache_dir cache/wdc_200k \
  --max_source_tables 200000 \
  --selection_seed 13 \
  --resume \
  --text_model_base_url http://127.0.0.1:8001/v1 \
  --image_model_base_url http://127.0.0.1:8000/v1
```

Use `--stop_after` to establish a stage barrier. To intentionally rebuild a
named stage and everything downstream, use `--from_stage`, which validates
upstream state and archives replaced state rather than deleting it:

```bash
conda run --no-capture-output -n MMDD python \
  scripts/build_wdc200k_mm_joinability_dataset.py \
  --input_dir wdc_schemaorg_2023 \
  --output_dir output_wdc_200k \
  --work_dir work_wdc_200k \
  --cache_dir cache/wdc_200k \
  --max_source_tables 200000 \
  --from_stage pages \
  --stop_after pages
```

Add `--refresh_page_cache` only with `--from_stage pages` or earlier, and
`--refresh_image_cache` only with `--from_stage images` or earlier. A plain
resume reuses both successes and terminal failures.

### Real-table 100/1,000 scale gates

The full corpus contains more mandatory `top100` candidates than a 100- or
1,000-table target, so those gates must not run by merely lowering
`--max_source_tables` on the full input. Create isolated, exact-size
real-table gate inputs instead:

```bash
conda run -n MMDD python scripts/create_wdc200k_scale_gate_input.py \
  --source_dir wdc_schemaorg_2023 \
  --target_dir gate_inputs/wdc_100 \
  --table_count 100 \
  --seed 13

conda run -n MMDD python scripts/create_wdc200k_scale_gate_input.py \
  --source_dir wdc_schemaorg_2023 \
  --target_dir gate_inputs/wdc_1000 \
  --table_count 1000 \
  --seed 13
```

The helper round-robins class/subset buckets and chooses low-row tables by
`(rows, stable_hash(seed, relative_path), relative_path)`. It creates filtered
production-format statistics ZIPs plus absolute source-data symlinks; it does
not copy, delete, fetch, or modify corpus data. The target must be absent or
strictly empty. `scale_gate_manifest.jsonl` records absolute sources, relative
targets, and source gzip hashes; `scale_gate_checksums.json` covers the
manifest and filtered ZIPs. These are operational scale-gate subcorpora, not
the formal 200K sampling result. Use each gate input with the matching
`--max_source_tables` and fresh output/work/cache roots. The 10K structural
gate uses the full corpus directly.

### Direct endpoints, dynamic vLLM, and tmux

The resume command above is the direct runner: the operator supplies already
running OpenAI-compatible text and image endpoints. The dynamic runner instead
starts and reallocates both vLLM modalities, and its unowned options pass
through to the staged WDC 200K builder:

```bash
conda run --no-capture-output -n MMDD python \
  scripts/run_mm_joinability_dynamic_vllm.py \
  --input_dir wdc_schemaorg_2023 \
  --output_dir output_wdc_200k \
  --text_model_path /path/to/text-model \
  --image_model_path /path/to/vision-model \
  --work_dir work_wdc_200k \
  --cache_dir cache/wdc_200k \
  --max_source_tables 200000 \
  --selection_seed 13 \
  --resume
```

The dynamic runner owns endpoint files, model identity, and staged
start/ready/done markers under `work_wdc_200k/runtime`.

For a structural tmux preflight, keep process output attached directly to the
tmux pane and use a second pane for the atomic progress snapshot:

```bash
tmux new-session -d -s wdc_200k \
  'conda run --no-capture-output -n MMDD python scripts/build_wdc200k_mm_joinability_dataset.py --input_dir wdc_schemaorg_2023 --output_dir output_wdc_200k --work_dir work_wdc_200k --cache_dir cache/wdc_200k --max_source_tables 200000 --selection_seed 13 --stop_after structural'
tmux new-window -t wdc_200k -n progress \
  "watch -n 5 'conda run -n MMDD python -m json.tool work_wdc_200k/progress.json'"
tmux attach-session -t wdc_200k
```

Do not redirect stdout/stderr to a log file: the runner emits bounded periodic
progress directly, while durable progress lives in
`work_wdc_200k/progress.json`. Scale-gate measurements and acceptance criteria
are recorded in
`docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md`.

## Stage-1 Logic Connectivity Pipeline

Stage-1 builds a coarse recall benchmark and training pipeline from the existing `output_medium` artifacts only. It does not refetch Wikipedia, does not rebuild `output_medium`, and does not use `query_views` as supervision.

Recommended consolidated entrypoints:

```bash
# 1) Build Stage-1 logic fragments, evidence paths, weak labels, and frozen embeddings.
conda run -n MMDD python scripts/stage1_prepare_data.py \
  --input_dir output_medium \
  --stage1_dir output_stage1_logic \
  --encoder_path ./Qwen3-VL-Embedding-2B \
  --embedding_batch_size 8 \
  --device cuda \
  --dtype bf16

# 2) Train teacher, run HITL annotation rounds, retrain teacher, and distill the student.
conda run -n MMDD python scripts/stage1_train_models.py \
  --stage1_dir output_stage1_logic \
  --hitl_rounds 3 \
  --hitl_batch_size 50 \
  --distill_loss pairwise \
  --force_retrain \
  --lan \
  --port 7860

# 3) Build ANN indexes and evaluate relation-aware multi-hop recall.
conda run -n MMDD python scripts/stage1_index_eval.py \
  --stage1_dir output_stage1_logic \
  --topk 10 50 100 \
  --max_hops 3 \
  --beam_width 64 \
  --beam_neighbors 50
```

`stage1_run_all.py` is available when you want a single command for the whole flow. It will pause during HITL rounds until labels are merged in the browser UI:

```bash
conda run -n MMDD python scripts/stage1_run_all.py \
  --input_dir output_medium \
  --stage1_dir output_stage1_logic \
  --hitl_rounds 3 \
  --hitl_batch_size 50 \
  --distill_loss pairwise \
  --force_retrain \
  --lan \
  --port 7860
```

To inspect constructed connection groups in a browser, run:

```bash
conda run -n MMDD python scripts/stage1_connection_viewer.py --stage1_dir output_stage1_logic --port 7861
```

The viewer paginates by `chain_id` and shows the visible query, hidden query, target fragment, logic pairs, qrels, and evidence paths for each group.

To inspect `build_mm_joinability_dataset.py` outputs in a browser, run:

```bash
conda run -n MMDD python scripts/mm_joinability_dataset_viewer.py --output_dir output_mm_joinability --port 7863
```

The viewer paginates through every `query_table_id -> target_table_id` pair in `qrels.jsonl` and shows the query table, target table, evidence recovery paths, and the referenced text/image material for each path.

By default, Flask GUIs bind to `127.0.0.1` and are available only on the same machine. Add `--lan` to `stage1_train_models.py`, `stage1_run_all.py`, `run_hitl_training_rounds.py`, `hitl_annotation_app.py`, `stage1_connection_viewer.py`, or `mm_joinability_dataset_viewer.py` to bind to `0.0.0.0`; the startup log prints both the local URL and the LAN URL for other devices. If the LAN URL is still unreachable, allow the selected port through the host firewall and make sure the devices are on the same network.

Use `--force_retrain` when you want to ignore partially generated training outputs and start the training/HITL loop cleanly. It removes teacher/student models, teacher scores, train pairs, open HITL selections, templates, and annotation status files while preserving prepared data, embeddings, and merged human labels. Add `--reset_human_labels` only when you also want to discard previously merged human labels.

Lower-level commands are still available for debugging or running individual steps:

```bash
conda run -n MMDD python scripts/build_stage1_logic_connectivity.py --input_dir output_medium --output_dir output_stage1_logic --seed 13
conda run -n MMDD python scripts/build_stage1_evidence_paths.py --input_dir output_medium --stage1_dir output_stage1_logic --seed 13
conda run -n MMDD python scripts/generate_weak_labels.py --stage1_dir output_stage1_logic
conda run -n MMDD python scripts/select_hitl_batch.py --stage1_dir output_stage1_logic --round_id 0 --batch_size 50
conda run -n MMDD python scripts/build_stage1_embeddings.py --input_dir output_medium --stage1_dir output_stage1_logic --encoder_path ./Qwen3-VL-Embedding-2B --batch_size 8 --device cuda --dtype bf16
conda run -n MMDD python scripts/build_teacher_training_data.py --stage1_dir output_stage1_logic
conda run -n MMDD python scripts/train_teacher.py --stage1_dir output_stage1_logic --embedding_dir output_stage1_logic/embeddings
conda run -n MMDD python scripts/train_student.py --stage1_dir output_stage1_logic --embedding_dir output_stage1_logic/embeddings --teacher_scores output_stage1_logic/teacher_scores.jsonl --distill_loss pairwise
conda run -n MMDD python scripts/build_hnsw_indices.py --stage1_dir output_stage1_logic --student_dir output_stage1_logic/student --space cosine --m 32 --ef_construction 200 --ef_search 100
conda run -n MMDD python scripts/eval_stage1_recall.py --stage1_dir output_stage1_logic --student_dir output_stage1_logic/student --hnsw_dir output_stage1_logic/hnsw_indices --topk 10 50 100 --max_hops 3 --beam_width 64 --beam_neighbors 50
```

`train_student.py` supports `--distill_loss pairwise` and `--distill_loss listwise`. ANN indexes store projected object vectors `u_b`; online retrieval computes the relation-aware query vector `u_a R_{\tau(a),\tau(b)}` before querying each target-type index. Path-aware evaluation uses beam-search multi-hop expansion and only returns table endpoints; assets are retained only as intermediate path evidence.

After filling `output_stage1_logic/human_labels_template_round_0.jsonl`, merge labels with:

```bash
conda run -n MMDD python scripts/merge_human_labels.py --stage1_dir output_stage1_logic --human_labels output_stage1_logic/human_labels_template_round_0.filled.jsonl
```

To run active-learning rounds where the teacher is retrained, uncertain paths are selected, and a browser UI collects human labels:

```bash
conda run -n MMDD python scripts/run_hitl_training_rounds.py \
  --stage1_dir output_stage1_logic \
  --embedding_dir output_stage1_logic/embeddings \
  --rounds 3 \
  --batch_size 50 \
  --force_retrain \
  --lan \
  --port 7860
```

Each round trains the teacher on the latest merged human labels, scores all candidate paths, writes `human_labels_template_round_<N>.jsonl`, starts the Flask annotation UI, and waits until the UI merges the completed labels into `human_labeled_paths.jsonl`. The selector excludes already merged labels and previously selected open batches by default.

To annotate an existing round without running training:

```bash
conda run -n MMDD python scripts/hitl_annotation_app.py --stage1_dir output_stage1_logic --round_id 0 --lan --port 7860
```

The default encoder is the local Qwen3-VL-Embedding-2B model. If `./Qwen3-VL-Embedding-2B` is not present, the wrapper also checks `./hf_models/Qwen3-VL-Embedding-2B`; pass `--encoder_path` for any other local path.
