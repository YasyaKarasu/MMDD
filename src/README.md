# Dataset construction

`src/` contains the research dataset builders extracted from `scripts_old/`.
It has no imports from `scripts_old/`, and it deliberately excludes training,
evaluation, annotation UIs, auto-checkers, GPU scheduling, checkpoint recovery,
and large-scale service orchestration.

There are three entry points:

- `build_table_dataset.py` builds normalized source tables, projected query
  views, entities, and optional text/image assets;
- `build_dataset.py` builds the multimodal joinability benchmark;
- `build_image_attribute_dataset.py` derives the image-attribute subset.

The joinability algorithm is:

1. adapt EntiTables JSON or WDC gzip JSONL into one source-table schema;
2. associate entity cells with text/image evidence;
3. extract each candidate attribute with that attribute masked from row context;
4. keep columns whose recovered-value ratio passes the threshold;
5. hide the recovered column in the query and expose it in the target table.

## Table workload

Build normalized source tables and projected query views without network calls:

```bash
conda run -n MMDD python src/build_table_dataset.py \
  --source entitables \
  --input-dir dataset/tables_redi2_1 \
  --output-dir output_research \
  --max-tables 100
```

## Joinability benchmark

Build only normalized source tables and entities (no network or model calls):

```bash
conda run -n MMDD python src/build_dataset.py \
  --source entitables \
  --input-dir dataset/tables_redi2_1 \
  --output-dir output_joinability \
  --max-tables 100
```

Run the complete pipeline with Wikipedia/page evidence and an
OpenAI-compatible model endpoint:

```bash
conda run -n MMDD python src/build_dataset.py \
  --source entitables \
  --input-dir dataset/tables_redi2_1 \
  --output-dir output_joinability \
  --fetch-assets \
  --text-model-base-url http://127.0.0.1:8001/v1 \
  --text-model-name Qwen3.5-9B \
  --image-model-base-url http://127.0.0.1:8000/v1 \
  --image-model-name Qwen3-VL-8B-Instruct
```

For reproducible offline construction, replace network/model calls with flat
JSONL inputs:

```bash
conda run -n MMDD python src/build_dataset.py \
  --source entitables \
  --input-dir dataset/tables_redi2_1 \
  --output-dir output_joinability \
  --assets-jsonl prepared_assets.jsonl \
  --extractions-jsonl prepared_extractions.jsonl
```

`prepared_extractions.jsonl` has one record per
`(source_table_id, source_row_id, asset_id, attribute_name)` and uses the fields
`entity_id`, `asset_type`, `value`, and `evidence`. The same format is emitted as
`attribute_extractions.jsonl` by an online run.

Derive the optional image-attribute subset:

```bash
conda run -n MMDD python src/build_image_attribute_dataset.py \
  --input-dir output_joinability \
  --output-dir output_image_attributes
```

Every large artifact is a flat UTF-8 JSONL file. This is intentional: the new
code favors an inspectable research implementation over stale-shard handling,
resume registries, or distributed writer infrastructure.
