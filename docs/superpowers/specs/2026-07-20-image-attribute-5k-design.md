# Image Attribute 5K Dataset Design

## Goal

Create a standalone, reproducible 5,000-sample image-attribute extraction
dataset from `output_mm_joinability_v5`. Each sample joins an entity, an
image, and an attribute label back to one row and one complete original table.
The output is independent of the source dataset and its shared cache.

## Inputs

The builder reads these sharded artifacts from the input dataset manifest:

- `source_tables`: complete relational source tables and their rows.
- `bridge_assets`: image asset metadata, including the original local file.
- `attribute_extractions`: VLM results for an entity-image pair and the
  candidate attributes from its source row.
- `table_asset_links`: validates that the image is attached to the referenced
  entity occurrence in the referenced table row and column.

Only image assets are eligible. Text assets, missing files, unreadable files,
and extraction records without an image asset are excluded.

## Labels

Each output sample represents one `(entity, image, table row, attribute)`
decision. It includes the complete ground-truth cell value from the source
table and an `extractable` boolean.

- A positive sample (`extractable: true`) requires a nonempty VLM attribute
  whose normalized name is a source-table column and whose normalized value
  matches the value in the referenced source row. VLM evidence and connection
  evidence are retained.
- A negative sample (`extractable: false`) uses a candidate source-table
  attribute for the same entity, image, table, and row that the VLM omitted.
  Negatives are therefore real non-extraction decisions, not random entity or
  attribute combinations. Their evidence fields are empty.

The final label composition is exactly 3,500 positives and 1,500 negatives.

## Diversity and Leakage Controls

The builder groups candidate samples by the number of columns in their source
table: `2-4`, `5-7`, `8-12`, and `13+`. It allocates each label class as evenly
as possible across these four buckets. Deterministic ranking, a fixed seed,
and per-table and per-entity caps prevent a small set of tables or entities
from dominating selection.

Tables are indivisible split groups. Samples from one `source_table_id` occur
in exactly one of train, validation, or test, including all images and labels
chosen from that table. The target counts are exactly 4,000 train, 500
validation, and 500 test samples, with a 70/30 positive/negative composition
and as-even-as-possible column-bucket distribution in every split.

If the exact global or split quotas cannot be fulfilled after eligibility,
diversity, and grouping constraints, the command fails with a detailed
eligibility and allocation report. It never silently emits fewer samples or
relaxes the requested label composition.

## Output Layout

The default directory is `output_mm_image_attribute_5k`:

```
output_mm_image_attribute_5k/
  images/<asset_id>.<suffix>
  tables/train.jsonl
  tables/val.jsonl
  tables/test.jsonl
  assets/train.jsonl
  assets/val.jsonl
  assets/test.jsonl
  samples/train.jsonl
  samples/val.jsonl
  samples/test.jsonl
  manifest.json
  stats.json
```

`tables/<split>.jsonl` contains de-duplicated, complete original
`source_tables` records. The script does not produce query tables and does not
truncate, project, or otherwise alter table rows or columns.

`samples/<split>.jsonl` stores stable identifiers, the split, entity metadata,
the internal relative image path, source table/row/entity-column references,
the target attribute name and ground-truth value, label, and retained VLM
evidence for positives. `assets/<split>.jsonl` contains only the referenced
image-asset records, with their `local_path` rewritten to the dataset-relative
image path.

The builder copies every selected image into `images/`, validates that it is a
readable raster file, uses a collision-safe stable asset filename, and records
a SHA-256 digest and byte count. After completion, no output artifact refers
to `output_mm_joinability_v5`, its cache, or an absolute input path except as
provenance in `manifest.json`.

## CLI and Validation

One Python CLI accepts input directory, output directory, seed, total count,
split counts, label counts, per-table/entity caps, and overwrite behavior.
Defaults implement this specification.

It emits `stats.json` with candidate exclusion counts, selected counts by
split/label/column bucket, unique entities/tables/assets, image-copy results,
and integrity-check results. `manifest.json` captures source manifest hashes,
arguments, schema version, and artifact file lists.

Tests use synthetic sharded input fixtures to prove positive matching,
negative construction, original-table preservation, image copying and path
rewriting, deterministic balanced selection, table-level split isolation, and
quota failure reporting.
