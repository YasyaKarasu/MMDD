# Image Attribute Dataset Table-Grouped Resplit Design

## Goal

Repartition the existing `output_mm_image_attribute_2k` dataset without
changing its selected samples, entities, source tables, or image files.

## Invariants

- Keep all 2,000 existing samples.
- Keep all 2,000 unique entities and image files.
- Keep all 1,366 existing source tables.
- Keep the global label totals: 1,369 positive and 631 negative samples.
- Never split a source table. Every sample with the same `source_table_id`
  must occur in exactly one split.
- Do not regenerate candidates or copy images.

## Target Distribution

| Split | Total | Positive | Negative |
| --- | ---: | ---: | ---: |
| train | 1,600 | 1,095 | 505 |
| val | 200 | 137 | 63 |
| test | 200 | 137 | 63 |

## Allocation

Load the existing sample files and group samples by `source_table_id`. Each
table group has a two-dimensional weight `(positive_count, negative_count)`.
Use deterministic dynamic programming to choose a complete-table subset for
validation with weight `(137, 63)`, then choose a disjoint complete-table
subset for test with the same weight. Assign every remaining table to train.

The current dataset contains enough positive-only and negative-only
single-sample tables for both targets, and a read-only feasibility check
confirmed that `(137, 63)` is reachable.

## Materialization

Rewrite these artifacts atomically:

- `samples/train.jsonl`, `samples/val.jsonl`, `samples/test.jsonl`
- `tables/train.jsonl`, `tables/val.jsonl`, `tables/test.jsonl`
- `stats.json`
- `README.md`

Leave `images/` and `manifest.json` unchanged except for adding split
statistics to the manifest if needed. Preserve each JSON record unchanged;
only its split file changes.

## Validation

After rewriting, verify:

- exact split and label counts;
- exactly 2,000 unique entities and 2,000 unique image paths;
- exactly 1,366 unique source tables;
- no `source_table_id` occurs in more than one split;
- every sample references a table in the same split;
- every `image_path` exists;
- the set of sample IDs, table IDs, and image paths is identical before and
  after repartitioning.

Automated tests cover exact two-dimensional table-group selection, failure
when a target is impossible, determinism, and table isolation.
