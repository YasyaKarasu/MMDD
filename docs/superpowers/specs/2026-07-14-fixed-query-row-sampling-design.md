# Fixed Query Row Sampling Design

## Goal

Generate query/target pairs with a configurable, fixed number of rows while
using valid entity rows, rather than non-empty join-attribute rows, as the
recovery denominator. Source tables that cannot supply enough valid entity
rows remain available as raw data-lake tables.

## Definitions

- `Q`: requested rows per generated query/target table. A new
  `--query_rows_per_table` argument controls this value and defaults to `5`.
- Valid entity row: a source row whose selected entity cell has a non-empty
  Wikipedia title that resolves to an entity ID. The candidate join-attribute
  cell may be empty.
- `V`: number of valid entity rows for a source table and candidate join
  column.
- Recovered row: a valid entity row for which model evidence successfully
  recovers and matches the candidate join-attribute value.
- `R`: number of recovered rows for the candidate join column.
- `min_ratio`: `--min_recovered_value_ratio`.

## Qualification Rules

Candidate columns use valid entity rows as the denominator. Empty candidate
join-attribute cells remain in `V` but cannot enter `R`, so they count as
recovery failures.

The required recovered-row count is:

```text
V <= Q: ceil(V * min_ratio)
V > Q:  ceil(Q * min_ratio)
```

A candidate column qualifies when:

```text
V >= min_recovery_denominator
R >= required recovered-row count
```

The existing `--min_recovery_denominator` argument remains an additional
valid-entity-row floor for compatibility. A source table must also have
`V >= Q` to emit a query/target pair. If `V < Q`, it remains in the data lake
as a `raw_data_lake_table` and emits no query, target, or qrel.

`recovered_value_ratio` remains `R / V`, including empty join-attribute cells
in the denominator. Qualification for `V > Q` is intentionally based on the
capped `Q` denominator rather than the full-table ratio.

## Row Selection

For each selected qualified column, choose exactly `Q` valid entity rows in a
deterministic source-row order:

1. Select the first `ceil(Q * min_ratio)` recovered rows.
2. Fill the remaining slots with valid but unrecovered rows. This group
   includes rows whose candidate join-attribute cell is empty.
3. If too few unrecovered rows exist, fill the remaining slots with additional
   recovered rows.

The query and target projections use the same selected source row IDs. The
query projection continues to require a non-empty entity cell. The target
projection must not require the join-attribute cell to be non-empty, because
an empty value represents an intentionally retained recovery failure.

This selection guarantees that every emitted query/target pair has exactly
`Q` aligned rows and that its selected rows meet or exceed `min_ratio` whenever
the source has enough valid entity rows.

## CLI And Compatibility

Add:

```text
--query_rows_per_table 5
```

The value must be a positive integer. Existing direct callers that construct
an `argparse.Namespace` without the new field use the default value `5` via a
compatibility lookup.

Keep `--min_rows_per_output_table`. It remains a lower-bound compatibility
check. Configuration is invalid when `min_rows_per_output_table` is greater
than `query_rows_per_table`, because an exact-size output could never satisfy
that lower bound.

The dynamic vLLM wrapper requires no dedicated option because it already
passes unknown builder arguments through to `build_mm_joinability_dataset.py`.

## Output Metadata

Qualified-column and hidden-attribute metadata will expose:

- `valid_entity_rows`: `V`.
- `recovered_rows`: `R`.
- `required_recovered_rows`: the threshold used for qualification.
- `recovered_value_ratio`: `R / V`.
- `selected_rows`: `Q` for emitted query/target pairs.

Keep `eligible_rows` as a compatibility alias for `valid_entity_rows`, with
its updated meaning documented by the new explicit field.

Dataset statistics and the manifest configuration will record
`query_rows_per_table` so a generated dataset's row-selection contract is
auditable.

## Failure And Fallback Behavior

- Fewer than `Q` valid entity rows: emit the source as a raw data-lake table.
- No candidate column reaches its required recovery count: emit the source as
  a raw data-lake table.
- A qualified column cannot produce aligned `Q` query and target rows: emit
  the source as a raw data-lake table and record the existing query/target
  split failure reason.
- Invalid CLI values or contradictory row bounds fail before expensive
  Wikipedia or model work begins.

## Testing

Add focused tests for:

1. Threshold calculation below, at, and above `Q`, including non-integer
   products that require `ceil`.
2. Empty candidate attributes counting in `V` but not `R`.
3. Deterministic selection of the required recovered rows followed by
   unrecovered rows, with recovered-row fallback when needed.
4. Source tables with fewer than `Q` valid entity rows remaining raw data-lake
   tables without query/target/qrel records.
5. Query and target projections containing the same `Q` source row IDs even
   when selected target join values are empty.
6. CLI default, explicit override, positivity validation, and contradictory
   `min_rows_per_output_table` validation.
7. Existing joinability extraction, dynamic-wrapper, and Stage-1 pipeline
   regression suites.

## Scope

This change does not alter source-table ingestion minimums, entity-column
selection, model prompting, extraction caching, best-qualified-column
selection, or the number of query tables emitted per source table.
