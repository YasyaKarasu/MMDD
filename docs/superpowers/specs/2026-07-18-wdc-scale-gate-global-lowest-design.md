# WDC Scale-Gate `global_lowest` Sampling Addendum

## Problem

The scale-gate helper currently round-robins class/subset buckets before
choosing low-row tables. That remains useful for broad category coverage, but
it is not a fast systems gate: the real 100-table sample expanded to 55,839
rows and 52,880 page URLs. Repeating that shape at 1,000 tables would make
basic end-to-end and resume validation unnecessarily slow.

This addendum adds an explicitly requested low-cost sampling mode to
`scripts/create_wdc200k_scale_gate_input.py`. It changes neither the formal
200K selection policy nor the default helper behavior.

## Interface and Compatibility

The helper accepts:

```text
--selection_mode {round_robin,global_lowest}
```

`round_robin` is the default for both the Python API and CLI. Existing calls
therefore retain their exact selection and ordering behavior.

`global_lowest` selects the globally smallest exact `N` existing candidates by:

```text
(rows, stable_hash(seed, relative_path), relative_path)
```

The complete key makes selection deterministic even when row counts or hashes
tie. Selected records are emitted in ascending key order. The checksums
metadata and CLI result record the selected mode so a gate input is
self-describing.

## Selection Data Flow

Both modes keep the current two-pass catalog workflow.

1. The first pass strictly validates every catalog path, inserts every relative
   path into the staging SQLite primary-key index, rejects duplicates, and
   counts existing source gzip files.
2. The requested mode creates its bounded candidate pool. `round_robin` uses
   the existing per-bucket quota pool unchanged. `global_lowest` uses one
   max-oriented heap capped at `N`.
3. The second pass strictly validates every catalog path again. Each existing
   source table is ranked and offered to the selected pool.
4. The pool must return exactly `N` records or the operation fails before
   publication.

The global pool retains at most `N` ranked candidates, so memory is bounded by
the gate size rather than the 4.9-million-table corpus. The SQLite index
remains disk-backed and is never published.

## Publication and Failure Semantics

Selection mode does not fork the materialization or publication path. Both
modes reuse the existing:

- canonical relative-path validation and source-containment checks;
- SQLite duplicate rejection;
- filtered production-format statistics ZIP creation;
- absolute source-data symlinks and source SHA-256 records;
- generated-file checksums and staging validation;
- sibling staging directory, fsync sequence, and atomic no-clobber publish.

Malformed paths, duplicate paths, insufficient existing tables, staging
failures, validation failures, and publish races all fail closed. No partial
target is exposed, and source corpus files are never copied, modified, or
deleted.

## Operational Use

Only the documented 100- and 1,000-table quick system-gate commands in
`README.md` and
`docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md` opt into
`--selection_mode global_lowest`.

These samples intentionally minimize source rows. They are suitable for
pipeline plumbing, interruption/resume, request deduplication, checksum,
memory, disk-reserve, and canonical-output validation. They are explicitly
non-representative of WDC category balance, table quality, row-count
distribution, and the formal stratified 200K dataset. The formal 200K launch
and its production selector remain unchanged.

## Test Contract

Automated coverage must prove:

- omitting `selection_mode` preserves the current round-robin result;
- `global_lowest` returns exactly `N`, is deterministic, and equals a full
  reference sort by the global key;
- the global candidate pool retains at most `N` records, which is stronger
  than the accepted `N + bucket_count` bound;
- malformed-path, duplicate-path, filtered-ZIP, symlink, checksum, staging,
  and no-clobber publication guarantees apply to both modes;
- CLI choices/defaults and the two quick-gate documentation sections name the
  intended mode and its non-representative scope.
