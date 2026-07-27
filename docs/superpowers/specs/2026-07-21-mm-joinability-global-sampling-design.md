# MM Joinability Global Table Sampling Design

## Goal

Replace shuffled-file block sampling with deterministic global table sampling across every EntiTables JSON file.

## Selection

The sampler scans all sorted JSON files and validates every table. Each valid table receives a stable priority computed from the configured seed, its relative file path, and table ID. The lowest priorities form the candidate stream.

For a capped run, candidate capacity is `max_source_tables * (unrecoverable_replacement_rounds + 1)`, which covers the initial pool and the worst-case number of replacement rounds. For an uncapped run, every valid table is retained. Stable path/table tie-breakers make ordering independent of filesystem enumeration.

## Memory and materialization

The first pass retains lightweight references rather than complete parsed tables. Selected references are sorted by priority and materialized in chunks no larger than `max_source_tables`: selected IDs are grouped by file, each required JSON is opened once per chunk, and full source-table records are yielded in priority order. This bounds live full-table memory while keeping the initial 40,000-table set globally uniform.

## Progress and compatibility

- The scan bar covers all discovered JSON files and is labeled `Scanning EntiTables for global sample`.
- Materialization has a separate progress bar.
- `--no_model_progress` disables both.
- Reuse `--seed`; do not consume replacement-policy RNG draws.
- Preserve parsing filters, counters, initial material preparation, replacement policy, cache cleanup, and model-start gating.

## Verification

Tests prove that a capped sample scans every input file, a late file can enter the initial sample, the same seed reproduces the same selection, different seeds can change it, candidate capacity includes two replacement rounds, and enumeration order cannot change results.
