# MM Joinability Random Sampling and Replacement Design

## Goal

Change `scripts/build_mm_joinability_dataset.py` so source tables are sampled in a reproducibly random order instead of path/table order. When a selected table cannot produce a query table, discard it with 50% probability and replace it with another random candidate. Allow at most two replacements per target slot; retain the final failed table after those two replacements. Remove cache material owned only by discarded tables while preserving material shared with retained tables.

## User-Visible Behavior

- `--seed` controls both candidate ordering and discard decisions. The same input and arguments produce the same candidate and replacement choices.
- Input JSON files are shuffled, and table IDs within each file are shuffled, before structural parsing selects candidates.
- The builder aims to retain `max_source_tables` source tables. A slot starts with one table and may consume at most two replacement tables.
- A table that produces at least one query table is always retained.
- A table that produces no query table is discarded only when the seeded probability draw is below the configured probability.
- A failed table is retained when the probability draw says not to discard it.
- A failed table occupying a slot after two prior replacements is retained without another discard draw.
- If the candidate stream is exhausted, the builder retains the tables already assigned and records the unfilled slot count rather than looping indefinitely.

Add these command-line arguments:

- `--unrecoverable_replacement_rounds`, integer, default `2`, constrained to be non-negative.
- `--unrecoverable_drop_probability`, float, default `0.5`, constrained to `[0, 1]`.

The existing definition of unrecoverable remains authoritative: `build_table_join_records` returns no query tables for the source table. Model and API failures that leave a table without a query table follow the same replacement policy.

## Architecture

Split dataset construction into candidate evaluation and final materialization.

### Candidate stream

Use a dedicated `random.Random(args.seed)` instance rather than module-global random state. Shuffle the discovered JSON file list and independently shuffle each valid payload's table items. Parse tables lazily in that seeded order and yield only structurally valid source tables. The stream continues beyond the initial target count so replacement slots can draw additional candidates without downloading all potential replacement material in advance.

The candidate stream owns structural counters such as processed and malformed tables. It does not write final `source_tables` shards as it scans.

### Slot and round controller

Represent each target slot with its current source table and replacement count. Fill up to `max_source_tables` slots from the candidate stream, then evaluate only newly assigned tables in a batch. After each batch:

1. Retain queryable tables.
2. For failed tables below the replacement limit, draw from the seeded RNG.
3. Retain failed tables when the draw is greater than or equal to the drop probability.
4. Discard failed tables when the draw is below the probability, clean their exclusive material, increment the slot replacement count, and assign the next candidate.
5. Once a slot reaches two replacements, retain its current table even when it remains unrecoverable.

Only replacement slots enter the next evaluation round. With the defaults, a slot sees at most the original table plus two replacements.

### Candidate evaluation

Candidate evaluation builds or reuses the entities and bridge assets needed by the current batch, resolves model extraction tasks, and calls `build_table_join_records` using non-persistent record collectors. The returned query-table list determines recoverability. Candidate evaluation records dependency ownership but does not append candidate records to final dataset shards.

Evaluation results may be kept as compact decisions and dependency sets. Full query, target, extraction, and recovery records are regenerated during final materialization from the caches established during evaluation.

### Final materialization

After all slots settle, rebuild aggregate entity references for only the final source-table pool. Write all formal artifacts using the existing sharded writers:

- `source_tables`
- `entities`
- `bridge_assets`
- `table_asset_links`
- `query_tables`
- `data_lake_tables`
- `attribute_extractions`
- `evidence_recoveries`
- qrels, splits, decisions, stats, and manifest files

Final model extraction calls should hit the candidate-evaluation cache and must not repeat successful inference. A retained but unrecoverable table is emitted through the existing raw data-lake path.

## Dependency Tracking and Cache Cleanup

Track these dependencies per evaluated table:

- entity IDs and normalized Wikipedia titles;
- bridge asset IDs;
- downloaded image paths and their source URLs;
- Wikipedia page-cache keys;
- Wikipedia imageinfo-cache keys;
- model extraction cache keys.

Maintain active reference counts across current and retained slots. When a table is discarded, decrement its dependencies and physically clean only dependencies whose active count reaches zero.

At a round boundary, after all concurrent model tasks have completed:

- remove exclusive text/image assets from the in-memory asset maps;
- unlink exclusive downloaded images;
- retain a downloaded image when another active asset references the same resolved path or source URL;
- remove exclusive page and imageinfo records from the clients' in-memory cache maps;
- remove exclusive model records from `ExtractionCache.items`;
- compact `wiki_pages.jsonl`, `wiki_images.jsonl`, and `model_attribute_extractions.jsonl` by writing a sibling temporary file and atomically replacing the original.

Cleanup never runs concurrently with cache append operations. File deletion and compaction errors are logged and counted, then dataset construction continues. Final output-write errors retain their current fail-fast behavior.

After slot selection completes, perform an orphan sweep using the final dependency graph. This removes evaluated candidate material that has no final-table reference while preserving every entity, image, metadata record, and model cache record shared with the final pool.

## Dynamic vLLM Compatibility

The dynamic runner continues to start one text and one image service through the existing marker protocol, but the builder treats all candidate rounds as one model-analysis phase.

- Write the start marker after the first candidate batch has prepared material.
- In round mode, signal that model services are required whenever unresolved candidate slots remain, even if the first batch happens to expose zero immediate extraction tasks. This prevents the runner from exiting while the builder waits for the ready marker.
- Keep both services available while replacement rounds submit new tasks.
- Write text and image done markers only after all slots settle and final cache-backed materialization no longer needs uncached inference.
- Preserve the non-dynamic direct-builder path by initializing its extractor once and reusing it across rounds.

The runner may continue waiting on the builder process when neither modality is complete; it must not interpret the end of one candidate round as modality completion.

## Statistics and Provenance

Add enough information to reproduce and audit selection:

- sampling seed and sampling mode;
- configured replacement rounds and drop probability;
- total random candidates structurally accepted;
- initial slots filled;
- per-round evaluated, unrecoverable, discarded, retained-failed, and replacement counts;
- candidate exhaustion and unfilled slot counts;
- final recoverable and unrecoverable source-table counts;
- removed page records, imageinfo records, model records, image files, and image bytes;
- shared dependencies protected from cleanup.

Include the new policy values in stats, provenance, and the dataset manifest configuration block. Table queryability decisions remain present only for final tables.

## Testing

Use pytest with `tmp_path`, synthetic EntiTables payloads, fake Wikipedia clients, and fake extractors. No test may require network access, model servers, large generated output, or GPU hardware.

Add focused tests covering:

1. Identical seeds produce identical file/table ordering and discard choices.
2. Different seeds can produce different candidate orderings.
3. Candidate selection does not truncate by lexicographic file and table order.
4. Queryable tables are retained without a probability draw.
5. Failed tables are replaced only when the seeded draw falls below 0.5.
6. Each slot evaluates at most the original table and two replacements.
7. A failed table after the second replacement is retained.
8. Candidate exhaustion terminates and reports any unfilled slots.
9. Discarding a table removes its exclusive image and all three relevant JSONL cache record types.
10. Shared entities, local image paths or URLs, and model cache keys survive cleanup.
11. Final source/query/data-lake artifacts, qrels, splits, decisions, stats, and manifest reference only the settled final pool.
12. Dynamic vLLM done markers are not written between candidate rounds and are written after the complete analysis phase.
13. CLI defaults and validation enforce two rounds and probability 0.5.

Follow test-driven development: add one failing behavior test, verify its expected failure, implement the smallest supporting change, and rerun the focused test before proceeding.

Run the targeted tests and regression suite with:

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_extraction.py -q
```

Run any new focused test module separately during development, run the dynamic-runner tests contained in the extraction suite, and finish with `git diff --check` for the modified script and tests.

## Non-Goals

- Do not change the recovery threshold or attribute matching rules.
- Do not prefetch or infer over a fixed three-times-larger candidate pool.
- Do not delete material that remains referenced by any final or currently active table.
- Do not hand-edit generated datasets in `output_medium/` or `output_stage1_logic/`.
- Do not add network-dependent test fixtures.
