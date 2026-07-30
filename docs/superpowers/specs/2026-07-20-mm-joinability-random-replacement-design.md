# MM Joinability Random Sampling and Replacement Design

## Goal

Change `scripts/build_mm_joinability_dataset.py` so source tables are sampled in a reproducibly random order instead of path/table order. On every replacement pass, consider all source-table slots that currently cannot produce a query table, discard each with 50% probability, and replace selected slots with another random candidate. Allow at most two global replacement passes. Prune discarded candidates from the active output material while retaining their persistent download and analysis caches for future runs.

## User-Visible Behavior

- `--seed` controls both candidate ordering and discard decisions. The same input and arguments produce the same candidate and replacement choices.
- Input JSON files are shuffled, and table IDs within each file are shuffled, before structural parsing selects candidates.
- The builder aims to retain `max_source_tables` source tables. A slot starts with one table and may consume at most one replacement table per global replacement pass.
- A table that produces at least one query table is always retained.
- A table that produces no query table is discarded only when the seeded probability draw is below the configured probability.
- A failed table not selected by one probability draw remains eligible in every later configured replacement pass.
- After the final replacement pass, any failed table still occupying a slot is retained.
- If the candidate stream is exhausted, the builder retains the tables already assigned and records the unfilled slot count rather than looping indefinitely.
- Candidate replacement never deletes downloaded images, Wikipedia response records, or model-extraction cache records; reruns can reuse all completed work.

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

Represent each target slot with its current source table and latest evaluation. Fill up to `max_source_tables` slots from the candidate stream, then evaluate newly assigned tables in a batch. On each global replacement pass:

1. Store the evaluations for newly assigned candidates.
2. Build the eligible pool from every slot whose current table is failed, including tables not selected on earlier passes.
3. Draw once for every eligible failed slot in stable slot order.
4. Keep an unselected failure in place so it participates in the next pass.
5. Replace selected failures when another candidate is available.

Only newly assigned replacements require another expensive evaluation, but the next replacement decision again uses the complete current failed pool. With the defaults, a slot still sees at most the original table plus two replacements.

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

## Dependency Tracking and Cache Retention

Track these dependencies per evaluated table:

- entity IDs and normalized Wikipedia titles;
- bridge asset IDs;
- downloaded image paths and their source URLs;
- Wikipedia page-cache keys;
- Wikipedia imageinfo-cache keys;
- model extraction cache keys.

Maintain active dependency ownership across current and retained slots. When a table is discarded, remove dependencies that no active slot references from the in-memory entity-to-asset and asset maps so final artifacts contain only the settled pool.

At a round boundary, after all concurrent model tasks have completed:

- remove exclusive text/image assets from the in-memory asset maps;
- retain all downloaded images, including images used only by replaced candidates;
- retain all page and imageinfo records in the clients' in-memory cache maps and JSONL files;
- retain all model records in `ExtractionCache.items` and `model_attribute_extractions.jsonl`;
- do not compact persistent caches as part of candidate selection.

Active-material pruning runs only after a replacement batch has registered its dependencies. Final output-write errors retain their current fail-fast behavior.

After slot selection completes, perform an orphan sweep using the final dependency graph. This removes evaluated candidate assets that have no final-table reference from the generated dataset, without removing reusable persistent cache records or files.

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
- per-round newly evaluated, all-current unrecoverable, discarded, retained-failed, and replacement counts;
- candidate exhaustion and unfilled slot counts;
- final recoverable and unrecoverable source-table counts;
- explicit `all_current_failed_slots` replacement scope;
- explicit persistent-cache retention policy;
- active entities and assets pruned from final material.

Include the new policy values in stats, provenance, and the dataset manifest configuration block. Table queryability decisions remain present only for final tables.

In per-round statistics, `evaluated` counts only newly assigned candidates that required analysis during that pass. `unrecoverable` counts the complete current failed pool considered for replacement. Before the terminal pass, `retained_failed` means deferred to a later pass rather than permanently settled.

## Entity Filtering and Multi-Query Construction

Filter structurally valid tables without any `candidate_entity_columns` before
they enter the seeded global sample. Candidate recognition continues to use the
configured wiki-link threshold plus the existing non-empty, numeric, uniqueness,
and header heuristics. Record filtered tables under
`no_candidate_entity_column`.

For a source table with multiple model-recoverable attributes, emit one variant
per qualifying bridge attribute when the table has enough ordinary context
columns. Treat every qualifying bridge attribute as target-only for the entire
source table, partition ordinary context columns once into query-only and
target-only sides, and never allow a sibling query and target to share a source
column. If the table is too narrow for both sides, fall back to the single
highest-recovery attribute.

When variants have identical visible column names and row values, write one
query table and attach all hidden attributes, target table IDs, chain IDs, and
qrels to it. Targets and recovery evidence remain attribute-specific.

## Testing

Use pytest with `tmp_path`, synthetic EntiTables payloads, fake Wikipedia clients, and fake extractors. No test may require network access, model servers, large generated output, or GPU hardware.

Add focused tests covering:

1. Identical seeds produce identical file/table ordering and discard choices.
2. Different seeds can produce different candidate orderings.
3. Candidate selection does not truncate by lexicographic file and table order.
4. Queryable tables are retained without a probability draw.
5. Failed tables are replaced only when the seeded draw falls below 0.5.
6. A failure not selected on one pass remains eligible on the next pass.
7. Each slot evaluates at most the original table and two replacements.
8. A failed table after the second replacement is retained.
9. Candidate exhaustion terminates and reports any unfilled slots.
10. Discarding a table prunes exclusive active assets but preserves its image and all three relevant JSONL cache record types.
11. Persistent model records load successfully after constructing a new cache instance.
12. Final source/query/data-lake artifacts, qrels, splits, decisions, stats, and manifest reference only the settled final pool.
13. Dynamic vLLM done markers are not written between candidate rounds and are written after the complete analysis phase.
14. CLI defaults and validation enforce two rounds and probability 0.5.
15. Tables without an entity-column candidate are excluded before global sampling.
16. Multi-attribute variants keep all sibling query/target source columns disjoint.
17. Identical visible variants merge into one query with multiple qrels.

Follow test-driven development: add one failing behavior test, verify its expected failure, implement the smallest supporting change, and rerun the focused test before proceeding.

Run the targeted tests and regression suite with:

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_extraction.py -q
```

Run any new focused test module separately during development, run the dynamic-runner tests contained in the extraction suite, and finish with `git diff --check` for the modified script and tests.

## Non-Goals

- Do not change the recovery threshold or attribute matching rules.
- Do not prefetch or infer over a fixed three-times-larger candidate pool.
- Do not delete downloaded or analyzed cache material merely because its source table was replaced.
- Do not hand-edit generated datasets in `output_medium/` or `output_stage1_logic/`.
- Do not add network-dependent test fixtures.
