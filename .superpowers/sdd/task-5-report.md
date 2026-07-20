# Task 5 Report: Integrate Final Selection and Materialization

## Status

Implemented and verified. `build_dataset(args)` now runs seeded candidate selection and bounded replacement before writing formal dataset artifacts. Final source/query/data-lake shards, qrels, splits, decisions, entities, bridge assets, and manifest entries are derived only from settled tables.

## TDD RED

Added `test_build_dataset_materializes_only_settled_replacement_tables` in `tests/test_mm_joinability_sampling.py` before modifying production code.

Command:

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py::test_build_dataset_materializes_only_settled_replacement_tables -q
```

Observed result: **1 failed**. The old builder ignored the monkeypatched random candidate stream, wrote its lexicographically parsed initial pool immediately, and emitted hashed `t0`/`t1` source IDs instead of the expected settled `t2`/`t1` pool. The failure was at the final source-shard assertion, which directly demonstrated the missing select-before-write behavior.

## GREEN Implementation

- Validates replacement policy and negative `max_source_tables` at `build_dataset` entry.
- Uses `SourceCandidateCounters`, `iter_random_source_tables`, the independently seeded replacement RNG, and `run_replacement_rounds` before formal artifact creation.
- Keeps bounded selection lazy. For the legacy programmatic `max_source_tables=None` case, exhausts the randomized stream once and selects its exact accepted count, avoiding the controller's invalid `None` target while preserving “all tables” behavior.
- Evaluates candidates through `CandidateEvaluationContext` and non-persistent record writers.
- Accumulates cleanup from immediate discards and an explicit final registry orphan sweep.
- Makes registry cleanup safe when `--no_wikipedia` supplies no Wikipedia client; the model cache is still compacted.
- Rebuilds entity and wiki-title maps from settled source tables only.
- Filters in-memory bridge assets and entity-to-asset links to final materializable entities, preserving the existing `max_entities` cap.
- Writes source, entity, bridge-asset, table-link, query, data-lake, extraction, recovery, qrel, split, decision, stats, and manifest artifacts from the final pool only.
- Regenerates queryability decisions during final materialization; candidate-only decisions are not persisted.
- Adds sampling mode/seed/policy, per-round replacement statistics, consumption/exhaustion/unfilled-slot fields, and complete cleanup counters to stats.
- Adds the reproducibility policy under `manifest["source_sampling"]`.

The integration test also verifies:

- `t0` is discarded, `t1` is retained, and `t2` settles as its replacement.
- source shards and decisions contain only `t1`/`t2`;
- bridge assets contain only final-table assets;
- qrels and splits exclude `t0`;
- manifest source shard totals equal rows written;
- round, exhaustion, cleanup, seed, and policy metadata have the expected shape and values.

## GREEN Verification

Focused integration command:

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py::test_build_dataset_materializes_only_settled_replacement_tables -q
```

Result: **1 passed**.

Required targeted regression command:

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py tests/test_mm_joinability_extraction.py -q
```

Fresh final result: **62 passed in 0.54s**.

Whitespace/error check:

```bash
git diff --check -- scripts/build_mm_joinability_dataset.py tests/test_mm_joinability_sampling.py tests/test_mm_joinability_extraction.py
```

Result: exit 0, no output.

Additional zero-target smoke run exercised `--max_source_tables 0`; it produced zero source rows without consuming input and wrote empty, explicit selection and cleanup metadata.

## Scope and Concerns

- Task 6's dynamic vLLM marker-lifetime redesign was intentionally not implemented. Task 5 necessarily moves extractor availability ahead of candidate evaluation and writes the existing start/ready handshake before selection; delayed done-marker semantics and nonzero round-mode startup signaling remain for Task 6.
- No changes were needed in `tests/test_mm_joinability_extraction.py`; the complete extraction suite was nevertheless included in the required verification.
- Generated `.superpowers` coordination/report files remain outside the Task 5 code commit unless explicitly staged by the coordinator.

## Blocking Review Findings Follow-up

### Status

Fixed all three blocking Task 5 review findings in one follow-up wave.

### RED 1: Dynamic Round-Mode Startup

Added:

- `test_dynamic_vllm_starts_servers_for_round_mode_with_unknown_task_counts`
- `test_model_start_marker_can_signal_round_mode`

Commands and observed RED results:

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_extraction.py::test_dynamic_vllm_starts_servers_for_round_mode_with_unknown_task_counts -q
```

Result: **1 failed** because the runner summed the zero text/image counts, returned with the builder, and started zero servers (`events.count("server_started") == 0`).

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_extraction.py::test_model_start_marker_can_signal_round_mode -q
```

Result: **1 failed** with `TypeError: write_model_start_marker() got an unexpected keyword argument 'round_mode'`.

GREEN implementation:

- Start markers now carry an explicit `round_mode` boolean.
- `build_dataset` emits `round_mode=true` before candidate-round evaluation.
- The dynamic runner interprets round-mode counts as not yet known instead of as zero pending work, so it starts the servers and releases the builder's ready-marker wait.
- Zero-count non-round markers retain the prior no-work fast path.

Focused GREEN command:

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_extraction.py::test_dynamic_vllm_starts_servers_for_round_mode_with_unknown_task_counts tests/test_mm_joinability_extraction.py::test_model_start_marker_can_signal_round_mode tests/test_mm_joinability_extraction.py::test_dynamic_vllm_skips_server_start_when_builder_has_no_pending_model_tasks -q
```

Result: **3 passed**.

### RED 2: Shared-Dependency Replacement Cleanup Ordering

Added `test_replacement_registers_shared_dependencies_before_discard_cleanup`.

RED command:

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py::test_replacement_registers_shared_dependencies_before_discard_cleanup -q
```

Result: **1 failed**. Immediate discard removed the shared asset and extraction cache before the replacement registered them, producing two fetches and two inferences instead of one each.

GREEN implementation:

- The replacement controller keeps the same `discard_table(table_id)` callback API.
- Drawn replacements enqueue their predecessors for cleanup.
- Pending discards are flushed only after the next replacement batch evaluates, its result IDs validate, and its dependencies register.
- Shared material is therefore visible to both reuse and registry protection before cleanup.

Focused GREEN command:

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py::test_replacement_registers_shared_dependencies_before_discard_cleanup tests/test_mm_joinability_sampling.py::test_failed_slot_replaces_twice_then_retains_at_limit tests/test_mm_joinability_sampling.py::test_build_dataset_materializes_only_settled_replacement_tables -q
```

Result: **3 passed**. The regression test proves one fetch, one inference, no cleanup before replacement registration, and preservation of the shared asset/model cache.

### RED 3: Stable Candidate Entity Budget

Added `test_candidate_selection_and_materialization_share_first_seen_entity_budget`.

RED command:

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py::test_candidate_selection_and_materialization_share_first_seen_entity_budget -q
```

Result: **1 failed**. Both candidate entities acquired assets during evaluation; final alphabetical slicing then retained `Page A`, although first-seen `Page Z` owned the one-entity budget.

GREEN implementation:

- `CandidateEvaluationContext` owns the persistent `max_entities` limit and eligible entity-ID set.
- Candidate entity IDs preserve first row/cell encounter order rather than passing through an unordered set.
- Only in-budget entity IDs can fetch assets, produce model evidence, or make a candidate queryable.
- The same registry-managed eligible material reaches final output; the later alphabetical entity re-slice and direct uncounted asset deletion were removed.
- Eligibility is intentionally stable across replacement rounds: discarded candidates do not release budget for later entities.

Focused GREEN command:

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py::test_candidate_selection_and_materialization_share_first_seen_entity_budget tests/test_mm_joinability_sampling.py::test_candidate_evaluation_collects_records_without_final_shard_writes tests/test_mm_joinability_sampling.py::test_build_dataset_materializes_only_settled_replacement_tables -q
```

Result: **3 passed**. Only `Page Z` materializes, `first` is queryable, and the out-of-budget `second` table is failed.

### Full Follow-up Verification

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py tests/test_mm_joinability_extraction.py -q
```

Result: **66 passed in 0.55s**.

```bash
git diff --check
```

Result: exit 0, no output.

### Remaining Concern

Task 6 still owns the complete dynamic done-marker lifetime redesign. This follow-up only supplies the minimal startup signal needed to prevent the new candidate-round path from deadlocking; it does not redefine when per-modality done markers should be emitted across the entire round lifecycle.
