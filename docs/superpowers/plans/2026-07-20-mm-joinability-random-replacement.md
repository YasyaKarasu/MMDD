# MM Joinability Random Sampling and Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Randomize source-table sampling reproducibly, replace 50% of unrecoverable tables for at most two rounds, and delete cache material owned only by discarded tables.

**Architecture:** Add a seeded lazy candidate iterator, a pure slot/round controller, and a reference-aware material registry. Evaluate candidates without final writes, settle the table pool, then materialize existing artifacts for only that pool from cached model results.

**Tech Stack:** Python 3.10+, standard-library `argparse`, `dataclasses`, `json`, `pathlib`, `random`, existing JSONL helpers, and pytest.

## Global Constraints

- `--seed` controls candidate order and discard draws.
- Replacement rounds default to `2`; drop probability defaults to `0.5`.
- A slot sees at most its original table and two replacements.
- A failed table at the limit, or without an available replacement, is retained.
- Cleanup preserves dependencies referenced by any active or final table.
- Candidate evaluation never writes final output shards.
- Tests never use network access, model servers, or GPUs.

---

### Task 1: Seeded Candidate Stream and Policy

**Files:**
- Modify: `scripts/build_mm_joinability_dataset.py`
- Create: `tests/test_mm_joinability_sampling.py`

**Interfaces:**
- Produces: `ReplacementPolicy(rounds: int, drop_probability: float)`
- Produces: `replacement_policy_from_args(args: argparse.Namespace) -> ReplacementPolicy`
- Produces: `SourceCandidateCounters(processed_tables, skipped_tables, skip_reasons)`
- Produces: `iter_random_source_tables(input_dir, args, counters) -> Iterator[dict[str, Any]]`

- [ ] **Step 1: Write failing policy and ordering tests**

Create `tests/test_mm_joinability_sampling.py`. Build two small EntiTables JSON files, each containing four structurally valid tables. Test that defaults equal `ReplacementPolicy(2, 0.5)`, negative rounds and probabilities outside `[0, 1]` raise `ValueError`, equal seeds produce equal source-table ID order, seed `13` is not lexicographic, and seed `29` differs.

Use this helper in the test:

```python
def candidate_ids(input_dir: Path, seed: int) -> list[str]:
    args = builder.parse_args([
        "--input_dir", str(input_dir),
        "--output_dir", str(input_dir / "out"),
        "--seed", str(seed),
    ])
    counters = builder.SourceCandidateCounters()
    return [
        table["source_table_id"]
        for table in builder.iter_random_source_tables(input_dir, args, counters)
    ]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py -q
```

Expected: failure because the policy, counters, and iterator do not exist.

- [ ] **Step 3: Implement the minimal interfaces**

Import `random`, `Iterator`, and `field as dataclass_field`, then add:

```python
@dataclass(frozen=True)
class ReplacementPolicy:
    rounds: int
    drop_probability: float


@dataclass
class SourceCandidateCounters:
    processed_tables: int = 0
    skipped_tables: int = 0
    skip_reasons: Counter[str] = dataclass_field(default_factory=Counter)


def replacement_policy_from_args(args: argparse.Namespace) -> ReplacementPolicy:
    rounds = int(args.unrecoverable_replacement_rounds)
    probability = float(args.unrecoverable_drop_probability)
    if rounds < 0:
        raise ValueError("unrecoverable replacement rounds must be non-negative")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("unrecoverable drop probability must be within [0, 1]")
    return ReplacementPolicy(rounds, probability)
```

Implement `iter_random_source_tables` with a dedicated `random.Random(args.seed)`: shuffle `list(input_dir.rglob("*.json"))`, load each payload, shuffle `list(payload.items())`, call the existing `parse_source_table`, update counters, and yield only valid tables.

Add parser arguments:

```python
parser.add_argument("--unrecoverable_replacement_rounds", type=int, default=2)
parser.add_argument("--unrecoverable_drop_probability", type=float, default=0.5)
```

- [ ] **Step 4: Verify GREEN**

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py tests/test_mm_joinability_extraction.py::test_query_rows_per_table_defaults_to_five -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_mm_joinability_dataset.py tests/test_mm_joinability_sampling.py
git commit -m "Add seeded joinability candidate stream"
```

---

### Task 2: Bounded Replacement Controller

**Files:**
- Modify: `scripts/build_mm_joinability_dataset.py`
- Modify: `tests/test_mm_joinability_sampling.py`

**Interfaces:**
- Produces: `CandidateEvaluation(source_table, queryable, decision)`
- Produces: `ReplacementRoundStats(round_index, evaluated, unrecoverable, discarded, retained_failed, replacements)`
- Produces: `ReplacementSelection(final_evaluations, rounds, candidates_consumed, candidate_exhausted, unfilled_slots)`
- Produces: `run_replacement_rounds(candidate_tables, target_count, policy, rng, evaluate_batch, discard_table) -> ReplacementSelection`

- [ ] **Step 1: Write failing state-machine tests**

Use tables `t0` through `t4`, an evaluator backed by `{table_id: queryable}`, and this deterministic RNG:

```python
class StubRandom:
    def __init__(self, draws: list[float]):
        self.draws = iter(draws)

    def random(self) -> float:
        return next(self.draws)
```

Cover four behaviors separately: a queryable table never draws or replaces; a failed table with draw `0.5` is retained at probability `0.5`; draws below `0.5` replace twice and the third failed table is retained; probability `1.0` with an exhausted candidate iterator retains the current table and does not call cleanup.

- [ ] **Step 2: Verify RED**

Run the four controller tests. Expected: failure because the controller types are missing.

- [ ] **Step 3: Implement the controller**

Use this signature:

```python
def run_replacement_rounds(
    *,
    candidate_tables: Iterator[dict[str, Any]],
    target_count: int,
    policy: ReplacementPolicy,
    rng: Any,
    evaluate_batch: Callable[[list[dict[str, Any]]], list[CandidateEvaluation]],
    discard_table: Callable[[str], None],
) -> ReplacementSelection:
```

Fill slots up to `target_count`, evaluate pending slots in batches, and preserve slot order. Validate that results match the batch length and source IDs. For a failed table below the limit, draw once. When a draw requests replacement, fetch the next candidate before calling `discard_table`; if exhausted, settle the current table. Increment replacement count only after a replacement exists. At the configured limit, settle without drawing.

- [ ] **Step 4: Verify GREEN**

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_mm_joinability_dataset.py tests/test_mm_joinability_sampling.py
git commit -m "Add bounded joinability replacement rounds"
```

---

### Task 3: Reference-Aware Cache Cleanup

**Files:**
- Modify: `scripts/build_mm_joinability_dataset.py`
- Modify: `tests/test_mm_joinability_sampling.py`

**Interfaces:**
- Produces: frozen `CandidateDependencies` sets for entities, assets, paths, URLs, page keys, imageinfo keys, and model keys
- Produces: `CacheCleanupStats`
- Produces: `CacheCleanupStats.add(other: CacheCleanupStats) -> None`
- Produces: `CandidateMaterialRegistry.register(table_id, dependencies)`, `.discard(table_id)`, and `.sweep(final_table_ids)`
- Produces: `compact_keyed_jsonl(path, records) -> None`

- [ ] **Step 1: Write failing exclusive/shared cleanup tests**

Create two temporary image files and three JSONL caches. Register table A with exclusive and shared dependencies and table B with only shared dependencies. After discarding A, assert its exclusive file and cache keys are gone, B's shared file and cache keys remain, and cleanup byte/record counters are correct. Add a test where image unlink raises `OSError`; assert the error is counted and the remaining cache compactions still occur.

- [ ] **Step 2: Verify RED**

Run only the cleanup tests. Expected: failure because the registry is missing.

- [ ] **Step 3: Implement atomic compaction and registry cleanup**

Implement:

```python
def compact_keyed_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
```

The registry computes retained unions from all registered tables except the discarded ID. Remove only set differences from in-memory asset maps, Wikipedia `page_cache`/`image_cache`, and `ExtractionCache.items`. Unlink an image only when neither its resolved path nor source URL is retained. Compact `wiki_pages.jsonl`, `wiki_images.jsonl`, and `model_attribute_extractions.jsonl` after dictionary updates. Handle each unlink/compaction `OSError` independently with a warning and error counter.

Implement `sweep(final_table_ids)` by discarding every registered table absent from the final ID set and accumulating each result with `CacheCleanupStats.add`. This gives final selection one explicit orphan-cleanup interface.

- [ ] **Step 4: Verify GREEN and cache regression**

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py tests/test_mm_joinability_extraction.py::test_resolve_extraction_tasks_does_not_cache_failed_model_outputs -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_mm_joinability_dataset.py tests/test_mm_joinability_sampling.py
git commit -m "Clean discarded joinability candidate caches"
```

---

### Task 4: Non-Persistent Candidate Evaluation

**Files:**
- Modify: `scripts/build_mm_joinability_dataset.py`
- Modify: `tests/test_mm_joinability_sampling.py`

**Interfaces:**
- Produces: `ListRecordWriter.records`
- Produces: `CandidateEvaluationContext`
- Produces: `evaluate_candidate_batch(source_tables, context, args) -> list[CandidateEvaluation]`

- [ ] **Step 1: Write a failing no-final-write test**

Monkeypatch `build_table_join_records` to write extraction/recovery records into supplied handles and return one recoverable and one failed decision. Assert returned queryability values, registry entries for both tables, and absence of `attribute_extractions` and `evidence_recoveries` under the final output directory.

- [ ] **Step 2: Verify RED**

Run that test. Expected: failure because the context and evaluator are missing.

- [ ] **Step 3: Implement record collection and dependency capture**

Add:

```python
class ListRecordWriter:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write_record(self, record: dict[str, Any]) -> None:
        self.records.append(record)
```

For each table, build/update its entity references and assets, call `build_table_join_records` with fresh `ListRecordWriter` instances, and register dependencies derived from the table, referenced assets, and extraction cache keys. Return only the source table, queryable boolean, and compact decision. Do not create `ShardedJsonlWriter` in this helper.

- [ ] **Step 4: Verify GREEN and extraction regressions**

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py tests/test_mm_joinability_extraction.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_mm_joinability_dataset.py tests/test_mm_joinability_sampling.py
git commit -m "Evaluate joinability candidates before materialization"
```

---

### Task 5: Integrate Final Selection and Materialization

**Files:**
- Modify: `scripts/build_mm_joinability_dataset.py`
- Modify: `tests/test_mm_joinability_sampling.py`
- Modify: `tests/test_mm_joinability_extraction.py`

**Interfaces:**
- Consumes: Tasks 1-4 interfaces
- Produces: existing `build_dataset(args) -> dict[str, Any]` with final-only artifacts and replacement metadata

- [ ] **Step 1: Write failing final-artifact integration tests**

Use a small target count and monkeypatched material/model functions. Arrange `t0` to fail and be discarded, `t1` to succeed, and `t2` to settle as the replacement. Assert final source shards and decisions contain `t1`/`t2`, never `t0`; manifest shard counts match rows; splits/qrels contain no discarded IDs; stats include seed, rounds, probability, per-round counts, cleanup counts, and candidate-exhaustion fields.

- [ ] **Step 2: Verify RED**

Run the new integration test. Expected: failure because `build_dataset` still fixes source tables before recovery.

- [ ] **Step 3: Refactor `build_dataset` to select before writing**

Validate the replacement policy at entry. Replace the lexicographic source-writing loop with the candidate iterator and controller:

```python
counters = SourceCandidateCounters()
candidate_tables = iter_random_source_tables(input_dir, args, counters)
selection_rng = random.Random(int(stable_hash("replacement", args.seed), 16))
selection = run_replacement_rounds(
    candidate_tables=candidate_tables,
    target_count=args.max_source_tables,
    policy=policy,
    rng=selection_rng,
    evaluate_batch=lambda batch: evaluate_candidate_batch(batch, evaluation_context, args),
    discard_table=lambda table_id: cleanup_totals.add(
        evaluation_context.registry.discard(table_id)
    ),
)
final_source_tables = [item.source_table for item in selection.final_evaluations]
```

Rebuild entity maps from `final_source_tables`, write only those tables to `source_writer`, filter final bridge assets to final entity IDs, run the registry orphan sweep, and then execute the existing split/query/data-lake/qrel/manifest writers over the final source shards. Keep progress closure in a `finally` covering evaluation and materialization.

- [ ] **Step 4: Add stats and manifest configuration**

Set legacy processed/skipped counters from `SourceCandidateCounters`. Add top-level sampling seed and policy values, `replacement_selection` with `dataclasses.asdict` round entries, and cleanup totals. Add the same reproducibility configuration under `manifest["source_sampling"]`. Decisions must be regenerated only for final tables.

- [ ] **Step 5: Verify GREEN**

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py tests/test_mm_joinability_extraction.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_mm_joinability_dataset.py tests/test_mm_joinability_sampling.py tests/test_mm_joinability_extraction.py
git commit -m "Integrate random joinability table replacement"
```

---

### Task 6: Dynamic vLLM Marker Lifetime

**Files:**
- Modify: `scripts/build_mm_joinability_dataset.py`
- Modify: `tests/test_mm_joinability_extraction.py`

**Interfaces:**
- Produces: `write_done_markers_after_selection(args, text_task_count, image_task_count) -> None`
- Extends: `write_model_start_marker(..., round_mode_requires_services: bool = False)`

- [ ] **Step 1: Write failing marker-timing tests**

Record marker writes around two candidate rounds. Assert no text/image done marker exists after round 0 and both exist after selection/final uncached work. Add a zero-first-round-task test asserting the start marker reports a positive runner startup count when unresolved candidate slots remain, while preserving actual text/image task counts separately.

- [ ] **Step 2: Verify RED**

Run the marker tests. Expected: failure because current precompute groups write done markers per invocation.

- [ ] **Step 3: Delay done markers across the complete analysis phase**

Prevent per-round `precompute_extraction_task_groups` calls from writing done markers. Accumulate actual text/image task counts across rounds and call `write_done_markers_after_selection` once after selection plus final uncached work. Extend the start marker with `round_mode_requires_services`; when true and unresolved slots exist, expose a positive startup count to the runner even if both actual first-round counts are zero.

No dynamic-runner code change is needed: `run_mm_joinability_dynamic_vllm.py` already keeps both initial services alive until a done marker or builder exit, so delayed markers span all rounds.

- [ ] **Step 4: Verify GREEN**

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_extraction.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_mm_joinability_dataset.py tests/test_mm_joinability_extraction.py
git commit -m "Keep joinability model services across replacement rounds"
```

---

### Task 7: Full Verification

**Files:**
- Verify: `scripts/build_mm_joinability_dataset.py`
- Verify: `tests/test_mm_joinability_sampling.py`
- Verify: `tests/test_mm_joinability_extraction.py`
- Verify: `docs/superpowers/specs/2026-07-20-mm-joinability-random-replacement-design.md`

**Interfaces:**
- Consumes: completed implementation
- Produces: verified repository state

- [ ] **Step 1: Run all relevant joinability tests**

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_sampling.py tests/test_mm_joinability_extraction.py tests/test_mm_joinability_viewer.py tests/test_mm_joinability_repair.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run Stage-1 regression tests**

```bash
conda run -n MMDD python -m pytest tests/test_stage1_pipeline.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Check CLI and formatting**

```bash
conda run -n MMDD python scripts/build_mm_joinability_dataset.py --help
git diff --check -- scripts/build_mm_joinability_dataset.py tests/test_mm_joinability_sampling.py tests/test_mm_joinability_extraction.py docs/superpowers/specs/2026-07-20-mm-joinability-random-replacement-design.md
```

Expected: help includes both replacement arguments; diff check prints no errors.

- [ ] **Step 4: Inspect scope without touching user data**

```bash
git status --short
git diff --stat HEAD~5..HEAD
```

Expected: feature commits contain only the intended script, tests, design, and plan. Existing untracked datasets and archives remain untouched.
