# WDC 200K Scale Validation

## Status and scope

This report separates automated code gates from measurements made against real
WDC Schema.org 2023 tables. Code-level behavior can be verified locally with
synthetic inputs. The 100-, 1,000-, and 10,000-table measurements below must be
filled from actual runs by the formal-run operator.

No 100-, 1,000-, or 10,000-table acceptance gate has completed, and no formal
200K launch has occurred. After this report was initially prepared, one
explicitly partial broad `round_robin` diagnostic was run and is disclosed
below. It does not replace or pass any acceptance gate. Every unmeasured
acceptance field remains marked **NOT RUN**; no value in this template is a
synthetic substitute for a scale result.

## Reproducibility record

- Pipeline commit: **NOT RECORDED — fill immediately before each gate**
- Python environment: `MMDD`
- Full corpus root: **NOT RECORDED**
- Full corpus statistics-archive identity/checksums: **NOT RECORDED**
- Host/kernel: **NOT RECORDED**
- CPU/RAM/GPU: **NOT RECORDED**
- Filesystem and mount options for output/work/cache: **NOT RECORDED**
- Free disk before gates: **NOT RECORDED**
- `min_free_disk_bytes`: **NOT RECORDED**
- Network policy/user agent: **NOT RECORDED**
- Model identities and prompt versions: **NOT RECORDED**

For each run, retain the exact command, `git rev-parse HEAD`, the gate-input
`scale_gate_manifest.jsonl` and `scale_gate_checksums.json` when applicable,
the final `progress.json`, stage manifests, and canonical
`dataset_manifest.json`. Do not redirect runner stdout/stderr to a log file;
observe it directly in tmux and record the bounded measurements below.

## Automated code gates

The following gates are implemented and have focused automated coverage. The
listed suite passed on the documented working tree at
`2026-07-18T05:39:09Z`.

| Gate | Code evidence | Status |
|---|---|---|
| Output/work/cache roots are distinct | `test_pipeline_config_requires_separate_roots` | **PASS** |
| Runtime markers stay under `work_dir/runtime` | `test_runtime_root_must_equal_work_runtime` and related runtime-path tests | **PASS** |
| `--dry_run` is read-only and does not use network | `test_dry_run_validates_without_writes_or_network` | **PASS** |
| Source rows have no implicit cap and are preserved | `test_iter_wdc_rows_has_no_implicit_row_limit`, `test_structural_expansion_preserves_all_rows_and_removes_image` | **PASS** |
| Structural barrier performs no network work | `test_stop_after_structural_emits_exact_counts_without_network` | **PASS** |
| Resume does not replay durable URL outcomes | `test_terminal_page_failure_is_not_replayed_on_resume`, `test_full_pipeline_uses_real_stage_contracts_and_resume_does_not_refetch` | **PASS** |
| Duplicate page URLs produce one physical request | `test_duplicate_page_urls_make_one_physical_request` | **PASS** |
| `--from_stage` archives rather than deletes state | `test_from_stage_moves_named_and_downstream_without_deleting`, `test_from_stage_rebuilds_downstream_and_reuses_durable_url_cache` | **PASS** |
| Checksums reject corrupt checkpoints/manifests | shard, registry, network snapshot, and materialization checksum tests | **PASS** |
| Disk reserve is checked before writes | `test_preflight_stops_before_disk_reserve_is_consumed` and guarded-write tests | **PASS** |
| Progress is atomic, bounded, and has rolling rate/ETA | `test_progress_stdout_is_bounded_between_periodic_snapshots`, `test_progress_rolling_rate_uses_a_fixed_time_window` | **PASS** |
| Dynamic runner targets staged WDC 200K builder | `test_dynamic_vllm_default_builder_is_wdc200k`, `test_dynamic_vllm_help_names_staged_wdc200k_builder` | **PASS** |
| Scale-gate subcorpus is exact, deterministic, and symlink-only | `tests/test_wdc200k_scale_gate_input.py` | **PASS** |

## Measurement method

### Input isolation

The full corpus has more mandatory `top100` candidates than the 100- and
1,000-table targets. Running the production selector on the full corpus with
either small target is therefore invalid by policy. Prepare exact real-table
gate inputs:

```bash
conda run -n MMDD python scripts/create_wdc200k_scale_gate_input.py \
  --source_dir wdc_schemaorg_2023 \
  --target_dir gate_inputs/wdc_100 \
  --table_count 100 \
  --selection_mode global_lowest \
  --seed 13

conda run -n MMDD python scripts/create_wdc200k_scale_gate_input.py \
  --source_dir wdc_schemaorg_2023 \
  --target_dir gate_inputs/wdc_1000 \
  --table_count 1000 \
  --selection_mode global_lowest \
  --seed 13
```

These quick gates globally minimize source row counts by
`(rows, stable_hash(seed, relative_path), relative_path)`. They validate
pipeline plumbing, interruption/resume, request deduplication, checksums,
resource bounds, and canonical output. They are not representative samples of
WDC category balance, table quality, or row-count distribution, and they do
not change the formal stratified 200K selection. The helper writes filtered
production-format statistics ZIPs, symlinks gzip tables to absolute source
paths, and records both source gzip hashes and generated-file checksums. The
target must be absent or strictly empty.

### Fresh roots and interruption

Each scale uses a new, empty set of output/work/cache roots. Record their
paths; do not delete or recycle prior runs. Run the pipeline with direct stdout.
For the resume gate, interrupt with one normal `Ctrl-C` after terminal page
outcomes exist, record the last `progress.json`, then rerun the identical
command with `--resume`.

Measure peak RSS for both the initial and resumed processes using the platform
process monitor or `/usr/bin/time -v` without redirecting output. Sample
output/work/cache byte counts and free bytes from `progress.json` at start,
stage boundaries, interruption, peak, and completion.

### Counting definitions

- **Selected/validated tables:** stage manifests and selection/structural
  counters, reconciled to the expected gate size.
- **Entities/source rows:** structural manifests and published source-table
  records. Compare every source table's emitted row count with its decompressed
  source gzip row count; report mismatches, not only totals.
- **Unique URL jobs:** normalized unique-job count in the stage result.
- **Duplicate physical requests:** total transport invocations minus distinct
  normalized URL keys attempted. It must be zero. Count the same key appearing
  in multiple entity references as deduplication, not as a duplicate request.
- **Terminal replay on resume:** transport invocations after resume for URL
  keys already stored as `success` or `terminal` before interruption. It must
  be zero.
- **Peak RSS:** maximum resident set size for the pipeline process, with the
  measurement tool and units recorded.
- **Disk growth:** ending bytes minus starting bytes, separately for output,
  work, and cache; also record maximum observed bytes and minimum free bytes.
- **Manifest validity:** validate every published shard path, record count,
  byte count, SHA-256, upstream fingerprint, stage registry, and final
  `dataset_manifest.json`.
- **ETA factor:** during the final half of a stage, for each sample with a
  non-null ETA compute `predicted_remaining_seconds / actual_remaining_seconds`.
  Report the maximum of that ratio and its reciprocal. Passing is at most 2.0.

## Broad round-robin diagnostic (partial, not acceptance)

After initial report preparation, a separate 100-table `round_robin`
diagnostic reached structural expansion and made partial page-stage progress.
Its provenance is:

- Pipeline commit:
  `4f178e8e52be85670c40d24f948857b8efc281e3`
- Gate input: `/home/oycy/MMDD/gate_inputs/wdc_100`
- `scale_gate_manifest.jsonl` SHA-256:
  `c23fc5fc528ea72600e72b4919db4eff1d5c5fcbbd3dc5b32efc921e2b28d8c5`
- `scale_gate_checksums.json` SHA-256:
  `243f0c5e9ea19bf8982aa7a818517c4a0814d6998e82e8d413fb9198c4dd88f2`
- Configured output/work/cache roots:
  `/home/oycy/MMDD/output_wdc_gate100b`,
  `/home/oycy/MMDD/work_wdc_gate100b`, and
  `/home/oycy/MMDD/cache/wdc_gate100b`. The output root was not created
  before interruption.
- Runtime root: `/home/oycy/MMDD/work_wdc_gate100b/runtime`
- `min_free_disk_bytes`: `107374182400`
- Measurement tool: `/usr/bin/time -v`
- Durable state paths:
  `/home/oycy/MMDD/work_wdc_gate100b/progress.json`,
  `/home/oycy/MMDD/cache/wdc_gate100b/page_cache/outcomes.sqlite3`, and
  `/home/oycy/MMDD/work_wdc_gate100b/page_jobs/jobs.sqlite3`

Both commands were launched from `/home/oycy/MMDD`. The exact initial command
captured from the tmux run was:

```bash
/usr/bin/time -v conda run -n MMDD --no-capture-output python \
  scripts/run_mm_joinability_dynamic_vllm.py \
  --input_dir /home/oycy/MMDD/gate_inputs/wdc_100 \
  --output_dir /home/oycy/MMDD/output_wdc_gate100b \
  --work_dir /home/oycy/MMDD/work_wdc_gate100b \
  --cache_dir /home/oycy/MMDD/cache/wdc_gate100b \
  --text_model_path /home/oycy/MMDD/hf_models/Qwen3.5-9B \
  --image_model_path /home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking \
  --run_fingerprint gate100b-20260718 \
  --runtime_dir /home/oycy/MMDD/work_wdc_gate100b/runtime \
  --max_source_tables 100 \
  --selection_seed 13 \
  --web_max_retries 0 \
  --web_max_response_seconds 8 \
  --web_global_concurrency 128 \
  --web_per_host_concurrency 2 \
  --max_image_attempts_per_entity 3 \
  --max_images_per_entity 3 \
  --min_free_disk_bytes 107374182400 \
  --progress_interval_seconds 5
```

The exact resume command was:

```bash
/usr/bin/time -v conda run -n MMDD --no-capture-output python \
  scripts/run_mm_joinability_dynamic_vllm.py \
  --input_dir /home/oycy/MMDD/gate_inputs/wdc_100 \
  --output_dir /home/oycy/MMDD/output_wdc_gate100b \
  --work_dir /home/oycy/MMDD/work_wdc_gate100b \
  --cache_dir /home/oycy/MMDD/cache/wdc_gate100b \
  --text_model_path /home/oycy/MMDD/hf_models/Qwen3.5-9B \
  --image_model_path /home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking \
  --run_fingerprint gate100b-20260718 \
  --runtime_dir /home/oycy/MMDD/work_wdc_gate100b/runtime \
  --max_source_tables 100 \
  --selection_seed 13 \
  --web_max_retries 0 \
  --web_max_response_seconds 8 \
  --web_global_concurrency 128 \
  --web_per_host_concurrency 2 \
  --max_image_attempts_per_entity 3 \
  --max_images_per_entity 3 \
  --min_free_disk_bytes 107374182400 \
  --progress_interval_seconds 5 \
  --resume
```

The initial process ran for 1:07.87 and peaked at 293,064 KiB RSS; its
approximate interval was `2026-07-18T06:51:05Z` to
`2026-07-18T06:52:13Z`. The resume ran for 7:14.51 and peaked at 359,208 KiB
RSS; its approximate interval was `2026-07-18T06:53:58Z` to
`2026-07-18T07:01:13Z`. These timestamps are derived from tmux pane end times
and `/usr/bin/time -v` elapsed durations rather than independent wall-clock
samples.

The structural state recorded 100 selected and validated tables, 55,839
entities, 52,880 unique pages, and 322,324,462 structural bytes. At cutoff
`1784357532.9271808`, the 391 durable outcomes had full-row digest
`e66fc91e303e61173b9e5166111c97352c0009a5736bcbb36b707cb55b28ef9f`.
That digest was unchanged after resume. The database then held 4,906 total
outcomes, while `progress.json` recorded 4,515 live resume outcomes.

Both processes were interrupted. The diagnostic did not finish the page stage,
did not publish a `dataset_manifest.json`, and did not perform canonical
artifact validation. It is broad diagnostic evidence only and does not count
as the 100-table acceptance gate below.

## 100-table acceptance gate

Status: **NOT RUN**

### Run identity

- Gate input: **NOT RUN**
- `scale_gate_manifest.jsonl` SHA-256: **NOT RUN**
- `scale_gate_checksums.json` SHA-256: **NOT RUN**
- Pipeline commit: **NOT RUN**
- Exact initial command: **NOT RUN**
- Exact resume command: **NOT RUN**
- Output root: **NOT RUN**
- Work root: **NOT RUN**
- Cache root: **NOT RUN**
- Start/interruption/resume/end timestamps: **NOT RUN**
- Interruption stage and counters: **NOT RUN**

### Results

| Measurement | Required | Observed |
|---|---:|---:|
| Selected tables | 100 | **NOT RUN** |
| Validated source tables | 100 | **NOT RUN** |
| Expected source rows | exact source count | **NOT RUN** |
| Emitted source rows | equal expected | **NOT RUN** |
| Row-count mismatches | 0 | **NOT RUN** |
| Entities | record exact value | **NOT RUN** |
| Unique page URLs | record exact value | **NOT RUN** |
| Unique image URLs attempted | record exact value | **NOT RUN** |
| Duplicate physical requests | 0 | **NOT RUN** |
| Terminal URL replays after resume | 0 | **NOT RUN** |
| Peak RSS, initial process | record tool and units | **NOT RUN** |
| Peak RSS, resumed process | record tool and units | **NOT RUN** |
| Output start/peak/end bytes | record all three | **NOT RUN** |
| Work start/peak/end bytes | record all three | **NOT RUN** |
| Cache start/peak/end bytes | record all three | **NOT RUN** |
| Minimum free bytes | above configured reserve | **NOT RUN** |
| Invalid shard/stage checksums | 0 | **NOT RUN** |
| Canonical artifact validation | pass | **NOT RUN** |

Decision and evidence notes: **NOT RUN**

## 1,000-table acceptance gate

Status: **NOT RUN**

### Run identity

- Gate input: **NOT RUN**
- Gate manifest/checksums SHA-256: **NOT RUN**
- Pipeline commit and exact commands: **NOT RUN**
- Fresh output/work/cache roots: **NOT RUN**
- Start/interruption/resume/end timestamps: **NOT RUN**

### Results

| Measurement | Required | Observed |
|---|---:|---:|
| Selected tables | 1,000 | **NOT RUN** |
| Validated source tables | 1,000 | **NOT RUN** |
| Expected/emitted source rows | exact equality | **NOT RUN** |
| Row-count mismatches | 0 | **NOT RUN** |
| Entities | record exact value | **NOT RUN** |
| Unique pages / image candidates | record exact values | **NOT RUN** |
| Duplicate physical requests | 0 | **NOT RUN** |
| Terminal URL replays after resume | 0 | **NOT RUN** |
| Peak RSS | stable or sublinear vs. 100 gate | **NOT RUN** |
| Peak-RSS growth calculation | record ratio and method | **NOT RUN** |
| Output/work/cache disk growth | record separately | **NOT RUN** |
| Invalid manifest checksums | 0 | **NOT RUN** |
| Final-half ETA factor | <= 2.0 | **NOT RUN** |
| Canonical artifact validation | pass | **NOT RUN** |

Decision and evidence notes: **NOT RUN**

## 10,000-table structural/preflight gate

Status: **NOT RUN**

This gate uses the full corpus directly with
`--max_source_tables 10000 --stop_after structural`. It is not a filtered
gate-input sample and must perform no network or GPU model work.

### Run identity

- Full corpus identity/checksums: **NOT RUN**
- Pipeline commit and exact command: **NOT RUN**
- Fresh output/work/cache roots: **NOT RUN**
- Start/end timestamps: **NOT RUN**

### Structural results

| Measurement | Required | Observed |
|---|---:|---:|
| Selected tables | 10,000 | **NOT RUN** |
| Validated source tables | 10,000 | **NOT RUN** |
| Source rows preserved | exact | **NOT RUN** |
| Entities | record exact value | **NOT RUN** |
| Unique page URLs | record exact value | **NOT RUN** |
| Direct image candidates | record exact value | **NOT RUN** |
| Projected page requests | record exact upper bound | **NOT RUN** |
| Projected image requests | record exact upper bound | **NOT RUN** |
| Projected output/work/cache bytes | record separately | **NOT RUN** |
| Peak RSS | record tool and units | **NOT RUN** |
| Invalid structural checksums | 0 | **NOT RUN** |
| Minimum free bytes | above configured reserve | **NOT RUN** |
| Reserve satisfiable for network continuation | explicit yes/no | **NOT RUN** |

Network-stage continuation decision and evidence: **NOT RUN**

## Formal 200K launch gate

Status: **BLOCKED UNTIL ALL SCALE GATES PASS**

Before formal launch, the operator must:

1. Mark the 100-table gate passing, including interruption/resume and canonical
   artifact validation.
2. Mark the 1,000-table gate passing, including RSS growth, zero terminal
   replay, exact rows, valid checksums, and ETA factor.
3. Mark the 10,000-table structural gate passing and explicitly confirm the
   live disk reserve can support projected network work.
4. Run the fresh verification commands below at the launch commit.
5. Launch with direct tmux stdout and a separate `progress.json` window; do not
   stop or reuse an unrelated tmux session as part of this report.

Formal command, tmux session, launch timestamp, commit, and first-network-
interval observations: **NOT RUN**

## Fresh verification record

Run:

```bash
conda run -n MMDD python -m pytest \
  tests/test_wdc200k_io.py \
  tests/test_wdc200k_selection.py \
  tests/test_wdc200k_structural.py \
  tests/test_wdc200k_fetch.py \
  tests/test_wdc200k_assets.py \
  tests/test_wdc200k_models.py \
  tests/test_wdc200k_materialize.py \
  tests/test_wdc200k_pipeline.py \
  tests/test_wdc200k_scale_gate_input.py \
  tests/test_wdc_mm_joinability_dataset.py \
  tests/test_mm_joinability_extraction.py \
  tests/test_stage1_pipeline.py -q
conda run -n MMDD python -m py_compile \
  scripts/wdc200k_io.py \
  scripts/wdc200k_selection.py \
  scripts/wdc200k_structural.py \
  scripts/wdc200k_fetch.py \
  scripts/wdc200k_assets.py \
  scripts/wdc200k_models.py \
  scripts/wdc200k_materialize.py \
  scripts/build_wdc200k_mm_joinability_dataset.py \
  scripts/create_wdc200k_scale_gate_input.py \
  scripts/run_mm_joinability_dynamic_vllm.py
git diff --check
```

- Verification timestamp: `2026-07-18T05:39:09Z`
- Working tree base: `4f34aff44a0efb9ae07ce54b6403c4a8595fa36e`
- Pytest result: **PASS — 560 passed in 42.81s**
- `py_compile` result: **PASS**
- CLI help smoke result: **PASS** for the pipeline, scale-gate helper, and
  dynamic vLLM runner
- `git diff --check` result: **PASS**
