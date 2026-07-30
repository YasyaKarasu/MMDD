# WDC ETA Advisory Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement deterministic read-only ETA evidence validation, update the authoritative scale policy, rerun the ordered 100/1,000/10,000 gates on fresh roots, and launch the formal 200K run only after every hard prerequisite passes.

**Architecture:** A standalone ETA validator reads immutable `progress.json` bytes, delegates validation to the existing `ProgressReporter` strict restore path through an isolated temporary copy, and emits stable JSON without changing either input. A separate structural validator reuses the existing registry chain and adds exhaustive table-row, artifact-allowlist, and disk-projection evidence. The authoritative report consumes those JSON results; operational gates proceed strictly 100 -> 1,000 -> 10,000 -> 200K with fresh absent roots and fail-stop decisions.

**Tech Stack:** Python 3.10, `argparse`, `hashlib`, `json`, `pathlib`, `tempfile`, existing WDC telemetry-v2/`ProgressReporter`, pytest, tmux, `/usr/bin/time -v`, GNU `cp`/`chmod`/`sha256sum`, conda environment `MMDD`.

---

## Scope And File Map

- Create `scripts/validate_wdc_eta_gate.py`: read-only strict restore, native-v2 enforcement, ETA enumeration, resume-prefix comparison, deterministic JSON CLI.
- Create `tests/test_wdc_eta_gate_validator.py`: focused unit and CLI coverage, including input immutability.
- Create `scripts/validate_wdc_structural_gate.py`: read-only row, registry, artifact, and projection validation for the structural boundary.
- Create `tests/test_wdc_structural_gate_validator.py`: focused exhaustive reconciliation, tamper, projection, CLI, and immutability coverage.
- Modify `docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md`: policy, validator evidence, historical v2 limitation, and ordered gate results.
- Do not modify the v2 schema, estimator, reporter publication, URL fetch behavior, manifests, or pipeline semantics.
- Never delete, rename, truncate, recycle, or write into an existing gate root. Every gate below first proves its new output/work/cache/evidence paths are absent.
- Never redirect runner or validator stdout/stderr to a log. tmux pane history and the bounded report are the observation record. `progress-before-resume.json` is approved pipeline-state evidence, not a log.

## Exact Validator Contract

`scripts/validate_wdc_eta_gate.py` defines `RestoredProgress(path: Path, sha256: str, stage_telemetry: dict[str, dict[str, Any]])`, `GateValidationError(code: str, message: str)`, `restore_progress_read_only(path: Path) -> RestoredProgress`, `canonical_prefix_sha256(samples: Sequence[dict[str, Any]]) -> str`, `validate_eta_gate(progress_path: Path, before_resume_path: Path | None = None) -> dict[str, Any]`, `build_parser() -> argparse.ArgumentParser`, and `main(argv: Sequence[str] | None = None) -> int`.

Success is one newline-terminated JSON object serialized with `sort_keys=True`, compact separators, `ensure_ascii=True`, and `allow_nan=False`. Its top-level keys are `status`, `progress`, `before_resume`, and `stages`; `status` is `ok`; `progress` contains absolute `path` and exact-byte `sha256`; `before_resume` is null or the same two fields; and `stages` always contains `pages` and `images`.

Each stage contains `telemetry_schema_version`, `completed_at`, `eligible_final_half_samples`, `excluded_final_half_samples`, `max_symmetric_eta_factor`, complete `eligible_samples`, complete `excluded_samples`, `worst_record`, `worst_reason`, and `resume_prefix`. Each eligible record is the restored canonical sample plus `stage`, original zero-based `sample_position`, `actual_remaining_seconds`, and `symmetric_factor`. Each excluded record adds those same fields with null factor and one `exclusion_reason`: `PREDICTION_MISSING`, `PREDICTION_NON_POSITIVE`, or `ACTUAL_REMAINING_NON_POSITIVE`, evaluated in that order.

Eligible records sort by `(-symmetric_factor, sample_position)`; the first is `worst_record`. Excluded records remain in ascending original position. With zero eligible records, maximum and worst are null and `worst_reason` is `ZERO_ELIGIBLE_SAMPLES`.

For each stage present in `--before-resume`, `resume_prefix` contains `prefix_length`, `before_prefix_sha256`, `final_prefix_sha256`, and `matches`. Canonical prefix bytes are compact, key-sorted, ASCII JSON with `allow_nan=False` and no trailing newline. A recursive type-strict comparator checks each before sample against the equal-length final prefix: scalar types and values must match, dictionary key/value trees must match, and list order must match, so `1` and `1.0` are different.

Errors are one stable JSON line on stdout with `status: error` plus `error.code` and `error.message`; no traceback or temporary path is printed. Validation errors exit `1`. Missing, unknown, duplicated, or bad-arity CLI options emit `CLI_USAGE_ERROR` on stdout, leave stderr empty, and exit `2`. Codes are `CLI_USAGE_ERROR`, `INPUT_NOT_FOUND`, `INPUT_NOT_FILE`, `INPUT_READ_FAILED`, `STRICT_RESTORE_FAILED`, `MISSING_URL_STAGE`, `INCOMPLETE_URL_STAGE`, `NON_NATIVE_V2`, `PREFIX_STAGE_MISSING`, `PREFIX_TOO_LONG`, `PREFIX_MISMATCH`, and `JSON_SERIALIZATION_FAILED`.

### Task 1: Read-Only Deterministic ETA Gate Validator

**Files:**
- Create: `scripts/validate_wdc_eta_gate.py`
- Create: `tests/test_wdc_eta_gate_validator.py`
- Read only: `scripts/build_wdc200k_mm_joinability_dataset.py`
- Read only: `scripts/wdc200k_eta.py`

- [ ] **Step 1: Write RED strict-restore and immutability tests**

Build native-v2 page/image fixtures with the real `ProgressReporter` and `UrlProgressSnapshot`. Before validation record both input files' bytes, `st_mtime_ns`, and mode. Assert success restores both stages without changing any recorded property. Add completed-v1, missing-stage, incomplete-stage, malformed-histogram, and mixed-v1/v2 cases; expect the exact error codes above.

```python
def test_validate_eta_gate_restores_native_v2_without_writing_inputs(tmp_path: Path) -> None:
    progress = _completed_native_v2_progress(tmp_path)
    before_bytes = progress.read_bytes()
    before_stat = progress.stat()
    result = validate_eta_gate(progress)
    assert result["status"] == "ok"
    assert set(result["stages"]) == {"pages", "images"}
    assert progress.read_bytes() == before_bytes
    assert progress.stat().st_mtime_ns == before_stat.st_mtime_ns
    assert stat.S_IMODE(progress.stat().st_mode) == stat.S_IMODE(before_stat.st_mode)


def test_validate_eta_gate_rejects_completed_v1(tmp_path: Path) -> None:
    with pytest.raises(GateValidationError) as caught:
        validate_eta_gate(_completed_v1_progress(tmp_path))
    assert caught.value.code == "NON_NATIVE_V2"
```

- [ ] **Step 2: Run strict-restore tests RED**

Run: `conda run -n MMDD python -m pytest tests/test_wdc_eta_gate_validator.py -k 'restores_native or completed_v1 or missing_stage or incomplete_stage or malformed or mixed' -q`

Expected: collection FAIL with `ModuleNotFoundError: No module named 'validate_wdc_eta_gate'`.

- [ ] **Step 3: Implement immutable loading and existing strict restore reuse**

Resolve and validate the input, read bytes once, hash those bytes, and copy only the captured bytes to a `TemporaryDirectory` file named `progress.json`. Construct a direct `PipelineConfig` with distinct temporary input/output/work/cache paths and `resume=True`, instantiate `ProgressReporter`, and never call `start`, `update`, `publish`, or `stop`. Deep-copy `reporter._stage_telemetry` through an in-memory JSON round trip before closing the temporary directory. Wrap input errors and reporter exceptions in the fixed error codes without exposing temporary paths. This preserves the existing strict schema, bounds, histogram decode, topology, epoch, estimator recomputation, `completed_at`, and summary checks without changing production behavior.

- [ ] **Step 4: Enforce fresh native-v2 completion**

For the final input, require page and image stages, non-null `completed_at`, `completed_units == total_units`, stage marker exactly `wdc200k-url-telemetry-v2`, and the same marker on every retained sample. For the before-resume input, allow incomplete stages but require every stage and retained sample it contains to be native v2. Completed-v1 remains readable by the pipeline but is rejected as fresh gate evidence in either input.

- [ ] **Step 5: Run strict-restore tests GREEN**

Run the Step 2 command again.

Expected: all selected tests PASS.

- [ ] **Step 6: Write RED enumeration, exclusion, ordering, and zero-case tests**

Use fixed logical timestamps to produce factors `(4.0, position 3)`, `(2.0, position 1)`, and `(2.0, position 2)` and assert that exact order and that `worst_record` equals element zero. Add one test that unions eligible and excluded positions and proves every `completed_units * 2 >= total_units` sample occurs exactly once. Assert all three exclusion reasons and explicit zero-eligible output.

```python
def test_eligible_records_sort_factor_desc_then_position_asc() -> None:
    records = validate_eta_gate(_progress_with_tied_factors())["stages"]["pages"]["eligible_samples"]
    assert [(row["symmetric_factor"], row["sample_position"]) for row in records] == [(4.0, 3), (2.0, 1), (2.0, 2)]


def test_zero_eligible_has_explicit_reason() -> None:
    stage = validate_eta_gate(_progress_with_zero_eligible())["stages"]["pages"]
    assert stage["eligible_final_half_samples"] == 0
    assert stage["max_symmetric_eta_factor"] is None
    assert stage["worst_record"] is None
    assert stage["worst_reason"] == "ZERO_ELIGIBLE_SAMPLES"
```

- [ ] **Step 7: Run enumeration tests RED**

Run: `conda run -n MMDD python -m pytest tests/test_wdc_eta_gate_validator.py -k 'eligible_records or final_half or exclusion or zero_eligible' -q`

Expected: FAIL because ETA record enumeration is absent.

- [ ] **Step 8: Implement exact enumeration**

For every final-half sample compute `actual = completed_at - timestamp`; classify with the fixed exclusion precedence; for eligible values compute `max(predicted / actual, actual / predicted)`; sort by `(-factor, position)`; and verify derived counts/maximum exactly equal the strict restored stage summary before returning it.

- [ ] **Step 9: Run enumeration tests GREEN and commit the core**

Run: `conda run -n MMDD python -m pytest tests/test_wdc_eta_gate_validator.py -k 'eligible_records or final_half or exclusion or zero_eligible' -q`

Expected: all selected tests PASS.

Commit:

```bash
git add scripts/validate_wdc_eta_gate.py tests/test_wdc_eta_gate_validator.py
git commit -m "Add read-only ETA gate validator core"
```

- [ ] **Step 10: Write RED prefix and CLI tests**

Cover an equal prefix, value/type change, reorder, insertion, shorter final list, missing final stage, exact canonical digests, stable success stdout across two invocations, stable stdout-only validation errors, and unchanged bytes/timestamps/modes for both inputs. Add parameterized RED CLI tests for missing `--progress`, unknown option, duplicate `--progress`, duplicate `--before-resume`, and bad arity; every case must return `2`, parse stdout as `CLI_USAGE_ERROR`, and assert `captured.err == ""`.

```python
def test_cli_prefix_mismatch_is_stable_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    before, final = _mismatched_progress_pair(tmp_path)
    assert main(["--progress", str(final), "--before-resume", str(before)]) == 1
    assert json.loads(capsys.readouterr().out) == {"error": {"code": "PREFIX_MISMATCH", "message": "pages sample prefix differs"}, "status": "error"}


@pytest.mark.parametrize("argv", [[], ["--unknown"], ["--progress", "a", "--progress", "b"], ["--progress", "a", "--before-resume", "b", "--before-resume", "c"], ["--before-resume"], ["--progress"]])
def test_cli_usage_errors_are_stdout_only(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "CLI_USAGE_ERROR"
    assert captured.err == ""
```

- [ ] **Step 11: Run prefix/CLI tests RED**

Run: `conda run -n MMDD python -m pytest tests/test_wdc_eta_gate_validator.py -k 'prefix or cli or stable_json' -q`

Expected: FAIL because prefix comparison and CLI output are absent.

- [ ] **Step 12: Implement prefix comparison and CLI**

Compare list lengths, then each object in order with a recursive comparator that requires `type(left) is type(right)` at every scalar/list/dictionary node; hash before and final-prefix canonical bytes; fail closed on any discrepancy. Define `JsonArgumentParser(argparse.ArgumentParser).error()` to raise `GateValidationError("CLI_USAGE_ERROR", message)` instead of writing usage. Register both path options with `action="append"`; require exactly one progress value and at most one before-resume value after parsing so duplicates also fail. `main()` catches CLI usage separately, prints exactly one sorted compact JSON line, and returns `2`; it returns `0` for success and `1` for validation errors. Add `raise SystemExit(main())` under the normal module entrypoint.

- [ ] **Step 13: Run full focused validation and help**

Run: `conda run -n MMDD python -m pytest tests/test_wdc_eta_gate_validator.py -q`

Run: `conda run -n MMDD python scripts/validate_wdc_eta_gate.py --help`

Expected: focused tests PASS; help shows required `--progress` and optional `--before-resume`.

- [ ] **Step 14: Mutation-check input immutability and commit**

Temporarily add `resolved.touch()` after reading the input, run the immutability test and observe failure on `st_mtime_ns`, remove the mutation, and rerun the full test file to PASS.

```bash
git add scripts/validate_wdc_eta_gate.py tests/test_wdc_eta_gate_validator.py
git commit -m "Complete deterministic ETA gate validation"
```

### Task 2: Authoritative Scale Report Policy And Historical V2 Evidence

**Files:**
- Modify: `docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md`
- Read only: `docs/superpowers/specs/2026-07-19-wdc-eta-advisory-gate-design.md`

- [ ] **Step 1: Validate the completed historical v2 progress**

Run without `--before-resume`, because no full pre-resume copy exists:

```bash
conda run -n MMDD python scripts/validate_wdc_eta_gate.py --progress /home/oycy/MMDD/work_wdc_gate100_eta_v2/progress.json
```

Expected: exit `0`; progress SHA-256 `ee6a68617ef85d8c8646ae052f229ecc84eaa69cd5631ea34ab83399369590e8`; page maximum `3.9951699758322845`; image maximum `7.574134329760511`. Retain the complete stdout in the bounded report, not a separate log.

- [ ] **Step 2: Replace all obsolete ETA decision rules**

Replace, rather than supplement, the measurement-method `<= 2.0` rule, the 100/1,000 acceptance rules, and formal-launch ETA requirement with these exact concepts:

```text
ETA accuracy decision: ADVISORY ONLY; no numeric pass threshold
Telemetry integrity decision: HARD PASS | HARD FAIL | BLOCKED
Validator evidence: exact command, input SHA-256, complete deterministic JSON
```

Preserve all dataset, artifact, resume, transport, disk, and telemetry-integrity hard gates. State that the validator and focused tests must pass before the report can make a new decision; otherwise classify `NOT_RUN_OR_INCOMPLETE` and keep the gate `BLOCKED`.

- [ ] **Step 3: Update every scale-section evidence template**

Require hard decision/classifications, applicable stage boundary, complete artifact hashes/counts/bytes, native-v2 status, exact ETA and structural validator commands/results where applicable, full eligible/excluded lists, maximum/worst record, evidence-copy path/SHA/mode, per-stage prefix length/digests/comparison, anomaly counters, per-root start/peak/end/minimum observed free bytes, reserve, duration, and RSS. In the 10K structural section explicitly mark network/model/final-dataset artifacts and URL ETA `NOT APPLICABLE`.

- [ ] **Step 4: Record the historical v2 run as BLOCKED with exact retained evidence**

Add this evidence without inferring a full prefix proof:

```text
Execution commit: b2a27a817fdfb29992a1c56afbd58ba766332a3e
Input scale_gate_manifest.jsonl SHA-256: b9003584ade84bd5b13e1fbbfe05d10b9bacbbb0832963d16c82792eb7642473
Input scale_gate_checksums.json SHA-256: 990a3d5677a30345405e55f615bbbbf8f8c272b9612f909027dba8dcada4b224
Final progress.json SHA-256: ee6a68617ef85d8c8646ae052f229ecc84eaa69cd5631ea34ab83399369590e8
Final dataset_manifest.json SHA-256: 612dd934d751cd30333c257ca2f4dbdc88b6796d5623abd5b9f14531ec1b179b
Canonical page sample-array SHA-256: dcc4fe8a4c47ae2da9b444eb168ea9a6d27f37a00f893a753dd0cd65356df374
Canonical image sample-array SHA-256: 5cfe01c0ebdf45b00364fdb2b8b91930fa8fae650fde0133071318ee59b93182
Page ETA: eligible=50 excluded=1 max=3.9951699758322845
Image ETA: eligible=71 excluded=1 max=7.574134329760511
Page duplicate/replay/unfinished/blocked: 0/0/0/0
Image duplicate/replay/unfinished/blocked: 0/0/0/0
Completion process elapsed: 6:43.00
Completion process maximum RSS: 7034428 KiB
Reserve: 107374182400 bytes
Per-root minimum observed free bytes: work=794408525824 cache=794408525824 output=794408525824
Decision: BLOCKED
Classification: NOT_RUN_OR_INCOMPLETE
Reason: no complete read-only progress-before-resume.json was preserved
```

The two sample-array digests prove only equality of those recorded arrays at the cutoff. They do not prove full `progress.json` identity or an object-by-object prefix from a retained evidence file. Embed the exact Step 1 validator stdout as the advisory result and state that it cannot cure the missing hard resume evidence.

- [ ] **Step 5: Verify precedence and commit the report policy**

Run: `rg -n 'ETA|eta|2\.0|PASS|BLOCKED|NOT_RUN_OR_INCOMPLETE|progress-before-resume|validator' docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md`

Run: `git diff --check -- docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md`

Expected: no ETA `<= 2.0` pass decision remains; historical v2 evidence is advisory but the gate is `BLOCKED`; downstream structural-only fields are `NOT APPLICABLE`.

```bash
git add docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md
git commit -m "Make WDC ETA accuracy advisory"
```

### Task 3: Fresh 100-Table Advisory Gate

**Files:**
- Modify after the run: `docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md`
- Evidence only: `/home/oycy/MMDD/evidence_wdc_gate100_eta_advisory_20260719/progress-before-resume.json`

Use these new roots and do not substitute any existing `gate100` root:

```text
input=/home/oycy/MMDD/gate_inputs/wdc_100_lowest
output=/home/oycy/MMDD/output_wdc_gate100_eta_advisory_20260719
work=/home/oycy/MMDD/work_wdc_gate100_eta_advisory_20260719
cache=/home/oycy/MMDD/cache/wdc_gate100_eta_advisory_20260719
evidence=/home/oycy/MMDD/evidence_wdc_gate100_eta_advisory_20260719
runtime=/home/oycy/MMDD/work_wdc_gate100_eta_advisory_20260719/runtime
tmux=wdc_gate100_eta_advisory_20260719
fingerprint=gate100-eta-advisory-20260719
```

- [ ] **Step 1: Prove prerequisites, input identity, and absent roots**

Run each command separately:

```bash
test ! -e /home/oycy/MMDD/output_wdc_gate100_eta_advisory_20260719
test ! -e /home/oycy/MMDD/work_wdc_gate100_eta_advisory_20260719
test ! -e /home/oycy/MMDD/cache/wdc_gate100_eta_advisory_20260719
test ! -e /home/oycy/MMDD/evidence_wdc_gate100_eta_advisory_20260719
test -d /home/oycy/MMDD/hf_models/Qwen3.5-9B
test -d /home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking
printf '%s\n' 'text_model_path=/home/oycy/MMDD/hf_models/Qwen3.5-9B served_name=Qwen3.5-9B' 'image_model_path=/home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking served_name=Qwen3-VL-8B-Thinking'
for model in /home/oycy/MMDD/hf_models/Qwen3.5-9B /home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking; do for relative in .hfd/repo_metadata.json config.json model.safetensors.index.json; do if test -f "$model/$relative"; then sha256sum "$model/$relative"; else printf 'NOT PRESENT  %s\n' "$model/$relative"; fi; done; done
sha256sum /home/oycy/MMDD/gate_inputs/wdc_100_lowest/scale_gate_manifest.jsonl /home/oycy/MMDD/gate_inputs/wdc_100_lowest/scale_gate_checksums.json
git rev-parse HEAD
```

Expected: every `test` exits zero; hashes are `b9003584ade84bd5b13e1fbbfe05d10b9bacbbb0832963d16c82792eb7642473` and `990a3d5677a30345405e55f615bbbbf8f8c272b9612f909027dba8dcada4b224`; record the exact launch commit.

- [ ] **Step 2: Run read-only dry-run preflight**

```bash
conda run -n MMDD python scripts/build_wdc200k_mm_joinability_dataset.py --input_dir /home/oycy/MMDD/gate_inputs/wdc_100_lowest --output_dir /home/oycy/MMDD/output_wdc_gate100_eta_advisory_20260719 --work_dir /home/oycy/MMDD/work_wdc_gate100_eta_advisory_20260719 --cache_dir /home/oycy/MMDD/cache/wdc_gate100_eta_advisory_20260719 --runtime_dir /home/oycy/MMDD/work_wdc_gate100_eta_advisory_20260719/runtime --max_source_tables 100 --selection_seed 13 --web_max_retries 0 --web_max_response_seconds 8 --web_global_concurrency 128 --web_per_host_concurrency 2 --max_image_attempts_per_entity 3 --max_images_per_entity 3 --min_free_disk_bytes 107374182400 --progress_interval_seconds 5 --dry_run --no-resume
```

Expected: exit zero, no network/GPU work, and all four new roots remain absent.

- [ ] **Step 3: Launch initial process with direct tmux stdout**

```bash
tmux new-session -d -s wdc_gate100_eta_advisory_20260719 -n run -c /home/oycy/MMDD/.worktrees/wdc-200k "sleep 3600"
tmux set-option -t wdc_gate100_eta_advisory_20260719 remain-on-exit on
tmux respawn-pane -k -t wdc_gate100_eta_advisory_20260719:run "/usr/bin/time -v conda run -n MMDD --no-capture-output python scripts/run_mm_joinability_dynamic_vllm.py --input_dir /home/oycy/MMDD/gate_inputs/wdc_100_lowest --output_dir /home/oycy/MMDD/output_wdc_gate100_eta_advisory_20260719 --work_dir /home/oycy/MMDD/work_wdc_gate100_eta_advisory_20260719 --cache_dir /home/oycy/MMDD/cache/wdc_gate100_eta_advisory_20260719 --text_model_path /home/oycy/MMDD/hf_models/Qwen3.5-9B --image_model_path /home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking --run_fingerprint gate100-eta-advisory-20260719 --runtime_dir /home/oycy/MMDD/work_wdc_gate100_eta_advisory_20260719/runtime --max_source_tables 100 --selection_seed 13 --web_max_retries 0 --web_max_response_seconds 8 --web_global_concurrency 128 --web_per_host_concurrency 2 --max_image_attempts_per_entity 3 --max_images_per_entity 3 --min_free_disk_bytes 107374182400 --progress_interval_seconds 5 --no-resume"
tmux new-window -d -t wdc_gate100_eta_advisory_20260719 -n progress -c /home/oycy/MMDD/.worktrees/wdc-200k "watch -n 5 cat /home/oycy/MMDD/work_wdc_gate100_eta_advisory_20260719/progress.json"
```

Expected: runner output remains in `:run`; watcher only reads atomic progress; no shell redirection is used.

- [ ] **Step 4: Perform one planned interruption during native-v2 image progress**

Wait until `stage == "images"`, page telemetry is complete native v2, and at least one nonbaseline image sample is durable. Record the counters and timestamp in the report, then run:

```bash
tmux send-keys -t wdc_gate100_eta_advisory_20260719:run C-c
tmux display-message -p -t wdc_gate100_eta_advisory_20260719:run '#{pane_dead} #{pane_dead_status}'
```

Expected: pane is dead from the normal interrupt. Do not resume until Step 5 completes.

- [ ] **Step 5: Preserve read-only pipeline-state evidence before resume**

```bash
mkdir --mode=0755 /home/oycy/MMDD/evidence_wdc_gate100_eta_advisory_20260719
cp --preserve=mode,timestamps /home/oycy/MMDD/work_wdc_gate100_eta_advisory_20260719/progress.json /home/oycy/MMDD/evidence_wdc_gate100_eta_advisory_20260719/progress-before-resume.json
chmod 0444 /home/oycy/MMDD/evidence_wdc_gate100_eta_advisory_20260719/progress-before-resume.json
sha256sum /home/oycy/MMDD/work_wdc_gate100_eta_advisory_20260719/progress.json /home/oycy/MMDD/evidence_wdc_gate100_eta_advisory_20260719/progress-before-resume.json
cmp --silent /home/oycy/MMDD/work_wdc_gate100_eta_advisory_20260719/progress.json /home/oycy/MMDD/evidence_wdc_gate100_eta_advisory_20260719/progress-before-resume.json
stat -c '%a %n' /home/oycy/MMDD/evidence_wdc_gate100_eta_advisory_20260719/progress-before-resume.json
```

Expected: identical hashes, `cmp` exit zero, evidence mode `444`. `cp` reads the original and never rewrites it.

- [ ] **Step 6: Resume with the identical command plus `--resume`**

```bash
tmux respawn-pane -k -t wdc_gate100_eta_advisory_20260719:run "/usr/bin/time -v conda run -n MMDD --no-capture-output python scripts/run_mm_joinability_dynamic_vllm.py --input_dir /home/oycy/MMDD/gate_inputs/wdc_100_lowest --output_dir /home/oycy/MMDD/output_wdc_gate100_eta_advisory_20260719 --work_dir /home/oycy/MMDD/work_wdc_gate100_eta_advisory_20260719 --cache_dir /home/oycy/MMDD/cache/wdc_gate100_eta_advisory_20260719 --text_model_path /home/oycy/MMDD/hf_models/Qwen3.5-9B --image_model_path /home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking --run_fingerprint gate100-eta-advisory-20260719 --runtime_dir /home/oycy/MMDD/work_wdc_gate100_eta_advisory_20260719/runtime --max_source_tables 100 --selection_seed 13 --web_max_retries 0 --web_max_response_seconds 8 --web_global_concurrency 128 --web_per_host_concurrency 2 --max_image_attempts_per_entity 3 --max_images_per_entity 3 --min_free_disk_bytes 107374182400 --progress_interval_seconds 5 --resume"
```

Expected: exit zero and complete dataset; record initial/resume elapsed time and maximum RSS from direct pane output.

- [ ] **Step 7: Run deterministic validator and hard evidence checks**

```bash
conda run -n MMDD python scripts/validate_wdc_eta_gate.py --progress /home/oycy/MMDD/work_wdc_gate100_eta_advisory_20260719/progress.json --before-resume /home/oycy/MMDD/evidence_wdc_gate100_eta_advisory_20260719/progress-before-resume.json
jq -e '.counters.selected_tables == 100 and .counters.validated_tables == 100 and .counters.source_tables == 100 and .counters.page_duplicate_physical_requests == 0 and .counters.page_terminal_replays == 0 and .counters.page_unfinished_transport_attempts == 0 and .counters.page_blocked_durable_replays == 0 and .counters.image_duplicate_physical_requests == 0 and .counters.image_terminal_replays == 0 and .counters.image_unfinished_transport_attempts == 0 and .counters.image_blocked_durable_replays == 0 and ([.disk.roots[] | .min_free_bytes >= 107374182400] | all)' /home/oycy/MMDD/work_wdc_gate100_eta_advisory_20260719/progress.json
sha256sum /home/oycy/MMDD/work_wdc_gate100_eta_advisory_20260719/progress.json /home/oycy/MMDD/output_wdc_gate100_eta_advisory_20260719/dataset_manifest.json
```

Expected: validator exit zero with both prefix matches and full advisory lists; hard `jq` check true; all manifests/registries/shards through final materialization validate during pipeline completion. Verify selected/validated/source-table counts are exactly 100 and reconcile every applicable artifact path/count/byte/SHA in the report.

- [ ] **Step 8: Decide, report, and stop on hard failure**

If any hard check fails, write exact `FAIL`/`BLOCKED` evidence to the report, commit it, and do not run Task 4. If all hard checks pass, mark the 100 gate hard `PASS`, preserve ETA only as advisory, and commit:

```bash
git add docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md
git commit -m "Record fresh 100-table advisory gate"
```

### Task 4: Fresh 1,000-Table Advisory Gate

**Files:**
- Modify after the run: `docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md`
- Evidence only: `/home/oycy/MMDD/evidence_wdc_gate1000_eta_advisory_20260719/progress-before-resume.json`

- [ ] **Step 1: Confirm the 100 hard PASS and prove fresh 1K prerequisites**

Do not proceed unless the authoritative report contains a hard `PASS` for Task 3. Run each command separately:

```bash
test ! -e /home/oycy/MMDD/output_wdc_gate1000_eta_advisory_20260719
test ! -e /home/oycy/MMDD/work_wdc_gate1000_eta_advisory_20260719
test ! -e /home/oycy/MMDD/cache/wdc_gate1000_eta_advisory_20260719
test ! -e /home/oycy/MMDD/evidence_wdc_gate1000_eta_advisory_20260719
test -d /home/oycy/MMDD/hf_models/Qwen3.5-9B
test -d /home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking
printf '%s\n' 'text_model_path=/home/oycy/MMDD/hf_models/Qwen3.5-9B served_name=Qwen3.5-9B' 'image_model_path=/home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking served_name=Qwen3-VL-8B-Thinking'
for model in /home/oycy/MMDD/hf_models/Qwen3.5-9B /home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking; do for relative in .hfd/repo_metadata.json config.json model.safetensors.index.json; do if test -f "$model/$relative"; then sha256sum "$model/$relative"; else printf 'NOT PRESENT  %s\n' "$model/$relative"; fi; done; done
sha256sum /home/oycy/MMDD/gate_inputs/wdc_1000_lowest/scale_gate_manifest.jsonl /home/oycy/MMDD/gate_inputs/wdc_1000_lowest/scale_gate_checksums.json
git rev-parse HEAD
```

Expected hashes: `d857ebe0e0f825b4696d5644bc46920d8a5f106803eb16b6b5c320c2f31071f2` and `ec0532b74f97fb7099e8c5e153e907df221f2b213d5734a1fea2d2e677cb9c7c`.

- [ ] **Step 2: Run read-only 1K dry-run preflight**

```bash
conda run -n MMDD python scripts/build_wdc200k_mm_joinability_dataset.py --input_dir /home/oycy/MMDD/gate_inputs/wdc_1000_lowest --output_dir /home/oycy/MMDD/output_wdc_gate1000_eta_advisory_20260719 --work_dir /home/oycy/MMDD/work_wdc_gate1000_eta_advisory_20260719 --cache_dir /home/oycy/MMDD/cache/wdc_gate1000_eta_advisory_20260719 --runtime_dir /home/oycy/MMDD/work_wdc_gate1000_eta_advisory_20260719/runtime --max_source_tables 1000 --selection_seed 13 --web_max_retries 0 --web_max_response_seconds 8 --web_global_concurrency 128 --web_per_host_concurrency 2 --max_image_attempts_per_entity 3 --max_images_per_entity 3 --min_free_disk_bytes 107374182400 --progress_interval_seconds 5 --dry_run --no-resume
```

Expected: exit zero without creating output/work/cache/evidence roots.

- [ ] **Step 3: Launch 1K initial process with direct stdout**

```bash
tmux new-session -d -s wdc_gate1000_eta_advisory_20260719 -n run -c /home/oycy/MMDD/.worktrees/wdc-200k "sleep 3600"
tmux set-option -t wdc_gate1000_eta_advisory_20260719 remain-on-exit on
tmux respawn-pane -k -t wdc_gate1000_eta_advisory_20260719:run "/usr/bin/time -v conda run -n MMDD --no-capture-output python scripts/run_mm_joinability_dynamic_vllm.py --input_dir /home/oycy/MMDD/gate_inputs/wdc_1000_lowest --output_dir /home/oycy/MMDD/output_wdc_gate1000_eta_advisory_20260719 --work_dir /home/oycy/MMDD/work_wdc_gate1000_eta_advisory_20260719 --cache_dir /home/oycy/MMDD/cache/wdc_gate1000_eta_advisory_20260719 --text_model_path /home/oycy/MMDD/hf_models/Qwen3.5-9B --image_model_path /home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking --run_fingerprint gate1000-eta-advisory-20260719 --runtime_dir /home/oycy/MMDD/work_wdc_gate1000_eta_advisory_20260719/runtime --max_source_tables 1000 --selection_seed 13 --web_max_retries 0 --web_max_response_seconds 8 --web_global_concurrency 128 --web_per_host_concurrency 2 --max_image_attempts_per_entity 3 --max_images_per_entity 3 --min_free_disk_bytes 107374182400 --progress_interval_seconds 5 --no-resume"
tmux new-window -d -t wdc_gate1000_eta_advisory_20260719 -n progress -c /home/oycy/MMDD/.worktrees/wdc-200k "watch -n 5 cat /home/oycy/MMDD/work_wdc_gate1000_eta_advisory_20260719/progress.json"
```

- [ ] **Step 4: Interrupt during native-v2 images and preserve evidence**

After pages are complete and images have at least one nonbaseline native-v2 sample, record counters, send `Ctrl-C`, wait for a dead pane, then run:

```bash
tmux send-keys -t wdc_gate1000_eta_advisory_20260719:run C-c
tmux display-message -p -t wdc_gate1000_eta_advisory_20260719:run '#{pane_dead} #{pane_dead_status}'
mkdir --mode=0755 /home/oycy/MMDD/evidence_wdc_gate1000_eta_advisory_20260719
cp --preserve=mode,timestamps /home/oycy/MMDD/work_wdc_gate1000_eta_advisory_20260719/progress.json /home/oycy/MMDD/evidence_wdc_gate1000_eta_advisory_20260719/progress-before-resume.json
chmod 0444 /home/oycy/MMDD/evidence_wdc_gate1000_eta_advisory_20260719/progress-before-resume.json
sha256sum /home/oycy/MMDD/work_wdc_gate1000_eta_advisory_20260719/progress.json /home/oycy/MMDD/evidence_wdc_gate1000_eta_advisory_20260719/progress-before-resume.json
cmp --silent /home/oycy/MMDD/work_wdc_gate1000_eta_advisory_20260719/progress.json /home/oycy/MMDD/evidence_wdc_gate1000_eta_advisory_20260719/progress-before-resume.json
stat -c '%a %n' /home/oycy/MMDD/evidence_wdc_gate1000_eta_advisory_20260719/progress-before-resume.json
```

Expected: the two SHA values match, `cmp` exits zero, and mode is `444`. Never alter the original `progress.json`.

- [ ] **Step 5: Resume the exact 1K command**

```bash
tmux respawn-pane -k -t wdc_gate1000_eta_advisory_20260719:run "/usr/bin/time -v conda run -n MMDD --no-capture-output python scripts/run_mm_joinability_dynamic_vllm.py --input_dir /home/oycy/MMDD/gate_inputs/wdc_1000_lowest --output_dir /home/oycy/MMDD/output_wdc_gate1000_eta_advisory_20260719 --work_dir /home/oycy/MMDD/work_wdc_gate1000_eta_advisory_20260719 --cache_dir /home/oycy/MMDD/cache/wdc_gate1000_eta_advisory_20260719 --text_model_path /home/oycy/MMDD/hf_models/Qwen3.5-9B --image_model_path /home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking --run_fingerprint gate1000-eta-advisory-20260719 --runtime_dir /home/oycy/MMDD/work_wdc_gate1000_eta_advisory_20260719/runtime --max_source_tables 1000 --selection_seed 13 --web_max_retries 0 --web_max_response_seconds 8 --web_global_concurrency 128 --web_per_host_concurrency 2 --max_image_attempts_per_entity 3 --max_images_per_entity 3 --min_free_disk_bytes 107374182400 --progress_interval_seconds 5 --resume"
```

Expected: completion exit zero; record both process durations/RSS and calculate peak-RSS growth against the 100 gate with units and formula.

- [ ] **Step 6: Validate 1K advisory and hard evidence**

```bash
conda run -n MMDD python scripts/validate_wdc_eta_gate.py --progress /home/oycy/MMDD/work_wdc_gate1000_eta_advisory_20260719/progress.json --before-resume /home/oycy/MMDD/evidence_wdc_gate1000_eta_advisory_20260719/progress-before-resume.json
jq -e '.counters.selected_tables == 1000 and .counters.validated_tables == 1000 and .counters.source_tables == 1000 and .counters.page_duplicate_physical_requests == 0 and .counters.page_terminal_replays == 0 and .counters.page_unfinished_transport_attempts == 0 and .counters.page_blocked_durable_replays == 0 and .counters.image_duplicate_physical_requests == 0 and .counters.image_terminal_replays == 0 and .counters.image_unfinished_transport_attempts == 0 and .counters.image_blocked_durable_replays == 0 and ([.disk.roots[] | .min_free_bytes >= 107374182400] | all)' /home/oycy/MMDD/work_wdc_gate1000_eta_advisory_20260719/progress.json
sha256sum /home/oycy/MMDD/work_wdc_gate1000_eta_advisory_20260719/progress.json /home/oycy/MMDD/output_wdc_gate1000_eta_advisory_20260719/dataset_manifest.json
```

Expected: validator/prefix checks pass; anomaly counters are zero; reserve observations pass; every final-stage manifest/registry/shard reference validates. Record separate output/work/cache growth and exact validator JSON.

- [ ] **Step 7: Decide, report, commit, and gate 10K**

On any hard failure, record `FAIL`/`BLOCKED`, commit the evidence, and stop. Only a hard `PASS` authorizes Task 5.

```bash
git add docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md
git commit -m "Record fresh 1000-table advisory gate"
```

### Task 5: Ordered 10,000-Table Structural Gate

**Files:**
- Create: `scripts/validate_wdc_structural_gate.py`
- Create: `tests/test_wdc_structural_gate_validator.py`
- Modify after the run: `docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md`

The validator defines `StructuralGateConfig(input_dir: Path, output_dir: Path,
work_dir: Path, cache_dir: Path, expected_tables: int, reserve_bytes: int)`,
`StructuralGateValidationError(code: str, message: str)`,
`validate_structural_gate(config: StructuralGateConfig) -> dict[str, Any]`,
`build_parser() -> argparse.ArgumentParser`, and
`main(argv: Sequence[str] | None = None) -> int`. It accepts required
`--input-dir`, `--output-dir`, `--work-dir`, `--cache-dir`,
`--expected-tables`, and `--reserve-bytes`; it writes one compact, key-sorted
JSON object to stdout and never writes any input or gate root.

Success JSON contains exact input archive paths/SHA values; selected,
validated, and source-table counts; per-table selected/validated/source row
counts and reconciliation result; applicable artifact allowlist and validated
path/count/bytes/SHA/input/upstream evidence; extra-file list; progress/disk
evidence; and projections for page, image, network, output, work, and cache.
Errors use stable stdout-only JSON and nonzero exit. Missing, unreadable,
malformed, inconsistent, checksum-invalid, unexpected, or insufficient-disk
evidence is a hard error and therefore leaves the gate `BLOCKED`.

- [ ] **Step 1: Write RED exhaustive row and registry tests**

Create a three-table structural fixture through existing selection/structural
helpers. Assert exact map equality across all records, without sampling:
`selected_tables.jsonl[relative_path].rows`,
`validated-selected-tables.jsonl[relative_path].rows`, and
`source_tables/*.jsonl[source_file].num_rows == len(rows)`. Assert unique and
identical relative-path sets, unique source-table IDs, exactly
`expected_tables` records at all three authorities, and absence of the source
`image` column. Mutate the middle table only and expect `ROW_RECONCILIATION_FAILED`.

Also assert the read-only validator reconstructs the exact gate
`PipelineConfig`, calls existing `_statistics_archives`, `_input_identity`,
`_validate_stage_registry` for `selection`, then `_registry_identity` and
`_validate_stage_registry` for `structural`. This reuses current checks for
registry completeness, config/input/upstream identities, producer-manifest
SHA, and every declared shard's confined path, records, bytes, and SHA.

- [ ] **Step 2: Run row/registry tests RED**

Run: `conda run -n MMDD python -m pytest tests/test_wdc_structural_gate_validator.py -k 'row or registry or identity' -q`

Expected: collection FAIL because `validate_wdc_structural_gate` does not exist.

- [ ] **Step 3: Implement strict read-only restore and exhaustive reconciliation**

Build the exact 10K `PipelineConfig` from CLI inputs with seed `13`, retries
`0`, response seconds `8`, global/per-host concurrency `128/2`, image attempts
and retention `3/3`, and `stop_after="structural"`. Invoke the existing
read-only registry validators, then stream all selected, validated, and source
JSONL records into SQLite tables inside `TemporaryDirectory`. Use indexed joins
to detect missing/extra paths or any row mismatch while keeping memory bounded.
Hash every statistics archive and every validated artifact from exact bytes.
Never create or change a file beneath input/output/work/cache.

- [ ] **Step 4: Run row/registry tests GREEN**

Run the Step 2 command again.

Expected: selected tests PASS.

- [ ] **Step 5: Write RED artifact allowlist and extra-detection tests**

The applicable regular-file allowlist is exactly the two pipeline registries,
their referenced producer manifests, every completed shard referenced by
those manifests, `selection/reserve.sqlite3`,
`structural/structural-counts.sqlite3`, and `progress.json`. Enumerate every
regular file under `selection/`, `structural/`, and `stage_manifests/`; reject
anything not in the derived allowlist as `EXTRA_ARTIFACT`. Tests add an
undeclared JSONL, remove a declared shard, change bytes without updating SHA,
forge count/bytes, escape a manifest path outside its stage root, and alter
input/upstream identity. Each exact corruption must fail; downstream network,
model, and final-dataset paths are outside the structural allowlist and appear
as `NOT_APPLICABLE`, not extra.

- [ ] **Step 6: Write RED projection and JSON CLI tests**

Recompute and cross-check:

```text
page_bytes = unique_page_urls * web_max_page_bytes
image_bytes = entities * max_image_attempts_per_entity * web_max_image_bytes
network_bytes = page_bytes + image_bytes
output_additional_bytes = structural_output_bytes + estimated_next_stage_bytes
work_additional_bytes = estimated_next_stage_bytes
cache_additional_bytes = network_bytes
projected_<root>_bytes = progress.disk.roots.<root>.current_bytes + <root>_additional_bytes
device_<root> = Path(<root>).stat().st_dev
live_free_<root> = shutil.disk_usage(Path(<root>)).free
diagnostic_fits_<root> = <root>_additional_bytes <= live_free_<root> - reserve_bytes
device_additional_bytes[device] = sum(<root>_additional_bytes for roots on device)
device_free_bytes[device] = the one identical live_free value for roots on device
device_available_bytes[device] = device_free_bytes[device] - reserve_bytes
fits_device[device] = device_additional_bytes[device] <= device_available_bytes[device]
hard_disk_projection_pass = all(fits_device.values())
```

Require persisted page/image/network counters to equal recomputation, all
progress-root minimum observed free bytes to meet reserve, and every device
available value to be nonnegative. Keep per-root current/projected/additional,
device, live-free, and diagnostic-fit fields in JSON, but never use the three
diagnostic fits independently as the hard decision. Group roots by exact
`Path(root).stat().st_dev`; require all live-free observations for roots in one
device group to be identical or fail `INCONSISTENT_DEVICE_FREE_BYTES`; count
that common free value and `reserve_bytes` once for the group; sum all root
additional bytes in the group; and require every `fits_device` to be true.

Add these two explicit RED tests using monkeypatched `Path.stat` device IDs and
`shutil.disk_usage` free values:

```python
def test_shared_device_individual_fit_but_combined_demand_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_devices(monkeypatch, output=7, work=7, cache=7)
    _patch_free(monkeypatch, output=200, work=200, cache=200)
    result = _validate_projection(output_additional=60, work_additional=60, cache_additional=0, reserve=100)
    assert result["roots"]["output"]["diagnostic_fits"] is True
    assert result["roots"]["work"]["diagnostic_fits"] is True
    assert result["devices"]["7"]["additional_bytes"] == 120
    assert result["devices"]["7"]["available_bytes"] == 100
    assert result["devices"]["7"]["fits"] is False
    assert result["hard_disk_projection_pass"] is False


def test_different_devices_each_fit_and_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_devices(monkeypatch, output=7, work=8, cache=9)
    _patch_free(monkeypatch, output=200, work=200, cache=100)
    result = _validate_projection(output_additional=60, work_additional=60, cache_additional=0, reserve=100)
    assert result["devices"]["7"]["fits"] is True
    assert result["devices"]["8"]["fits"] is True
    assert result["devices"]["9"]["fits"] is True
    assert result["hard_disk_projection_pass"] is True
```

Other tests exercise exact equality, one-byte-short device headroom, negative
device headroom, inconsistent same-device free values, inconsistent counters,
stable success JSON, stdout-only machine errors, CLI usage errors, and input
bytes/mtime/mode immutability.

Run: `conda run -n MMDD python -m pytest tests/test_wdc_structural_gate_validator.py -k 'shared_device_individual_fit or different_devices_each_fit' -q`

Expected: both tests FAIL because device aggregation is not implemented; the
shared-device case specifically exposes the false per-root pass.

- [ ] **Step 7: Implement artifact/projection JSON and run all tests GREEN**

Add deterministic sorted artifact/row failure records, per-root diagnostics,
device groups sorted by numeric device ID, and the exact device hard-decision
fields above. Use a custom parser error path so CLI errors are JSON on stdout
with empty stderr. Run:

```bash
conda run -n MMDD python -m pytest tests/test_wdc_structural_gate_validator.py -q
conda run -n MMDD python scripts/validate_wdc_structural_gate.py --help
```

Expected: all focused tests PASS and help exposes only the six fixed options.

- [ ] **Step 8: Commit the structural validator**

```bash
git add scripts/validate_wdc_structural_gate.py tests/test_wdc_structural_gate_validator.py
git commit -m "Add read-only WDC structural gate validator"
```

- [ ] **Step 9: Require 1K hard PASS and prove new structural roots absent**

Do not proceed unless Task 4 is a hard `PASS`. Run:

```bash
test ! -e /home/oycy/MMDD/output_wdc_gate10000_structural_20260719
test ! -e /home/oycy/MMDD/work_wdc_gate10000_structural_20260719
test ! -e /home/oycy/MMDD/cache/wdc_gate10000_structural_20260719
test ! -e /home/oycy/MMDD/evidence_wdc_gate10000_structural_20260719
find /home/oycy/MMDD/wdc_schemaorg_2023 -maxdepth 2 -type f -name '*_statistics.zip' -print0
git rev-parse HEAD
```

Expected: roots are absent and exactly 42 statistics archives are listed. Record SHA-256 for each archive with `find /home/oycy/MMDD/wdc_schemaorg_2023 -maxdepth 2 -type f -name '*_statistics.zip' -print0 | sort -z | xargs -0 sha256sum` directly in the report; this output is evidence, not a redirected log.

- [ ] **Step 10: Run structural dry-run**

```bash
conda run -n MMDD python scripts/build_wdc200k_mm_joinability_dataset.py --input_dir /home/oycy/MMDD/wdc_schemaorg_2023 --output_dir /home/oycy/MMDD/output_wdc_gate10000_structural_20260719 --work_dir /home/oycy/MMDD/work_wdc_gate10000_structural_20260719 --cache_dir /home/oycy/MMDD/cache/wdc_gate10000_structural_20260719 --max_source_tables 10000 --selection_seed 13 --web_max_retries 0 --web_max_response_seconds 8 --web_global_concurrency 128 --web_per_host_concurrency 2 --max_image_attempts_per_entity 3 --max_images_per_entity 3 --min_free_disk_bytes 107374182400 --progress_interval_seconds 5 --stop_after structural --dry_run --no-resume
```

Expected: exit zero and roots remain absent.

- [ ] **Step 11: Launch structural-only tmux run**

```bash
tmux new-session -d -s wdc_gate10000_structural_20260719 -n run -c /home/oycy/MMDD/.worktrees/wdc-200k "sleep 3600"
tmux set-option -t wdc_gate10000_structural_20260719 remain-on-exit on
tmux respawn-pane -k -t wdc_gate10000_structural_20260719:run "/usr/bin/time -v conda run -n MMDD --no-capture-output python scripts/build_wdc200k_mm_joinability_dataset.py --input_dir /home/oycy/MMDD/wdc_schemaorg_2023 --output_dir /home/oycy/MMDD/output_wdc_gate10000_structural_20260719 --work_dir /home/oycy/MMDD/work_wdc_gate10000_structural_20260719 --cache_dir /home/oycy/MMDD/cache/wdc_gate10000_structural_20260719 --max_source_tables 10000 --selection_seed 13 --web_max_retries 0 --web_max_response_seconds 8 --web_global_concurrency 128 --web_per_host_concurrency 2 --max_image_attempts_per_entity 3 --max_images_per_entity 3 --min_free_disk_bytes 107374182400 --progress_interval_seconds 5 --stop_after structural --no-resume"
tmux new-window -d -t wdc_gate10000_structural_20260719 -n progress -c /home/oycy/MMDD/.worktrees/wdc-200k "watch -n 5 cat /home/oycy/MMDD/work_wdc_gate10000_structural_20260719/progress.json"
```

Expected: no URL request, model, or GPU work; exit zero at structural boundary.

- [ ] **Step 12: Validate structural hard evidence and disk projection**

```bash
conda run -n MMDD python scripts/validate_wdc_structural_gate.py --input-dir /home/oycy/MMDD/wdc_schemaorg_2023 --output-dir /home/oycy/MMDD/output_wdc_gate10000_structural_20260719 --work-dir /home/oycy/MMDD/work_wdc_gate10000_structural_20260719 --cache-dir /home/oycy/MMDD/cache/wdc_gate10000_structural_20260719 --expected-tables 10000 --reserve-bytes 107374182400
```

Expected: exit zero with 10,000 exact selected/validated/source tables, every
table's full row count reconciled, complete stage-boundary artifact evidence,
empty extra list, exact page/image/network/output/work/cache projections, and
per-root diagnostics plus device-group evidence. Every device must show
`sum(root additional_bytes) <= one live_free_bytes - one reserve_bytes`; all
device fits must be true.
Record complete validator JSON. Mark downstream network/model/final artifacts
and URL ETA `NOT APPLICABLE`.

- [ ] **Step 13: Decide, report, commit, and gate formal launch**

On hard failure, record and stop. Only a hard `PASS` plus affirmative network disk projection authorizes Task 6.

```bash
git add docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md
git commit -m "Record 10000-table structural gate"
```

### Task 6: Final Verification, Reviews, Report Commit, And Formal 200K Launch

**Files:**
- Verify: `scripts/validate_wdc_eta_gate.py`
- Verify: `tests/test_wdc_eta_gate_validator.py`
- Verify: `scripts/validate_wdc_structural_gate.py`
- Verify: `tests/test_wdc_structural_gate_validator.py`
- Verify and modify: `docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md`

- [ ] **Step 1: Complete policy/gate results and create the launch commit**

Confirm the report contains the final advisory policy, historical `BLOCKED`
classification, exact hard `PASS` evidence for 100/1K/10K, commands, hashes,
validator results, and formal-launch prerequisites. Commit all pre-launch code,
tests, and report changes once, then record the resulting HEAD as the launch
candidate:

```bash
git diff --check
git add scripts/validate_wdc_eta_gate.py scripts/validate_wdc_structural_gate.py tests/test_wdc_eta_gate_validator.py tests/test_wdc_structural_gate_validator.py docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md
git commit -m "Authorize formal WDC 200K launch"
git rev-parse HEAD
```

Expected: one exact launch-candidate commit containing completed policy and all
three gate results.

- [ ] **Step 2: Run focused and full verification at that exact commit**

Run each command with fresh output:

```bash
conda run -n MMDD python -m pytest tests/test_wdc_eta_gate_validator.py -q
conda run -n MMDD python -m pytest tests/test_wdc_structural_gate_validator.py -q
conda run -n MMDD python -m pytest tests/test_wdc200k_io.py tests/test_wdc200k_selection.py tests/test_wdc200k_structural.py tests/test_wdc200k_fetch.py tests/test_wdc200k_assets.py tests/test_wdc200k_models.py tests/test_wdc200k_materialize.py tests/test_wdc200k_pipeline.py tests/test_wdc200k_scale_gate_input.py tests/test_wdc_mm_joinability_dataset.py tests/test_mm_joinability_extraction.py tests/test_stage1_pipeline.py -q
conda run -n MMDD python -m py_compile scripts/validate_wdc_eta_gate.py scripts/validate_wdc_structural_gate.py scripts/wdc200k_io.py scripts/wdc200k_selection.py scripts/wdc200k_structural.py scripts/wdc200k_fetch.py scripts/wdc200k_assets.py scripts/wdc200k_models.py scripts/wdc200k_materialize.py scripts/build_wdc200k_mm_joinability_dataset.py scripts/create_wdc200k_scale_gate_input.py scripts/run_mm_joinability_dynamic_vllm.py
git diff --check
git status --short
git rev-parse HEAD
```

Expected: every pytest and compilation command exits zero; both Git checks are
clean; HEAD is the launch-candidate commit from Step 1. Any failure blocks
launch.

- [ ] **Step 3: Review the exact launch candidate and restart on any change**

Use `superpowers:requesting-code-review` for a spec-compliance review and a
code-quality review of the exact Step 2 HEAD. If either review requires any
code, test, or report change, implement it with TDD, create a new commit, and
return to Task 6 Step 2 so the new HEAD receives full verification and both
reviews. After both reviews accept the exact HEAD, do not edit or commit
anything before the tmux launch.

- [ ] **Step 4: Prove formal paths absent and record full-corpus/model identity**

```bash
test ! -e /home/oycy/MMDD/output_wdc_200k_eta_advisory_20260719
test ! -e /home/oycy/MMDD/work_wdc_200k_eta_advisory_20260719
test ! -e /home/oycy/MMDD/cache/wdc_200k_eta_advisory_20260719
test ! -e /home/oycy/MMDD/evidence_wdc_200k_eta_advisory_20260719
test -d /home/oycy/MMDD/hf_models/Qwen3.5-9B
test -d /home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking
printf '%s\n' 'text_model_path=/home/oycy/MMDD/hf_models/Qwen3.5-9B served_name=Qwen3.5-9B' 'image_model_path=/home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking served_name=Qwen3-VL-8B-Thinking'
for model in /home/oycy/MMDD/hf_models/Qwen3.5-9B /home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking; do for relative in .hfd/repo_metadata.json config.json model.safetensors.index.json; do if test -f "$model/$relative"; then sha256sum "$model/$relative"; else printf 'NOT PRESENT  %s\n' "$model/$relative"; fi; done; done
find /home/oycy/MMDD/wdc_schemaorg_2023 -maxdepth 2 -type f -name '*_statistics.zip' -print0 | sort -z | xargs -0 sha256sum
git rev-parse HEAD
date --utc --iso-8601=seconds
```

Expected: all formal roots absent, 42 archive hashes visible, and commit equals
the reviewed launch commit. Record each model's absolute path, explicit served
name, and SHA-256 for `.hfd/repo_metadata.json`, `config.json`, and
`model.safetensors.index.json`; if any named metadata file is absent, record
that absolute path as `NOT PRESENT` rather than omitting it.

- [ ] **Step 5: Run formal read-only dry-run**

```bash
conda run -n MMDD python scripts/build_wdc200k_mm_joinability_dataset.py --input_dir /home/oycy/MMDD/wdc_schemaorg_2023 --output_dir /home/oycy/MMDD/output_wdc_200k_eta_advisory_20260719 --work_dir /home/oycy/MMDD/work_wdc_200k_eta_advisory_20260719 --cache_dir /home/oycy/MMDD/cache/wdc_200k_eta_advisory_20260719 --runtime_dir /home/oycy/MMDD/work_wdc_200k_eta_advisory_20260719/runtime --max_source_tables 200000 --selection_seed 13 --web_max_retries 0 --web_max_response_seconds 8 --web_global_concurrency 128 --web_per_host_concurrency 2 --max_image_attempts_per_entity 3 --max_images_per_entity 3 --min_free_disk_bytes 107374182400 --progress_interval_seconds 5 --dry_run --no-resume
```

Expected: exit zero; formal roots remain absent.

- [ ] **Step 6: Launch formal 200K in tmux with direct stdout**

```bash
tmux new-session -d -s wdc_200k_eta_advisory_20260719 -n run -c /home/oycy/MMDD/.worktrees/wdc-200k "sleep 3600"
tmux set-option -t wdc_200k_eta_advisory_20260719 remain-on-exit on
tmux respawn-pane -k -t wdc_200k_eta_advisory_20260719:run "/usr/bin/time -v conda run -n MMDD --no-capture-output python scripts/run_mm_joinability_dynamic_vllm.py --input_dir /home/oycy/MMDD/wdc_schemaorg_2023 --output_dir /home/oycy/MMDD/output_wdc_200k_eta_advisory_20260719 --work_dir /home/oycy/MMDD/work_wdc_200k_eta_advisory_20260719 --cache_dir /home/oycy/MMDD/cache/wdc_200k_eta_advisory_20260719 --text_model_path /home/oycy/MMDD/hf_models/Qwen3.5-9B --image_model_path /home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking --run_fingerprint wdc-200k-eta-advisory-20260719 --runtime_dir /home/oycy/MMDD/work_wdc_200k_eta_advisory_20260719/runtime --max_source_tables 200000 --selection_seed 13 --web_max_retries 0 --web_max_response_seconds 8 --web_global_concurrency 128 --web_per_host_concurrency 2 --max_image_attempts_per_entity 3 --max_images_per_entity 3 --min_free_disk_bytes 107374182400 --progress_interval_seconds 5 --no-resume"
tmux new-window -d -t wdc_200k_eta_advisory_20260719 -n progress -c /home/oycy/MMDD/.worktrees/wdc-200k "watch -n 5 cat /home/oycy/MMDD/work_wdc_200k_eta_advisory_20260719/progress.json"
date --utc --iso-8601=seconds
```

Expected: formal runner stays live in `wdc_200k_eta_advisory_20260719:run`, direct stdout is visible, and progress watcher is read-only. Put the exact launch timestamp, commit, command, roots, model identity, session/window names, and first progress SHA in the user/future-session handoff record, not in a post-review report edit.

- [ ] **Step 7: Observe one bounded initial stability window**

For 15 minutes after launch, inspect the read-only progress window at least
once per minute while its existing `watch -n 5` refresh continues. Confirm the
runner pane remains live, guard checks succeed, progress advances, minimum
observed free bytes remain at least `107374182400`, URL telemetry is native v2
when network work begins, and no duplicate/replay/unfinished/blocked counter
becomes nonzero. ETA factors remain advisory.

The pipeline's own guards and validators remain the continuous authority: any
hard integrity, correctness, artifact, replay, or reserve failure must fail
closed and exit the runner. This plan does not promise indefinite manual agent
monitoring beyond the initial 15-minute window.

- [ ] **Step 8: Stop and preserve evidence if the initial window exposes a hard failure**

If the runner has not already exited fail-closed, send a normal interrupt,
wait until the pane is dead, and preserve the last pipeline state read-only:

```bash
tmux send-keys -t wdc_200k_eta_advisory_20260719:run C-c
tmux display-message -p -t wdc_200k_eta_advisory_20260719:run '#{pane_dead} #{pane_dead_status}'
mkdir --mode=0755 /home/oycy/MMDD/evidence_wdc_200k_eta_advisory_20260719
cp --preserve=mode,timestamps /home/oycy/MMDD/work_wdc_200k_eta_advisory_20260719/progress.json /home/oycy/MMDD/evidence_wdc_200k_eta_advisory_20260719/progress-hard-failure.json
chmod 0444 /home/oycy/MMDD/evidence_wdc_200k_eta_advisory_20260719/progress-hard-failure.json
sha256sum /home/oycy/MMDD/work_wdc_200k_eta_advisory_20260719/progress.json /home/oycy/MMDD/evidence_wdc_200k_eta_advisory_20260719/progress-hard-failure.json
```

Expected: the pane is dead and evidence hashes match. Acceptance remains
`NOT_RUN_OR_INCOMPLETE`/`BLOCKED`; do not relaunch automatically.

- [ ] **Step 9: Hand off a healthy live run without another commit**

Provide the user or next session this exact handoff record:

```text
Attach: tmux attach-session -t wdc_200k_eta_advisory_20260719
Runner pane: wdc_200k_eta_advisory_20260719:run
Progress pane: wdc_200k_eta_advisory_20260719:progress
Pane status: tmux display-message -p -t wdc_200k_eta_advisory_20260719:run '#{pane_dead} #{pane_dead_status} #{pane_current_command}'
Progress refresh: 5 seconds
Recommended human/future-session check: once per minute or after any pane exit
Launch commit: exact reviewed HEAD from Task 6 Steps 1-3
Acceptance: NOT_RUN_OR_INCOMPLETE until formal completion and validators pass
```

Do not edit or commit the report after review and before launch. Do not commit
launch-time observations. A future completion session must run the applicable
hard validators and ETA validator, then update the report from
`NOT_RUN_OR_INCOMPLETE`; a healthy live process is never itself an acceptance
`PASS`.

---

## Plan Self-Review Results

- Spec coverage: Task 1 covers strict restore, exact SHA, native v2, all final-half records, ordering, worst/zero cases, deterministic JSON, errors, prefix equality, and no input writes. Task 2 replaces the three obsolete decision rules and records historical evidence only at its true strength. Tasks 3-6 enforce fresh absent roots, 100 GiB, complete applicable artifacts, preserved interruption evidence, ordered hard passes, direct tmux stdout, and fail-stop launch authority.
- TDD and commits: every validator behavior begins RED, receives minimal implementation, runs GREEN, mutation-checks immutability, and commits in bounded units. Report and each completed gate receive separate commits.
- Interface consistency: CLI options are exactly `--progress` and optional `--before-resume`; stage names are `pages`/`images`; ordering is factor descending then zero-based position ascending; prefix hashing uses the same canonical serialization in implementation, tests, validator JSON, and report.
- Operational consistency: the initial/resume commands differ only by `--no-resume` versus `--resume`; 1K requires 100 hard PASS; 10K requires 1K hard PASS; formal launch requires all three hard passes plus fresh verification/review. No command deletes an existing root or redirects runner output.
- Deferred-work marker review: the completed plan contains no unspecified implementation steps or copied-forward markers.
- Final document checks: run `git diff --check -- docs/superpowers/plans/2026-07-19-wdc-eta-advisory-gate.md` and inspect `git status --short`; expected whitespace-clean plan and no implementation/report changes during planning.
