# Task 8 Dynamic Runtime Review Fix

Status: COMPLETE; READY FOR REVIEW

## Scope

- Changed the dynamic vLLM runner's default runtime directory from
  `OUTPUT_DIR/_dynamic_vllm` to the output sibling
  `OUTPUT_DIR.parent/.<OUTPUT_DIR.name>.wdc200k-runtime`.
- Preserved an explicit `--runtime_dir` unchanged.
- Added integration-level command assertions for the Task 8 builder's two
  endpoint files and four model markers.
- Kept subprocess output inherited and process-group isolation enabled.

## TDD evidence

The new default-runtime test first failed because the builder received
`output/_dynamic_vllm/text_endpoints.txt`; the explicit-runtime test passed.
After the runner change, both tests passed and all six runtime paths were under
the output sibling directory.

## Verification

```text
Focused red/green tests
2 passed, 57 deselected

Dynamic runner test file
59 passed

Dynamic runner and Task 8 pipeline tests
78 passed

py_compile
PASS

git diff --check
PASS
```
