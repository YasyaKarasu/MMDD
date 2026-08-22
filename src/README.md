# Dataset construction

`src/` contains the standalone research implementation for MMDD dataset
construction. It does not import `scripts_old/` and does not contain training,
evaluation, annotation, GPU scheduling, marker protocols, or service
orchestration.

The two source families share one joinability algorithm in
`mmdd_dataset/joinability.py`:

1. choose a candidate entity column;
2. test whether evidence-backed extraction recovers a hidden attribute;
3. project the visible entity/context columns into a query;
4. project the recovered attribute/context columns into a target;
5. emit the query, target, qrel, recovery path, and table decision.

`build_joinability_for_table()` is the streaming boundary. The compact
EntiTables builder and the WDC backend both call this same function. WDC adds
only a source adapter and a disk-backed execution backend.

## WDC stages

The scalable WDC entry point is `build_wdc_dataset.py`. Its work directory has
exactly five stage directories:

```text
work/
  select_sample/
  normalize/
  fetch_evidence/
  extract/
  materialize/
```

Each directory has one `manifest.json`. A manifest records the parameters,
seed, input fingerprint, output shard record/byte/SHA-256 information, counts,
and completion state. JSONL parts are first written as `.tmp` files and then
atomically renamed. Resume trusts only checksum-valid parts listed in the
manifest.

Run the stages separately:

```bash
conda run -n MMDD python src/build_wdc_dataset.py select_sample \
  --input-dir wdc_schemaorg_2023 \
  --work-dir work_wdc_research \
  --output-dir output_wdc_research \
  --target-tables 200000 --seed 13 --shard-size 100

conda run -n MMDD python src/build_wdc_dataset.py normalize \
  --input-dir wdc_schemaorg_2023 \
  --work-dir work_wdc_research \
  --output-dir output_wdc_research \
  --target-tables 200000 --seed 13 --shard-size 100

conda run -n MMDD python src/build_wdc_dataset.py fetch_evidence \
  --input-dir wdc_schemaorg_2023 \
  --work-dir work_wdc_research \
  --output-dir output_wdc_research \
  --target-tables 200000 --seed 13 --shard-size 100 \
  --concurrency 32 --evidence-kind all

conda run -n MMDD python src/build_wdc_dataset.py extract \
  --input-dir wdc_schemaorg_2023 \
  --work-dir work_wdc_research \
  --output-dir output_wdc_research \
  --target-tables 200000 --seed 13 --shard-size 100 \
  --concurrency 8 \
  --text-model-base-url http://127.0.0.1:8001/v1 \
  --image-model-base-url http://127.0.0.1:8000/v1

conda run -n MMDD python src/build_wdc_dataset.py materialize \
  --input-dir wdc_schemaorg_2023 \
  --work-dir work_wdc_research \
  --output-dir output_wdc_research \
  --target-tables 200000 --seed 13 --shard-size 100
```

All commands that reopen a work directory must use the same stage-relevant
parameters. To run end to end, replace the stage name with `run`. To continue
at the first incomplete stage, use `resume`.

Page fetching can be stopped before images with `--evidence-kind pages`.
Later run the same stage with `--evidence-kind images`. Page/image tasks,
results, and model tasks/results are persistent JSONL shards. Network and model
outcomes are additionally cached in SQLite so an interrupted shard does not
repeat completed external calls. Concurrency submission is bounded to twice
the configured worker count.

### Preserve evidence from a legacy 200K work directory

The old 200K builder stored exported network outcomes under `work_wdc_*`, while
the downloaded image bytes lived in a separate shared cache. Preserve those
two pieces in one self-contained evidence cache before removing an obsolete
work directory. Start with the read-only inspection:

```bash
conda run -n MMDD python src/consolidate_wdc_evidence_cache.py inspect \
  --work-dir work_wdc_200k_eta_advisory_20260719 \
  --cache-dir cache/wdc_200k_evidence_v1
```

Create the cache next. `--image-root` is the old cache root that contains its
`images/` directory. `auto` uses hard links on the same filesystem and falls
back to copying when necessary. Published outcome shards and image records use
only paths relative to the new cache.

```bash
conda run -n MMDD python src/consolidate_wdc_evidence_cache.py consolidate \
  --work-dir work_wdc_200k_eta_advisory_20260719 \
  --image-root cache/wdc_webtable \
  --cache-dir cache/wdc_200k_evidence_v1 \
  --copy-mode auto --min-free-gb 10

conda run -n MMDD python src/consolidate_wdc_evidence_cache.py verify \
  --cache-dir cache/wdc_200k_evidence_v1
```

The new fetch stage can reuse successful URL outcomes from this cache. Legacy
terminal failures are deliberately retried under the current fetch policy.

```bash
conda run -n MMDD python src/build_wdc_dataset.py fetch_evidence \
  --input-dir wdc_schemaorg_2023 \
  --work-dir work_wdc_research \
  --output-dir output_wdc_research \
  --target-tables 200000 --seed 13 --shard-size 100 \
  --concurrency 32 --evidence-kind all \
  --reuse-evidence-cache cache/wdc_200k_evidence_v1
```

Deleting the old work tree is a separate, irreversible command. It repeats the
full shard and image verification and requires the exact directory name. It
does not delete `cache/wdc_webtable`, which may be shared by other runs.

```bash
conda run -n MMDD python src/consolidate_wdc_evidence_cache.py delete-work \
  --work-dir work_wdc_200k_eta_advisory_20260719 \
  --cache-dir cache/wdc_200k_evidence_v1 \
  --confirm-delete-work work_wdc_200k_eta_advisory_20260719
```

Import model results instead of calling endpoints with
`--import-extractions results.jsonl`. Each record may provide `task_id`, or the
fields `source_table_id`, `source_row_id`, `asset_id`, and `attribute_name` from
which the deterministic task ID is reconstructed. Imported and online records
use the same `attribute_extractions` format.

Inspect scale without creating work/output directories or making external
calls:

```bash
conda run -n MMDD python src/build_wdc_dataset.py run \
  --input-dir wdc_schemaorg_2023 \
  --work-dir work_wdc_research \
  --output-dir output_wdc_research \
  --target-tables 200000 --dry-run
```

The estimate reports candidate/selected tables, sampled entities, upper-bound
page/image/model task counts, and approximate disk bytes. Every writing stage
also checks free space against `--min-free-gb` before its large outputs.

## Network safety

Arbitrary WDC evidence URLs are restricted to HTTP and HTTPS, may not contain
credentials, and may not target localhost, `.local` names, loopback,
link-local, private, reserved, or otherwise non-global IP addresses. DNS is
checked before each request, and every redirect target is checked again.

## Final WDC output

The final directory uses `mmdd_joinability_sharded_v2`. It is self-contained:
all manifest paths are relative, image bytes are copied under `images/`, and no
record or manifest refers to `work_*`. The main artifacts retain the existing
names: `source_tables`, `entities`, `bridge_assets`, `table_asset_links`,
`attribute_extractions`, `query_tables`, `data_lake_tables`, `qrels`,
`evidence_recoveries`, and `table_queryability_decisions`.

Compatibility changes from the old WDC builder are intentional:

- the final manifest no longer carries producer registries, certificate
  chains, or absolute upstream paths;
- every potentially large artifact, including `qrels` and decisions, is a
  sharded directory rather than a large single JSONL file;
- `splits.json` is a small summary, while detailed assignments live in the
  sharded `split_assignments` artifact.

Only files listed in `dataset_manifest.json` are authoritative. The helper
`mmdd_dataset.wdc_runtime.iter_dataset_artifact()` resolves them relative to
the final directory and validates their checksums.

## Compact EntiTables builder

The existing compact entry point remains available for small research runs:

```bash
conda run -n MMDD python src/build_dataset.py \
  --source entitables \
  --input-dir dataset/tables_redi2_1 \
  --output-dir output_joinability \
  --max-tables 100
```

It intentionally keeps its simple in-memory orchestration. Use the WDC entry
point for corpus-scale, resumable construction.
