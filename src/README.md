# Source implementation

`src/` contains the standalone research implementation for MMDD. Dataset
construction lives in `mmdd_dataset/`; the directed multimodal joinability
Teacher/Student implementation lives in `mmdd_stage1/`. Neither package
imports `scripts_old/`. Annotation, GPU scheduling, marker protocols, and
service orchestration remain outside `src/`.

## Directed joinability Teacher/Student

`cache_stage1_features.py` freezes Qwen3-VL-Embedding and stores both feature
granularities required by the method: pooling-before hidden states for the
Teacher and the normalized final object embedding for the Student. Its input
is JSONL. Text and image objects use `text` and/or a local `image` path. A
table additionally supplies `table_parts`, with schema text first and one
entry per example row after it:

```json
{"object_id":"q1","object_type":"table","text":"full serialized table","table_parts":["schema: player, country","row: Messi | Argentina"]}
{"object_id":"e1","object_type":"text","text":"Lionel Messi represents Argentina."}
{"object_id":"i1","object_type":"image","image":"images/i1.jpg","text":"independent evidence image"}
```

Build a lazy per-object feature cache with the local 8B encoder:

```bash
conda run -n MMDD python src/cache_stage1_features.py \
  --input-jsonl stage1_objects.jsonl \
  --output-dir cache/stage1_qwen8b \
  --model-dir hf_models/Qwen3-VL-Embedding-8B
```

Edge warm-up data contains a query, an unordered candidate list, and its one
positive object:

```json
{"query_id":"q1","candidate_ids":["t1","t2","t3"],"positive_id":"t1","dataset":"2k","split":"train"}
```

Path-level data groups evidence by candidate target. Evidence order is the
retrieval-score order; `--max-evidence-per-target` retains its prefix:

```json
{"query_id":"q1","positive_target_id":"t1","positive_target_ids":["t1"],"candidates":[{"target_id":"t1","evidence_ids":["e1","i1"]},{"target_id":"t2","evidence_ids":[]}],"dataset":"2k","split":"train"}
```

Run the four training stages explicitly:

```bash
conda run -n MMDD python src/train_stage1.py teacher-edge \
  --features cache/stage1_qwen8b --train-data edge_lists.jsonl \
  --output checkpoints/teacher_edge.pt

conda run -n MMDD python src/train_stage1.py teacher-path \
  --features cache/stage1_qwen8b --train-data target_lists.jsonl \
  --teacher-checkpoint checkpoints/teacher_edge.pt \
  --output checkpoints/teacher_path.pt

conda run -n MMDD python src/train_stage1.py student-edge \
  --features cache/stage1_qwen8b --train-data edge_lists.jsonl \
  --teacher-checkpoint checkpoints/teacher_path.pt \
  --output checkpoints/student_edge.pt

conda run -n MMDD python src/train_stage1.py student-path \
  --features cache/stage1_qwen8b --train-data target_lists.jsonl \
  --teacher-checkpoint checkpoints/teacher_path.pt \
  --student-checkpoint checkpoints/student_edge.pt \
  --output checkpoints/student_path.pt
```

The Teacher uses one shared Relation Transformer with modality, direction,
and ordered type-pair identities. The Student learns one projection per type
and one relation matrix per ordered type pair. Target scoring combines direct
`Q -> T` and evidence `Q -> E -> T` paths; the default aggregation is
LogSumExp. Student relation queries and projected target vectors preserve the
bilinear score exactly as an inner product for ANN indexing.

`--train-data` accepts multiple files. Every record should carry `dataset`;
when it does not, the input filename stem is used. Sampling assigns dataset
mass proportional to `n_d ** alpha`. The default `--dataset-sampling-alpha 0`
gives 2K and 20K equal epoch mass, while `1` preserves their natural sample
ratio. Intermediate values provide temperature-style sampling. Every history
record includes `dataset_samples` so the realized balance is auditable:

```bash
conda run -n MMDD python src/train_stage1.py teacher-edge \
  --features cache/stage1_qwen8b \
  --train-data edge_lists_2k.jsonl edge_lists_20k.jsonl \
  --dataset-sampling-alpha 0 \
  --output checkpoints/teacher_edge.pt
```

After training, list every indexable target/evidence object in a corpus JSONL
with `object_id` and optional `object_type`, then build and query the per-type
HNSW indexes:

```bash
conda run -n MMDD python src/build_stage1_index.py \
  --features cache/stage1_qwen8b \
  --student-checkpoint checkpoints/student_path.pt \
  --corpus stage1_corpus.jsonl --output-dir indices/stage1

conda run -n MMDD python src/retrieve_stage1.py \
  --query-id q1 --features cache/stage1_qwen8b \
  --student-checkpoint checkpoints/student_path.pt \
  --index-dir indices/stage1 --output retrieval_q1.json
```

Retrieval expands only `Q -> T` and `Q -> E -> T`, keeps the evidence object
on each path, and applies LogSumExp to all paths ending at the same target.

### Hard-negative refresh

After several Student epochs, rebuild the HNSW indexes from that exact Student
checkpoint and refresh its candidate distribution:

```bash
conda run -n MMDD python src/refresh_stage1_hard_negatives.py \
  --features cache/stage1_qwen8b \
  --teacher-checkpoint checkpoints/teacher_path.pt \
  --student-checkpoint checkpoints/student_path_round0.pt \
  --index-dir indices/stage1_round0 \
  --target-lists target_lists_2k.jsonl target_lists_20k.jsonl \
  --output-target-lists hard_targets_round1.jsonl \
  --output-edge-lists hard_edges_round1.jsonl \
  --mining-round 1
```

The refresh runs current-Student `Q -> T` and `Q -> E -> T` retrieval, removes
every ID in `positive_target_ids`, retains high-ranked wrong targets, extracts
their evidence objects and complete wrong paths, and asks the frozen Teacher
to rescore both target/path and mixed-type edge lists. The output stores
aligned `teacher_logits`; `student-edge` and `student-path` consume these
cached soft labels directly. The loader also checks the Teacher checkpoint,
evidence truncation, and path-aggregation configuration before reuse:

```bash
conda run -n MMDD python src/train_stage1.py student-edge \
  --features cache/stage1_qwen8b --train-data hard_edges_round1.jsonl \
  --teacher-checkpoint checkpoints/teacher_path.pt \
  --student-checkpoint checkpoints/student_path_round0.pt \
  --output checkpoints/student_edge_round1.pt

conda run -n MMDD python src/train_stage1.py student-path \
  --features cache/stage1_qwen8b --train-data hard_targets_round1.jsonl \
  --teacher-checkpoint checkpoints/teacher_path.pt \
  --student-checkpoint checkpoints/student_edge_round1.pt \
  --output checkpoints/student_path_round1.pt
```

Rebuild the indexes from `student_path_round1.pt` before the next refresh.
This keeps the Teacher frozen and changes only the candidate distribution, as
specified by the training scheme.

## Stage-2 verification

Stage 2 is an executable RATA/FOCUS pipeline over canonical dataset artifacts
and Stage-1 retrieval JSON. It keeps only `Q -> E -> T` paths for multimodal
verification and checks direct `Q -> T` results separately. Repeated paths to
one evidence object and repeated evidence paths to one target are aggregated
with LogSumExp. The resulting evidence weights are used to average the
evidence-conditioned RATA boundary states.

The RATA reader uses Qwen3-VL's existing `<|object_ref_start|>` and
`<|object_ref_end|>` tokens around every target header. Qwen is frozen; only
the linear candidate head is trained. The loss is the negative log of
`softmax(r_T) * rho_(T,c)` for the gold target column. Training retrieval
files contain one JSON object or JSONL record per query in the format emitted
by `retrieve_stage1.py`; records whose positive target has no retrieved
evidence path are skipped:

```bash
conda run -n MMDD python src/train_stage2.py \
  --dataset-root output_mm_joinability_v15 \
  --retrieval-results retrieval_train.jsonl \
  --model-dir hf_models/Qwen3-VL-8B-Instruct \
  --output checkpoints/stage2_candidate.pt
```

For each query row, the Qwen backend captures the later attention layers'
`v_proj` outputs. It builds separate entity and attribute relevance maps over
text or image tokens, multiplies and normalizes them, and averages the maps
over layers. Text evidence is processed in overlapping token windows and
reduced to a coherent high-relevance span. Image maps are Gaussian-smoothed;
FOCUS-style separated anchors, adaptive ROI expansion, NMS, and an existence
confidence pass select the crop. The selected span/crop is then used to
generate the bridge value. The augmented query column is accepted only when
enough query rows semantically match values in the selected target column.
The output contains both row-level provenance and the materialized
`augmented_query` table.

```bash
conda run -n MMDD python src/run_stage2.py \
  --dataset-root output_mm_joinability_v15 \
  --retrieval-results retrieval_q1.json \
  --scorer-checkpoint checkpoints/stage2_candidate.pt \
  --model-dir hf_models/Qwen3-VL-8B-Instruct \
  --output stage2_q1.json
```

`mmdd_stage2/verifier.py` contains the paper-derived math,
`mmdd_stage2/qwen.py` is the only model-specific boundary, and
`mmdd_stage2/pipeline.py` implements table/column selection, row filling,
direct-path checking, and final semantic joinability.

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
