# Source implementation

`src/` contains the standalone research implementation for MMDD. Dataset
construction lives in `mmdd_dataset/`; the directed multimodal joinability
Teacher/Student implementation lives in `mmdd_stage1/`. Neither package
imports `scripts_old/`. Annotation, GPU scheduling, marker protocols, and
service orchestration remain outside `src/`.

## Directed joinability Teacher/Student

Build the initial Stage-1 files directly from a completed dataset artifact:

```bash
conda run -n MMDD python src/build_stage1_training_data.py \
  --dataset-root output_mm_joinability_v15 \
  --output-dir work/stage1_v15
```

This writes `stage1_objects.jsonl`, `edge_lists.jsonl`, `target_lists.jsonl`,
and `stage1_corpus.jsonl`. Initial target lists draw from random,
TF-IDF-similar non-joinable, type/structure-matched, and corrupted-path
negatives without retaining those construction-only labels. Candidate evidence
contains all unique recovery or source-provenance assets.
Table serialization cleans each cell and retains at most its first 1024
characters so anomalously long WDC cells cannot consume the encoder context.
Use `--max-cell-chars` to change this limit; changing it requires rebuilding
the Stage-1 objects and feature cache.
Target lists built with the former per-target evidence cap must be regenerated,
along with their cached Teacher scores, before path training.

`cache_stage1_features.py` freezes Qwen3-VL-Embedding and stores the two
feature granularities in separate tiers. The base `objects/` tier contains the
normalized final embedding for every object plus query-row routing embeddings;
Student training, ANN indexing, retrieval, and Stage 2 read only this tier.
The optional `teacher_objects/` tier contains pooling-before states only for
objects referenced by the supplied training lists. Its input is JSONL. A text
object uses only its body in `text`, and an image object uses only its local
`image` file. Tables use only column names and cell values.
Page titles, captions, sections, entity labels, source names, and provenance
fields are never serialized into model input. Tables carry their retrieval
identity in `embedding_role`: query tables use `query` and candidate tables use
`target`; text and image objects are evidence by definition.
A table additionally supplies `table_parts`, with schema text first and one
entry per example row after it. The cache derives the complete table text and
query-row routing views from this single list:

```json
{"object_id":"q1","object_type":"table","embedding_role":"query","table_parts":["Columns: player | country","Row: Messi | Argentina"]}
{"object_id":"e1","object_type":"text","text":"Lionel Messi represents Argentina."}
{"object_id":"i1","object_type":"image","image":"images/i1.jpg"}
```

The encoder uses separate instructions for query tables, query-row routing
views, target tables, text evidence, and image evidence. They emphasize join
keys and row identity on the query side, attributes offered by target tables,
explicitly stated facts in text, and visually grounded facts in images.
`--instruction` remains available as an explicit global override.

Each table is encoded once for its Teacher/Student features. The cache uses
tokenizer offsets to retain the schema/row tokens from that same sequence and
immediately mean-pools them to one float32 vector per schema/example-row group.
This is the exact first operation performed by the Teacher and avoids retaining
the much larger token matrix. The Student uses the final embedding from the
same forward pass. For query objects, the cache derives one `schema + row`
routing view per example row. Those short views are embedded in one additional
batch and cached in the base tier for Stage-2 evidence assignment.

Build a lazy per-object feature cache with the local 8B encoder:

```bash
conda run -n MMDD python src/cache_stage1_features.py \
  --input-jsonl stage1_objects.jsonl \
  --output-dir cache/stage1_qwen8b \
  --model-dir hf_models/Qwen3-VL-Embedding-8B \
  --teacher-data edge_lists.jsonl target_lists.jsonl \
  --teacher-split all
```

The dev listwise objectives require Teacher features for the dev records, so
the training cache uses `--teacher-split all`; test records are cached but are
never loaded by training or checkpoint selection. Omit `--teacher-data` for a
base-only retrieval/Stage-2 cache. Re-running the
same command with additional hard-negative files writes only missing Teacher
objects; already cached base objects are not encoded again. `--teacher-split`
defaults to `train` and accepts `all` when all record splits are needed.

Convert an existing all-hidden-state cache without running Qwen again:

```bash
conda run -n MMDD python src/compact_stage1_feature_cache.py \
  --input-dir cache/stage1_qwen8b_legacy \
  --output-dir cache/stage1_qwen8b \
  --teacher-data edge_lists.jsonl target_lists.jsonl \
  --teacher-split all
```

The conversion is resumable and can also prune an existing two-tier cache into
a new directory containing only the Teacher objects referenced by the current
lists. Verify the new cache before removing the legacy directory. Feature
caches created before the row-routing format still need to be rebuilt; Stage 2
fails explicitly when a selected query has no cached `row_embeddings`.

Edge warm-up data contains a source object, an unordered same-destination-type
candidate list, and its one positive object. The historical `query_id` field
identifies the source, including for evidence-to-target edges:

```json
{"query_id":"q1","source_type":"table","candidate_ids":["e1","e2"],"positive_id":"e1","destination_type":"text","dataset":"2k","split":"train"}
```

Construction emits query-to-target edges plus query-to-evidence and
evidence-to-target edges for all text/image recoveries whenever a valid negative
exists. Each positive edge gets its own list; other known positives are excluded
from its negatives. `source_type` and `destination_type` are checked against the
feature cache during scoring.

Path-level data groups all unique evidence by candidate target. Mined evidence
remains in retrieval-score order:

```json
{"query_id":"q1","direct_positive_target_id":"t1","evidence_positive_target_id":"t2","positive_target_ids":["t1","t2"],"candidates":[{"target_id":"t1","evidence_ids":[]},{"target_id":"t2","evidence_ids":["e1","i1"]}],"dataset":"2k","split":"train"}
```

Run the four training stages explicitly:

```bash
conda run -n MMDD python src/train_stage1.py teacher-edge \
  --features cache/stage1_qwen8b \
  --base-data edge_lists.jsonl --dev-data edge_lists.jsonl \
  --output checkpoints/teacher_edge.pt

conda run -n MMDD python src/train_stage1.py teacher-path \
  --features cache/stage1_qwen8b \
  --base-data target_lists.jsonl --dev-data target_lists.jsonl \
  --teacher-checkpoint checkpoints/teacher_edge.pt \
  --output checkpoints/teacher_path.pt

conda run -n MMDD python src/train_stage1.py student-edge \
  --features cache/stage1_qwen8b \
  --base-data edge_lists.jsonl --dev-data edge_lists.jsonl \
  --teacher-checkpoint checkpoints/teacher_path.pt \
  --output checkpoints/student_edge.pt

conda run -n MMDD python src/train_stage1.py student-path \
  --features cache/stage1_qwen8b \
  --base-data target_lists.jsonl --dev-data target_lists.jsonl \
  --teacher-checkpoint checkpoints/teacher_path.pt \
  --student-checkpoint checkpoints/student_edge.pt \
  --corpus stage1_corpus.jsonl \
  --primary-metric recall@10 --min-delta 0.001 --patience 3 \
  --output checkpoints/student_path.pt
```

Training records are filtered to `train`; the fixed gate records are filtered
to `dev`. Teacher edge/path and Student edge stages select on their matching dev
listwise objective and never build an ANN index. Every Student path epoch saves
a candidate checkpoint, rebuilds indexes over the same complete shared corpus,
and evaluates the fixed dev queries. Its retrieval record contains Recall@1/5/
10/50/100 and MRR@100 for fused, direct, and evidence rankings, plus the number
and fraction of dev queries whose global top 10 contains a labeled positive
evidence path. `--primary-metric` also accepts nested names such as
`direct.recall@10` or `evidence.mrr@100`.

For an output such as `student_path.pt`, training writes:

- `student_path.pt`: best checkpoint selected by the dev gate;
- `student_path.last.pt`: final attempted checkpoint, never overwriting best;
- `student_path.epochs/epoch_NNN.pt`: per-epoch candidates;
- `student_path.dev_indices/epoch_NNN/`: per-epoch full-corpus indexes;
- `student_path.pt.history.json`: epoch objectives, retrieval metrics, sampling
  counts, best epoch, and stop reason;
- `student_path.pt.selection.json`: best checkpoint/index fingerprints and the
  Stage-2 evidence-coverage decision.

The Teacher uses one shared Relation Transformer with modality, direction,
and ordered type-pair identities. The Student learns one projection per type
and one relation matrix per ordered type pair. Target/path training keeps the
direct `Q -> T` and evidence `Q -> E -> T` channels separate. The direct
channel uses dataset qrels as its positives; a recovery target absent from
those qrels is a direct hard negative. The evidence channel uses query-scoped
recoveries as its positives, so the two listwise losses may use different
positive target indices. The evidence channel
first aggregates each target's paths with the configured LogSumExp, top-k
mean, or top-k sum, then uses a separate listwise loss over targets that have
evidence. A batch row contributes evidence loss only when its positive and at
least one negative have evidence. The two channel losses are added; their
scores are never combined into one training logit. The path checkpoint stores
the evidence aggregation configuration.

Online retrieval computes `direct_score` and `evidence_score` separately and
ranks each target in both channels. Reciprocal Rank Fusion (RRF) combines the
two ranks into the single `score` used for global top-k truncation and Recall@k
evaluation. `--rrf-k` controls the rank constant (default 60). The channel
ranks and `direct_score` are intermediate values and are not written to the
retrieval results. Student relation queries and projected target vectors
preserve the bilinear score exactly as an inner product for ANN indexing.

`--base-data` and `--dev-data` accept multiple files. Every record should carry `dataset`;
when it does not, the input filename stem is used. Sampling assigns dataset
mass proportional to `n_d ** alpha`. The default `--dataset-sampling-alpha 0`
gives 2K and 20K equal epoch mass, while `1` preserves their natural sample
ratio. Intermediate values provide temperature-style sampling. Every history
record includes `dataset_samples` so the realized balance is auditable:

```bash
conda run -n MMDD python src/train_stage1.py teacher-edge \
  --features cache/stage1_qwen8b \
  --base-data edge_lists_2k.jsonl edge_lists_20k.jsonl \
  --dev-data edge_lists_2k.jsonl edge_lists_20k.jsonl \
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
  --index-dir indices/stage1 --corpus stage1_corpus.jsonl \
  --output retrieval_q1.json
```

Retrieval expands only `Q -> T` and `Q -> E -> T`, keeps the evidence object
on each path, and restores the two-level path aggregation from the Student
checkpoint. Explicit retrieval flags may override that saved configuration.
All paths participate in channel aggregation and RRF ranking, but they are not
all serialized. By default, Recall@100 keeps 100 ranked target IDs/scores while
only the first 10 targets retain a direct marker and the top four aggregated
evidence paths. `--path-result-k` and `--evidence-path-k` control those two
limits; they must be at least the corresponding Stage-2 `--max-targets` and
`--top-k-evidence` values. Each target keeps only its final fusion `score`, a non-null
`evidence_score` when available for Stage 2, and compact paths:
`{"kind":"direct"}` or
`{"kind":"evidence","evidence_id":"e1","path_score":1.2}`.

### Hard-negative refresh

After several Student epochs, rebuild the HNSW indexes from that exact Student
checkpoint and refresh its candidate distribution:

```bash
conda run -n MMDD python src/refresh_stage1_hard_negatives.py \
  --features cache/stage1_qwen8b \
  --teacher-checkpoint checkpoints/teacher_path.pt \
  --student-checkpoint checkpoints/student_path_round0.pt \
  --index-dir indices/stage1_round0 \
  --corpus stage1_corpus.jsonl \
  --target-lists target_lists_2k.jsonl target_lists_20k.jsonl \
  --output-target-lists hard_targets_round1.jsonl \
  --output-edge-lists hard_edges_round1.jsonl \
  --hard-targets-per-query 16 \
  --hard-evidence-per-type 16 \
  --hard-paths-per-query 16 \
  --mining-round 1
```

With a two-tier cache, first mine without Teacher scoring, supplement only the
newly referenced Teacher objects, then rerun the same refresh command without
`--mine-only` to write logits:

```bash
conda run -n MMDD python src/refresh_stage1_hard_negatives.py \
  --mine-only --features cache/stage1_qwen8b \
  --student-checkpoint checkpoints/student_path_round0.pt \
  --index-dir indices/stage1_round0 --corpus stage1_corpus.jsonl \
  --target-lists target_lists.jsonl \
  --output-target-lists hard_targets_round1.jsonl \
  --output-edge-lists hard_edges_round1.jsonl

conda run -n MMDD python src/cache_stage1_features.py \
  --input-jsonl stage1_objects.jsonl --output-dir cache/stage1_qwen8b \
  --model-dir hf_models/Qwen3-VL-Embedding-8B \
  --teacher-data hard_targets_round1.jsonl hard_edges_round1.jsonl

# Rerun the first refresh command without --mine-only and add:
# --teacher-checkpoint checkpoints/teacher_path.pt
```

The refresh mines three independent current-Student distributions. `Q -> T`
ANN returns hard targets outside `positive_target_ids`. Per-modality `Q -> E`
ANN returns hard evidence outside the query-scoped positive evidence set,
whether or not that evidence reaches a selected target. Finally, every `Q -> E`
ANN hit is expanded through `E -> T`; complete paths ending at a non-GT
target are ranked directly by `s(Q,E) + s(E,T)` to obtain path-hard negatives.
Path mining does not use target aggregation or RRF. A target found by both the
direct and path channels is deduplicated while retaining its mined evidence.

Shared mining provenance and the three quotas are written once to the
corresponding `.jsonl.metadata.json` sidecars instead of repeated on every
record. The frozen Teacher then rescores the target/path lists and their
directly supervised edge lists. Independently mined same-modality evidence is
used for query-to-evidence lists; path evidence remains attached to its wrong
target for the Evidence target channel. Edge outputs store aligned
`teacher_logits`; target outputs separately store aligned
`teacher_direct_logits` and `teacher_evidence_logits`.
`student-edge` and `student-path` consume these cached soft labels directly.
The loader rejects the obsolete merged target `teacher_logits` format and also
checks declared edge types, the Teacher checkpoint, and path-aggregation
configuration before reuse:

```bash
conda run -n MMDD python src/train_stage1.py student-edge \
  --features cache/stage1_qwen8b \
  --base-data edge_lists.jsonl --hard-data hard_edges_round1.jsonl \
  --dev-data edge_lists.jsonl --hard-fraction 0.5 \
  --teacher-checkpoint checkpoints/teacher_path.pt \
  --student-checkpoint checkpoints/student_path_round0.pt \
  --hard-source-checkpoint checkpoints/student_path_round0.pt \
  --hard-learning-rate 2e-5 \
  --output checkpoints/student_edge_round1.pt

conda run -n MMDD python src/train_stage1.py student-path \
  --features cache/stage1_qwen8b \
  --base-data target_lists.jsonl --hard-data hard_targets_round1.jsonl \
  --dev-data target_lists.jsonl --hard-fraction 0.5 \
  --teacher-checkpoint checkpoints/teacher_path.pt \
  --student-checkpoint checkpoints/student_edge_round1.pt \
  --hard-source-checkpoint checkpoints/student_path_round0.pt \
  --corpus stage1_corpus.jsonl --hard-learning-rate 2e-5 \
  --output checkpoints/student_path_round1.pt
```

The hard fraction is sampled explicitly and deterministically every epoch; a
value of `0.5` gives one sampled hard record per sampled base record. Base and
hard counts are separate in history. Hard inputs must have one round number and
matching source-Student, frozen-Teacher, and path-aggregation fingerprints.

Use the multi-round driver to rebuild, mine, supplement the two-tier Teacher
cache when needed, rescore, mixed-train, and dev-gate until the round patience
or maximum is reached:

```bash
conda run -n MMDD python src/run_stage1_rounds.py \
  --features cache/stage1_qwen8b --objects stage1_objects.jsonl \
  --corpus stage1_corpus.jsonl \
  --teacher-checkpoint checkpoints/teacher_path.pt \
  --initial-selection checkpoints/student_path.pt.selection.json \
  --base-edge-data edge_lists.jsonl --base-path-data target_lists.jsonl \
  --dev-edge-data edge_lists.jsonl --dev-path-data target_lists.jsonl \
  --test-data target_lists.jsonl --output-dir runs/stage1_mining \
  --max-mining-rounds 3 --round-patience 2 \
  --hard-fraction 0.5 --hard-learning-rate 2e-5
```

Each `round_NN/` owns its mining index, pending and Teacher-scored hard files,
metadata, Student edge/path checkpoints, per-epoch dev indexes, and metrics.
Step markers bind all reusable outputs to their input fingerprints; mismatches
raise instead of silently reusing an old index or logits. `final_selection.json`
records the selected round, epoch, checkpoint, complete dev metrics, round stop
reason, and the single final test evaluation. Test labels are loaded only after
round selection has finished.

## Stage-2 verification

Stage 2 is an executable RATA/FOCUS pipeline over canonical dataset artifacts
and Stage-1 retrieval JSON. It first truncates the single global target list by
its RRF `score` order. Inside that fixed candidate pool, it keeps
`Q -> E -> T` paths for multimodal verification and checks direct `Q -> T`
paths separately. The evidence branch uses the Stage-1 `evidence_score` for
`softmax(r_T)`, but this does not create a second retrieval queue or change
which targets passed the global cutoff. Repeated paths to one evidence object
are aggregated with LogSumExp for evidence selection. For each candidate
target, the query, target, and all selected top-k evidence objects are placed
in one transformer input and produce one set of RATA boundary states.
Retrieval JSON created before these channel scores were added must be
regenerated.

The RATA reader uses Qwen3.5's existing `<|object_ref_start|>` and
`<|object_ref_end|>` tokens around every target header. Qwen is frozen; only
the linear candidate head is trained. Because the Stage-1 evidence-channel
distribution `softmax(r_T)` is fixed, training runs the reader only for the
gold target and optimizes the column loss `-log rho_(T,c)`. History records the
optimized term as `column_loss` and the fixed Stage-1 term as `table_loss`,
while inference still ranks all candidate pairs with
`softmax(r_T) * rho_(T,c)`. Training retrieval files
contain one JSON object or JSONL record per query in the format emitted by
`retrieve_stage1.py`; records whose positive target has no retrieved evidence
path are skipped:

```bash
conda run -n MMDD python src/train_stage2.py \
  --dataset-root output_mm_joinability_v15 \
  --retrieval-results retrieval_train.jsonl \
  --stage1-gate runs/stage1_mining/final_selection.json \
  --model-dir hf_models/Qwen3.5-9B \
  --output checkpoints/stage2_candidate.pt
```

After the target column is fixed, each selected evidence object's original
Qwen embedding is compared with the cached query-row routing embeddings and
assigned to exactly one row by cosine argmax. A row may receive zero or many
evidence objects, but each evidence object runs through FOCUS at most once.
The routing embeddings contain only `Columns: ...` plus the current `Row: ...`
for a query row, only `content` for text evidence, and only pixels for image
evidence. No external table or asset metadata participates in routing.
Rows with no assigned evidence produce an empty value without localization or
generation. For assigned evidence, the Qwen backend captures the later
full-attention layers' `v_proj` outputs. It builds separate entity and attribute
relevance maps over text or image tokens and combines them. For text evidence,
joint pre-softmax logits from overlapping token windows are averaged on the
full evidence token axis, normalized once per layer, averaged over layers, and
reduced to a coherent high-relevance span. Image maps are
Gaussian-smoothed; FOCUS-style separated anchors, adaptive ROI expansion, NMS,
and an existence confidence pass select the crop. The output names the
modality-local values explicitly as `text_span_relevance` or
`image_presence_probability`; they are never compared across evidence objects.
All localized text spans and image crops routed to one row are instead placed
in one multimodal prompt with fixed single-letter labels. The Qwen next-token logits
for those labels form one row-local listwise decision, and the highest-logit
candidate is used to generate the bridge value. Rows with one candidate skip
the redundant reranker forward. Aggregated retrieval path scores determine the
top-k evidence set but do not modify this final selection. Candidate order and
labels remain fixed; no order rotation is applied. The generated column is
accepted only when enough query rows semantically match values in the selected
target column. The output keeps only the selected target/column,
generated row values, compact evidence provenance, direct matches, and the
final verification summary; it does not duplicate the full input table.

```bash
conda run -n MMDD python src/run_stage2.py \
  --dataset-root output_mm_joinability_v15 \
  --retrieval-results retrieval_q1.json \
  --stage1-gate runs/stage1_mining/final_selection.json \
  --stage1-features cache/stage1_qwen8b \
  --scorer-checkpoint checkpoints/stage2_candidate.pt \
  --model-dir hf_models/Qwen3.5-9B \
  --output stage2_q1.json
```

The result schema is intentionally compact:

```json
{"query_id":"q1","selection":{"target_id":"t1","column_index":1,"column_name":"Club"},"rows":[{"row_id":0,"value":"Barcelona","evidence":{"evidence_id":"e1","evidence_type":"text","text_span":"Messi plays for Barcelona.","text_span_relevance":0.93}}],"verification":{"joinable":true,"coverage":1.0,"mean_similarity":0.91}}
```

`mmdd_stage2/verifier.py` contains the paper-derived math,
`mmdd_stage2/qwen.py` is the only model-specific boundary, and
`mmdd_stage2/pipeline.py` implements table/column selection, row filling,
direct-path checking, and final semantic joinability.

The two source families share one joinability algorithm in
`mmdd_dataset/joinability.py`:

1. choose a candidate entity column;
2. test whether evidence-backed extraction recovers a hidden attribute;
3. project the visible entity and selected additional table columns into a query;
4. project the recovered attribute and selected additional table columns into a target;
5. emit the query, target, qrel, recovery path, and table decision.

`build_joinability_for_table()` is the streaming boundary. The compact
EntiTables builder and the WDC backend both call this same function. WDC adds
only a source adapter and a disk-backed execution backend.

Train/dev/test partitions apply only to query tables. Queries derived from the
same source table remain together, while every split retrieves against the same
complete `data_lake_tables` artifact. Data-lake table records therefore have no
`split` field; query-scoped qrels and evidence recoveries inherit the query
split. Stage-1 negative construction follows the same contract and samples
targets and evidence from the complete shared corpus for every query split.

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

The final directory uses `mmdd_joinability_sharded_v3`. It is self-contained:
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
- `splits.json` is a small query-split summary, while detailed query assignments
  live in the sharded `split_assignments` artifact; `data_lake_tables` is the
  single shared corpus and is never assigned to train/dev/test.

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
  --max-tables 100 \
  --text-model-base-url http://127.0.0.1:8001/v1
```

Its default auto-check policy sends every local blind extraction to Luna. A
matching local/Luna result is final; a disagreement is sent to Terra. Provide
the remote endpoint/model flags when they differ from their OpenAI-compatible
defaults. For large builds that must never call a remote checker, select the
preserved local-only interface explicitly:

```bash
conda run -n MMDD python src/build_dataset.py \
  --source entitables \
  --input-dir dataset/tables_redi2_1 \
  --output-dir output_joinability_local_only \
  --text-model-base-url http://127.0.0.1:8001/v1 \
  --auto-check-mode local
```

It intentionally keeps its simple in-memory orchestration. Use the WDC entry
point for corpus-scale, resumable construction.
