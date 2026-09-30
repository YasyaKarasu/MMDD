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
mean-pools them into ordered contiguous segments. By default it stores one
float32 vector per schema/example-row group; `--table-tokens-per-group` can keep
several segments per group for table-token-budget ablations without retaining
the full token matrix. The Student uses the final embedding from the same
forward pass. For query objects, the cache derives one `schema + row` routing
view per example row. Those short views are embedded in one additional batch
and cached in the base tier for Stage-2 evidence assignment.

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

For a multi-token table cache, pass the same
`--teacher-table-tokens-per-group K` to both Teacher stages. Fresh Teachers use
one token per group when the flag is omitted; loaded checkpoints otherwise keep
their saved token budget.

Student stages with a nonzero `--distillation-weight` automatically cache all
base-train and fixed-dev Teacher logits before optimization. `--distillation-weight 0`
is a genuinely Teacher-free supervised run and does not require
`--teacher-checkpoint`. Sidecars default to `FEATURES/teacher_logits/` and are
keyed by the full Teacher checkpoint SHA-256, candidate-list fingerprint, and
path aggregation settings. `--teacher-logit-cache` selects another directory,
and `--teacher-logit-batch-size` controls only the one-time Teacher pass. Once
both sidecars exist, Student training does not instantiate the Teacher or read
the Teacher hidden-feature tier. Changing the Teacher, candidates, split, or
path aggregation creates a distinct cache entry.

Student training also preloads every raw embedding referenced by its base,
hard, and dev examples into one contiguous CPU tensor. This avoids repeated
per-object `torch.load` calls without duplicating embeddings on disk. Disable
it with `--no-preload-embeddings` only when host memory is constrained. Pair
projection and path aggregation are vectorized, so `--batch-size` defaults to
64 for Student stages and 8 for Teacher stages; either can be overridden.

Student projector initialization is selected with `--student-initialization`.
The default `random` mode preserves the original behavior. `identity_noise`
requires `--student-dim` to match the embedding dimension and initializes each
projector as `I + noise`. `random_orthogonal` shares one row-orthogonal
projector across object types and initializes every relation as `I`, so its
initial score is the raw inner product restricted to one random low-rank
subspace. Use `--student-init-noise-std` to set the projector noise in
`identity_noise` mode. `pca` uses the same geometry but takes the shared
subspace from a projection artifact built over frozen corpus embeddings:

```bash
conda run -n MMDD python src/build_stage1_pca_basis.py \
  --features cache/stage1_qwen8b --corpus stage1_corpus.jsonl \
  --student-dim 128 --device cuda --output work/stage1_pca_128.pt
conda run -n MMDD python src/train_stage1.py student-edge \
  ... --student-dim 128 --student-initialization pca \
  --student-pca-basis work/stage1_pca_128.pt
```

Before training, the zero-training identity probe can exercise the complete
Student index and dev retrieval path with `P = I` and `R = I`. It runs on CPU;
`--raw-index` optionally evaluates an existing corpus-matched raw index in the
same invocation and records the direct Recall@10 delta:

```bash
conda run -n MMDD python src/probe_stage1_identity.py \
  --features cache/stage1_qwen8b \
  --corpus stage1_corpus.jsonl \
  --dev-data target_lists.jsonl \
  --output-dir indices/stage1_identity \
  --raw-index indices/stage1_raw
```

Use the zero-training PCA dimension probe to settle the Student dimension
before another training run. It computes the complete centered covariance
spectrum once, then evaluates shared `P = U_d^T`, `R = I` Students at the
requested dimensions. Retrieval applies `P` to the original frozen embeddings
without subtracting the PCA mean, matching the Student's linear projection.
Only table indexes are built because the reported metric is dev direct
Recall@10:

```bash
conda run -n MMDD python src/probe_stage1_pca_dimensions.py \
  --features cache/stage1_qwen8b \
  --corpus stage1_corpus.jsonl \
  --dev-data entitables_target_lists.jsonl wdc_target_lists.jsonl \
  --dimensions 128 256 512 1024 2048 \
  --raw-index indices/stage1_raw \
  --device cuda:0 \
  --output-dir work/stage1_pca_dimension_ceiling
```

The output contains `summary.json`, exact dimension and variance CSV files,
the reusable `pca_spectrum.pt`, one resumable index directory per dimension,
and `pca_dimension_ceiling.png`. The summary selects the smallest tested
dimension whose direct Recall@10 reaches at least 90% of the raw-embedding
baseline; change that rule with `--raw-fraction-threshold`.

The spectrum artifact can be passed directly to training. A geometry-preserving
Student configuration freezes the shared PCA projection, uses a lower relation
rate, and anchors the nine relation matrices to identity. In-batch negatives
expand only the supervised lists; KD remains aligned to the original cached
Teacher lists:

```bash
conda run -n MMDD python src/train_stage1.py student-edge \
  --features cache/stage1_qwen8b \
  --base-data edge_lists.jsonl --dev-data edge_lists.jsonl \
  --student-dim 1024 --student-init pca \
  --student-pca-basis work/stage1_pca_dimension_ceiling/pca_spectrum.pt \
  --freeze-projection --relation-learning-rate 1e-5 --anchor-weight 0.1 \
  --distillation-weight 0 --in-batch-negatives \
  --in-batch-max-negatives 256 --output checkpoints/student_edge.pt
```

`--edge-type-oversample text_table:2 image_table:2` can increase the share of
evidence-to-table edges. It is off by default. Frozen projection state and the
initial projection anchors are retained across Student checkpoints.

Training records are filtered to `train`; the fixed gate records are filtered
to `dev`. Teacher edge/path and Student edge stages select on their matching dev
listwise objective and never build an ANN index. Every Student path epoch saves
a candidate checkpoint, rebuilds indexes over the same complete shared corpus,
and evaluates the fixed dev queries. Its retrieval record contains the
requested `--train-eval-ks` cutoffs (defaulting to `--recall-ks` =
10/20/30/40/50) and MRR at the maximum requested cutoff for fused, direct, and
evidence rankings, both overall and under `by_dataset`, plus the number
and fraction of dev queries whose global top 10 contains a labeled positive
evidence path. The same record includes a `raw_embedding` baseline that runs
the identical zero/one-hop retrieval directly on the frozen normalized Qwen
embeddings, without the Student projection or relation matrices. This
corpus-bound raw index is built once and reused across epochs and mining rounds.
`--primary-metric` also accepts nested names such as
`direct.recall@10` or `evidence.mrr@50`.
Student-path evaluation records epoch 0 before the first optimizer step by
default, and this initial checkpoint participates in best-checkpoint selection.
If `--student-checkpoint` is omitted, path training starts directly from
`--student-initialization`; this supports a fresh PCA baseline when edge
training is known to damage retrieval geometry.
Use `--no-eval-epoch-zero` only to reproduce historical runs.

To retrain a Teacher on retrieval-aligned lists, first expand the original
handcrafted lists with frozen raw-ANN neighbors:

```bash
conda run -n MMDD python src/build_stage1_retrieval_aligned_data.py \
  --features features_qwen3_vl_embedding_8b \
  --edge-data entitables/edge_lists.jsonl wdc/edge_lists.jsonl \
  --target-data entitables/target_lists.jsonl wdc/target_lists.jsonl \
  --corpus mixed_stage1_data/stage1_corpus.jsonl \
  --raw-index-root stage1_mining/raw_embedding_index \
  --output-dir task7_teacher_retrain/data
```

The default width is 16: one positive, up to four original handcrafted
negatives, then raw hard negatives. New path targets receive the query's top
raw text and image evidence. `preflight.json` reports exactly how many selected
objects still need Teacher hidden states before training.

For large missing sets, `cache_stage1_features.py --teacher-output-dir ...`
can write disjoint Teacher-only shards on separate GPUs. Merge completed shards
with `merge_stage1_teacher_cache.py`; the main `teacher_manifest.jsonl` is
replaced atomically. Use `partition_stage1_teacher_work.py` with the main and
staging manifests to repartition only unfinished objects when one GPU finishes
early. On a shared filesystem with insufficient room for a second copy, pass
`--move` to the merge command so each staged file is installed with an atomic
rename and its staging space is released immediately. Audit newly selected images with
`audit_stage1_teacher_images.py` first so corrupt or decompression-bomb inputs
can be excluded instead of silently becoming the upstream wrapper's `NULL`
fallback. During `teacher-path`, `--teacher-rerank` together with
`--teacher-rerank-dev-data` and
`--primary-metric teacher_rerank.recall@10` selects checkpoints by raw-top-100
reranking rather than listwise dev loss alone. Use
`--teacher-rerank-interval 2` to run that expensive gate every second epoch;
the default remains every epoch.

For an output such as `student_path.pt`, training writes:

- `student_path.pt`: best checkpoint selected by the dev gate;
- `student_path.last.pt`: final attempted checkpoint, never overwriting best;
- `student_path.epochs/epoch_NNN.pt`: per-epoch candidates;
- `student_path.dev_indices/epoch_NNN/`: best and latest full-corpus indexes;
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
ranks each target in both channels. `--fusion-mode` selects ordinary RRF,
weighted RRF, or evidence-gated RRF. Weighted RRF uses `--direct-weight` and
`--evidence-weight`; setting the latter to zero makes fused ranking exactly
direct ranking while retaining discovered evidence paths for Stage 2.
The Stage-1 round-2 default is `--fusion-mode weighted_rrf
--evidence-weight 0.05`. Evaluation also emits `fused_e0` and `fused_e005`
for direct-only and default-fusion comparisons from the same retrieval pass.
When a raw embedding index is available, `evidence_identity_baseline` records
the evidence channel with identity relations.

For frozen-PCA Student path runs, `--anchor-weight-evidence` gives the four
table-to/from-text/image relations an independent identity-anchor weight.
Every epoch history records `relation_drift` (`||R-I||_F`) for all nine
directed relation matrices. Repeated `--per-dataset-gate
DATASET:METRIC>=VALUE` constraints restrict checkpoint selection to qualifying
epochs and fall back to epoch 0 with `gate_unsatisfied: true` if none qualify.
A clean dataset-gated KD ablation can use `--distillation-datasets` to apply
the global `--distillation-weight` only to named datasets.
`--evidence-modality-weights text=1 image=0.3` applies optional modality priors
inside the evidence channel. `--rrf-k` controls the rank constant (default 60). The channel
ranks and `direct_score` are intermediate values and are not written to the
retrieval results. Student relation queries and projected target vectors
preserve the bilinear score exactly as an inner product for ANN indexing.
Evidence-to-target expansion submits all evidence relation vectors for one
query to HNSW in one batch. Relation vectors are cached for the lifetime of the
loaded index, so evidence reused across full-dev retrieval is projected once.

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
all serialized. By default, `k=10`, the direct pool is `gamma*k=40`, and both
the evidence and evidence-to-target pools are `gamma_evidence*k=20`.
`--direct-k`, `--evidence-k`, and `--targets-per-evidence` remain explicit
advanced overrides. The first `k` target IDs/scores are returned unless
`--result-k` overrides that serialization limit; only the first 10 targets
retain a direct marker and the top four aggregated evidence paths.
`--path-result-k` and `--evidence-path-k` control those two path-detail limits;
they must cover the Stage-2 **input** `--input-candidate-budget` (N, default 50)
and `--top-k-evidence` values. The recovery budget M is not a path-detail limit.
Returning 50 IDs with only 10 targets' paths is insufficient for N=50, even if
M is smaller than 10. Stage 2 rejects this input before loading models; it does
not automatically retrieve or export replacement data. Each target keeps only its final fusion `score`, a
non-null `evidence_score` when available for Stage 2, and compact paths:
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
  --output-target-lists hard_targets_round1.pending.jsonl \
  --output-edge-lists hard_edges_round1.pending.jsonl

conda run -n MMDD python src/cache_stage1_features.py \
  --input-jsonl stage1_objects.jsonl --output-dir cache/stage1_qwen8b \
  --model-dir hf_models/Qwen3-VL-Embedding-8B \
  --teacher-data \
    hard_targets_round1.pending.jsonl hard_edges_round1.pending.jsonl

conda run -n MMDD python src/refresh_stage1_hard_negatives.py \
  --features cache/stage1_qwen8b \
  --teacher-checkpoint checkpoints/teacher_path.pt \
  --student-checkpoint checkpoints/student_path_round0.pt \
  --index-dir indices/stage1_round0 --corpus stage1_corpus.jsonl \
  --target-lists target_lists.jsonl \
  --pending-target-lists hard_targets_round1.pending.jsonl \
  --pending-edge-lists hard_edges_round1.pending.jsonl \
  --output-target-lists hard_targets_round1.jsonl \
  --output-edge-lists hard_edges_round1.jsonl
```

The scoring pass reads the persisted pending candidates directly. It does not
repeat ANN retrieval after Teacher features have been supplemented.

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

The natural-evidence column selector (`NAT_E`) and the bridge-first reranking stage are documented
in `EVIDENCE_SELECTOR_AND_BRIDGE_RERANK.md`. The merged provenance and remaining runtime limits are
recorded in `MIGRATION_GAPS.md`.

For a fresh R7-style C30 run, create a new audit output with
`run_stage2_columns.py audit --candidate-scope-file <stage1-c30-scope.json>`. The scope may list
candidate target tables per query or candidate columns per `(dataset, query, target)` pair. Pass
`--view-seeds 13001 26002` to the cache entry points when reproducing the R7 second view; custom
seeds are part of each cache contract. After fresh bridge score files exist,
`run_stage2_b_plus_idf.py` fuses them with frozen IDF visible scores without loading a previous
Stage-2 checkpoint.

Stage 2 is an executable RATA/FOCUS pipeline over canonical dataset artifacts
and Stage-1 retrieval JSON. `--input-candidate-budget` sets N (default 50): the
first N unique targets in the Stage-1 result order. Duplicate target IDs and
missing path detail are errors. `--recovery-budget` independently sets M
(default 20, zero is allowed). `--max-targets` remains an alias for N only;
`--max-direct-targets` has been removed.

Inside this single Top-N pool, a `Q -> T` path creates a direct branch and a
`Q -> E -> T` path creates an evidence branch. A target can have both. Branches
depend only on retrieved paths, never on gold implicit/explicit labels, and
path existence does not imply joinability. All direct candidates are verified
against the original query; they neither consume M nor get truncated by it.

For **every evidence candidate**, the query, target and top-k evidence objects
are placed in one reader input and produce RATA boundary states for all target
columns. Stage 2 reuses `joint_candidate_probabilities` to compute
`P(T,c) = softmax(r_T) * softmax(g_T,c)` over the complete evidence pool.
The table prior is exported `stage2_table_score` when present, otherwise
`evidence_score`; the existing requirement for an evidence-channel score remains.
Repeated evidence paths have already been aggregated by Stage 1, and bundles
preserve its compact path order. No M cutoff is applied before reader scoring.

Each table selects `c_T = argmax_c P(T,c)` and receives recovery priority
`s_T = max_c P(T,c)`. The top M unique tables by this priority are recovered;
ties preserve Stage-1 order (column ties preserve header order). Summing column
probabilities would discard column information, and taking flattened top-M
pairs would allow one table to consume multiple slots; neither is used.
The full evidence pool still pays the column-selection reader cost. Only
row routing, FOCUS localization, value generation and evidence-branch final
verification are limited to M tables. Preselected columns are reused without
another reader call. Each target has its own generated column, representing
its augmented query `Q_T+`, without modifying the original query or another
candidate's row values.

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

For mixed-source training, pass one dataset root per retrieval file in the
same order. A single root is still shared by every retrieval file:

```bash
conda run -n MMDD python src/train_stage2.py \
  --dataset-root output_entitables output_wdc \
  --retrieval-results retrieval_entitables_train.jsonl retrieval_wdc_train.jsonl \
  --stage1-gate runs/stage1_mining/final_selection.json \
  --model-dir hf_models/Qwen3.5-9B \
  --output checkpoints/stage2_candidate.pt
```

For each table admitted to recovery, after the target column is fixed, each selected evidence object's original
Qwen embedding is compared with the cached query-row routing embeddings and
assigned to exactly one row by cosine argmax. A row may receive zero or many
evidence objects, but each evidence object runs through FOCUS at most once per
recovered target (localization is target-attribute-specific).
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
marked joinable only when enough query rows match the selected target column.
Both branches use the same exact/fuzzy-or-semantic matching rule, similarity
threshold and minimum coverage. Empty values count in the denominator of all
query rows and never count as matches. Direct verification chooses its best
original query-column/target-column pair by coverage, then mean similarity.

Verified targets are merged by `target_id` and sorted by coverage descending,
mean similarity descending, then Stage-1 rank ascending. If both branches were
verified, the better branch under the same rule supplies the final score;
an exact tie keeps direct. Both branch results and all generated row evidence
remain in the output. There is no branch-specific normalization, and `P(T,c)`
is never a final joinability score. Verified failures remain in the reranked
list; evidence-only targets outside M are separately marked not attempted.

```bash
conda run -n MMDD python src/run_stage2.py \
  --dataset-root output_mm_joinability_v15 \
  --retrieval-results retrieval_q1.json \
  --stage1-gate runs/stage1_mining/final_selection.json \
  --stage1-features cache/stage1_qwen8b \
  --scorer-checkpoint checkpoints/stage2_candidate.pt \
  --model-dir hf_models/Qwen3.5-9B \
  --input-candidate-budget 50 --recovery-budget 20 \
  --output stage2_q1.json
```

Output contains `input_candidate_budget`, the actual `input_candidate_count`,
`recovery_budget`, `reranked_candidates`, and `unattempted_candidates`.
Every candidate includes its `target_id`, 1-based `stage1_rank`, `stage1_score`,
`table_prior` (raw r_T), `table_probability`, `selection`, per-column
`joint_probabilities`, `recovery_priority`, and `selected_for_recovery`.
Direct-only candidates without an exported table prior use null; their
`stage1_score` is still retained and they do not participate in the evidence softmax.
Each `branches.direct` / `branches.evidence` record retains its own
`verification`; direct records identify the matched query and target columns,
and evidence records retain `evidence_ids` plus row-aligned `rows` with values
and compact localization provenance. Unrouted rows have empty values and null
evidence. `final_branch`, `verification` and 1-based `rerank_rank` identify the
merged result. Unattempted candidates have null final verification and rank,
not fabricated zero scores; their evidence branch has status `not_attempted`
and reason `recovery_budget`. Full input tables are not duplicated.

`Stage2Verifier.verify(query, retrieval_results, targets, evidence, ...)` takes
the already selected Top-N path records. `candidate_logits`, the candidate-head
training interface, and Oracle loading/cache interfaces remain unchanged.
This workflow change is covered by synthetic offline unit tests only, not
training or model-inference experiments. Recall@1/3/5/7/9 evaluation, ablations,
budget comparisons and effectiveness claims are outside this implementation.

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
- `splits.json` is a small query-split summary; per-query assignments live on
  `query_tables[*].split`, which is their single source. `data_lake_tables` is
  the single shared corpus and is never assigned to train/dev/test.

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

The joinability construction mirrors the legacy builder's current algorithm:
implicit queries require at least two ordinary context columns, so the weakest
qualified bridges are demoted into the shared context pool until the floor is
met and sources that cannot reach it emit no implicit queries. Exact redundant
columns (identical row-aligned cell values after serialization) share one query
bridge that fans out to a seeded 1..k subset of their physical members, each
with its own chain, target table, qrel, and member-specific evidence
recoveries. A materialized query table from a Wikipedia-shaped corpus also gains
a synthetic `entity_url` column derived from the entity cell's wiki title
(`entitables`); WDC tables mint a synthetic `wdc_<hash>` title instead, so that
column would hold a fabricated `en.wikipedia.org` URL and is omitted. Dataset
cell text is cleaned and truncated to 1024 characters with URLs preserved.
