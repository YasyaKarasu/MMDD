# Source implementation

`src/` contains the standalone research implementation for MMDD. Dataset
construction lives in `mmdd_dataset/`; the directed multimodal joinability
Teacher/Student implementation lives in `mmdd_stage1/`. The AbeBooks explicit
regeneration utility reuses the maintained dataset builder in `scripts_old/`;
the training package does not depend on legacy consumers. Annotation, GPU scheduling, marker protocols, and
service orchestration remain outside `src/`.

The current standalone AbeBooks version is
`dataset/abebooks_authors_publishers_context2_all5_20261001`: **138 queries, all
five rows and two visible columns**. The 69 implicit queries now show title plus
publication year (67 queries) or publisher (2 author-join queries). Explicit
queries retain title plus their join column. Query memberships, splits, labels,
recoverable rows, and the 174 targets are preserved from the publisher version.
Run `enrich_abebooks_query_context.py --dataset <publisher-copy> --output <new-copy>`
to add real source context while checking hidden-value leakage, retained target
enrichment, and direct context equality joins. Rebuild Stage-1 and validate with
`validate_abebooks_standalone_dataset.py` afterwards. See the
[two-column report](../docs/abebooks_query_context_20261001.zh-CN.md).

Before CQET v4.1 training, migrate older standalone qrels in a new copy with
`project_abebooks_sources.py --dataset <dataset> --output <new-copy>`.
This preserves all facts and changes the legacy `explicit_join_column` reason
to the existing training vocabulary `explicit_visible_join_column`; otherwise
CQET classifies explicit queries as mixed and loses their Direct positives.
The standalone builder now emits the compatible reason. Optional source-field
deletion (`--drop-columns`) and conservative author display normalization
(`--natural-authors`) regenerate table cells from original source rows. Every
query, split, target membership, and pair judgment must remain valid, or the
copy is rejected. `--remove-assets-json` accepts IDs from a recorded curation
rule and refuses to remove annotated witnesses. Changed files are backed up
inside the copy; original data remains unchanged. These are dataset controls,
with fresh namespaces and retraining, not changes to the retrieval method.
`--contextualize-book-text` restores the attached source book's title before
each text fragment while preserving its full body. This content control
requires fresh text features; existing recovery facts retain their original
annotation provenance and record the content transformation.

The previous title-only implicit-query version is
`dataset/abebooks_authors_publishers_all5_balanced_20261001`: **82 train / 28 dev /
28 test**, with exactly 50% implicit and 50% explicit in each split. It contains
117 author-join queries and 21 publisher-join queries (9 implicit / 12 explicit).
Both attributes occur in every split. All queries have five rows, and implicit
queries require at least two verified rows. Publisher supervision uses reviewed
cover text/logos, with blind local reader errors preserved in the audit; it is
model-assisted annotation, not human gold. Parent-company logos do not establish
specific imprints. All 174 target tables retain their previous columns and cells.
See the [publisher expansion report](../docs/abebooks_authors_publishers_dataset_20261001.zh-CN.md)
for commands, normalization rules, counts, and training inputs.

The previous author-only standalone version is
`dataset/abebooks_standalone_all5_balanced_20261001`: **80 train / 26 dev / 26 test**
queries, with exactly 50% implicit and 50% explicit in each split. **Every query
has five rows**, including explicit queries. Implicit queries need at least two
verified recoverable rows. Unreviewed rows remain unsupervised. Each source row
occurs in at most one query. To balance equally sized views, surplus implicit
views expose their source author column and become explicit queries; they are
not duplicated. Remaining odd explicit views are downsampled within each split.
Build with `build_abebooks_standalone_dataset.py --implicit-rows 5 --explicit-rows 5
--minimum-recovered-rows 2 --balanced`, build Stage-1, then run
`validate_abebooks_standalone_dataset.py --dataset-root <copy> --stage1-dir <stage1>`.
All original source records, assets, and candidate memberships are preserved.
See the [all-query five-row report](../docs/abebooks_standalone_all5_dataset_20261001.zh-CN.md).

The historical mixed-size standalone version is
`dataset/abebooks_standalone_5row_balanced_20261001`: **140 train / 46 dev / 46 test**
queries, with exactly 50% implicit and 50% explicit in each split. Every implicit
query contains **five rows, with at least two verified recoverable rows**; explicit
queries contain two rows. Unreviewed rows remain unsupervised. Source rows are
never reused between queries. It retains all 174 candidates, 1,289 source records,
and 5,441 assets. The original and both earlier standalone copies are preserved.
Build a new copy with `build_abebooks_standalone_dataset.py --implicit-rows 5
--explicit-rows 2 --minimum-recovered-rows 2 --balanced`, then build Stage-1 and run
`validate_abebooks_standalone_dataset.py --dataset-root <copy> --stage1-dir <stage1>`
to validate and export split-specific training inputs. See
[`the five-row construction report`](../docs/abebooks_standalone_5row_dataset_20261001.zh-CN.md)
for full commands, partial-recovery counts, and training inputs.

The historical balanced two-row version is
`dataset/abebooks_standalone_balanced_20261001`: **236 train / 78 dev / 78 test**
queries, with exactly 50% implicit and 50% explicit in each split. It retains all
196 implicit queries and downsamples 65 explicit queries within the existing
split boundaries, using source-stratified rounds with seed 13. All candidate
tables, evidence, retained query inputs/IDs, and source-group coverage are preserved.
The parent remains unchanged. Use `rebalance_abebooks_standalone.py --dataset
<parent> --output <new-copy>` to reproduce it, then rebuild Stage-1 supervision.
See the balanced copy's `REPORT.md` and `TRAINING_PROTOCOL.json` for its files.

The unbalanced parent is `dataset/abebooks_standalone_20261001`:
457 nonoverlapping two-row queries, split into **273 train / 92 dev / 92 test**
with 118/39/39 fully evidence-supported implicit queries. It retains all 174
candidate memberships and 5,441 assets, changes complementary book projections,
and qualifies new blind local author readings. See
[`docs/abebooks_standalone_dataset_20261001.zh-CN.md`](../docs/abebooks_standalone_dataset_20261001.zh-CN.md)
for annotation limits, independent group counts, reproduction commands, and
training inputs. `build_abebooks_standalone_dataset.py` refuses an existing output
directory. `review_abebooks_author_evidence.py prepare/infer` prepares and reads
evidence without supplying the source author answer.

Stage-1 construction honors this dataset's `recovery_records_only_no_provenance_fallback`
manifest policy. It does not turn unreviewed source assets into positive evidence.
Use `edge_lists` for edge supervision and `target_lists.evidence_supervised` for
verified path supervision; use the complete `target_lists` for full-split retrieval
evaluation. All three splits have both image and text supervision. Fresh features
and model runs are required; dataset validation does not establish a training gain.

The 13-query AbeBooks author pilot is an audit case set, **not a training or
benchmark dataset**. Its 9/3/1 split does not meet the experiment's requirements;
exported Stage-1 files establish format compatibility only. Do not use this
subset as the final dataset optimization result.

For this fixed-lake audit, `curate_abebooks_dataset.py` copies the
dataset to a new directory, preserves every candidate/source/asset, and audits
author queries using content-bound evidence reviews. Run it from an isolated
working directory with absolute paths:

```bash
cd /tmp
conda run -n MMDD python /path/to/MMDD/src/curate_abebooks_dataset.py \
  --dataset /path/to/MMDD/dataset/abebooks_joinability_no4_disjoint_20260930 \
  --output /path/to/MMDD/dataset/abebooks_author_join_pilot_20261001 \
  --evidence-reviews /path/to/MMDD/configs/abebooks_author_evidence_review_20261001.jsonl
```

The output directory must not exist. Reviews record Codex inspection of existing
approved assets; their content hashes are verified before use. Only full author
values agreeing with the source are usable; partial lists, surname-only values,
and editor/author uncertainty remain documented. Queries need at least two
supported rows and complete judgments across the candidate lake. Equal author
values that join different captured book records are rejected. Alternative
valid target views are all exported as positives, including their recovery paths.
Original splits and IDs remain; excluded queries and all original labels are
archived. `audit/` contains per-query decisions, full-lake row-pair judgments,
normalization mappings, source/evidence split audits, and offline value-replay
controls. Replay is not fresh model recovery or a Direct retrieval result.
The generated `REPORT.md` describes the exact scope and remaining limitations.
Rebuild Stage-1 files and use a new cache namespace for this supervised subset.

Regenerate AbeBooks explicit tasks on sources that never successfully produced
implicit queries (including historical queries removed by later cleaning):

```bash
conda run -n MMDD python src/regenerate_abebooks_explicit.py \
  --source dataset/abebooks_joinability_no4_balanced_20260930 \
  --output dataset/abebooks_joinability_no4_disjoint_20260930
```

This preserves implicit queries, labels, evidence and splits; rebuilds explicit
queries, targets and labels together; and removes obsolete explicit targets,
including orphan targets. Source groups stay in one split. Candidate shortages
are recorded in `REGENERATION.json`, never filled from implicit sources.
`explicit/` contains the standalone explicit export. Rebuild Stage-1 files and
all feature/ANN caches against the resulting full dataset before evaluating it.

For a source-field ablation, `rebuild_abebooks_sources.py` projects the original
source tables before calling the maintained implicit and explicit constructors.
Its default book schema retains title, authors, publisher, and publication year.
It replays only approved facts in the input dataset's `evidence_recoveries`,
never unreviewed extraction candidates or new model annotations. Every generated
implicit query is retained; explicit candidates are deterministically balanced
on disjoint source tables. Whole source groups are split approximately 80/10/10,
with equal implicit/explicit counts in each split. `REBUILD.json` records input
hashes and any old facts not represented by the newly constructed row/layout
views. Existing review provenance is preserved; the new layouts are not claimed
to have undergone a new model review.
`--keep-columns` can retain additional existing source fields such as
`edition_number`. `--natural-authors` changes unambiguous catalog names from
`Family, Given` to `Given Family` before construction. It preserves ambiguous
lists and qualifiers, records every changed source cell, and maps normalized
recovery values back to their original approved values for the fact audit.
`--source-reference <original-dataset>` restores original source fields, entity
references and asset links before projection, checking that all currently
retained cells agree. It reads no annotations or query/target tables from that
reference: approved facts still come only from `--dataset`. This can repair an
entity anchor accidentally removed by an earlier column projection.
`--strip-series-notes` removes parenthetical clauses containing the word
`series` from all source book titles before query construction. It preserves
main titles and other qualifiers, changes no evidence, and records each change
in `TITLE_NORMALIZATION.jsonl`. When combined with title grouping, it runs
after grouping so the source partition can remain fixed for a paired control.
`--contextualize-book-text` restores the source-page heading to every attached
book text asset as `Book title: ...`, followed by the complete original fragment.
It uses source row identity only, treats labelled and unlabelled text equally,
and preserves all images and seller text. `TEXT_CONTEXT.jsonl` records the
prefix and original-content hash. Re-encode these changed texts before training.

`--group-by-title` tests a different source-table construction: title-only
word/bigram TF-IDF groups all book rows into the original number and sizes of
tables. It uses no annotations or retrieval scores, preserves every row and
evidence asset, and records original row locations for replaying existing facts.
`--group-by-authors` uses the same capacity-constrained grouping with author
text instead of titles. The two grouping options are mutually exclusive.
`--group-by-publisher` instead makes one source table per exact original
publisher value, preserving all rows and evidence. Table counts and sizes
change under this option. It tests whether publisher values spread across
many unrelated source tables make evidence-based target identification
ambiguous. It uses no labels and introduces no publisher canonicalization.
`--publisher-min-rows 5` coalesces smaller publisher groups into one tail table
without deleting rows. This reduces candidate-table count and changes the task;
report that change alongside recall. Source seller fields may need restoration
to supply enough disjoint explicit candidates; construction refuses to discard
implicit queries merely to force class balance.
Alternatively, `encode_abebooks_source_titles.py --source-tables <source-jsonl>
--output-dir <new-directory>` locally encodes every book title after removing
series notes, using GPU 1 and the frozen Qwen encoder. Pass its
`title_embeddings.npz` to `--title-embeddings` for semantic grouping with the
same capacities and assignments. Row identities and original title inputs are
checked before use. These vectors only construct source tables; they do not
replace retrieval features or change the method's prompts or trainable models.
The existing query constructor may then use fewer facts; inspect the retained
fact counts rather than comparing scores as if the query set were unchanged.
For small datasets, `--proportional-splits` keeps query counts near 80/10/10
while maintaining global 50/50 class balance. An odd dev/test size necessarily
has one extra class; this extra alternates between dev and test.

Run this offline workflow from an isolated working directory, using absolute
paths (replace `/path/to/MMDD` with the repository location):

```bash
cd /tmp
conda run -n MMDD python /path/to/MMDD/src/rebuild_abebooks_sources.py \
  --dataset /path/to/MMDD/dataset/abebooks_joinability_no4_disjoint_20260930 \
  --output /path/to/MMDD/dataset/abebooks_source_bibliographic
```

`run_abebooks_source_experiment.py prepare --dataset <versioned-dataset>`
prepares a new experiment directory and records its source reconstruction.
`prepare_abebooks_ablation_features.py encode-tables` encodes its changed tables;
`compose-reference --reference <previous-run>` can reuse frozen evidence features
only after exact equality checks on encoder inputs. When text changes,
`encode-changed-text --reference <previous-run>` encodes modified text in both
retrieval and Teacher tiers; `compose-reference` then replaces its vectors and
content tokens while reusing only unchanged evidence. For a split-only rebuild,
`reuse-identical-tables --reference <previous-experiment-root>` also reuses table
tensors after checking complete input equality apart from object IDs, updates
input fingerprints, and writes `TABLE_INPUT_PARITY.json`. PCA and all learned
models are still fitted afresh. `curate-reference-hubs`
creates a separate evidence-filter control from train-only popularity counts,
protecting every gold content alias and retaining identical tables and qrels.
`curate-reference-text-hubs` applies the same 10%-of-training-queries threshold
only to text content classes, retaining every image and all labelled aliases.
`curate-reference-unlabelled-text` is a stronger catalog ablation: it retains
all images and only text classes with existing recovery annotations. It uses
all-split labels and must not be presented as an unlabeled deployment filter.
`curate-reference-duplicate-book-images` compares covers only within identical
title/author/publisher/year records. It removes unlabelled near copies whose
complete-image RGB pixels differ by at most 3/255 on average after resizing to
128x128, protecting every labelled content class. This is approximate image
curation; original images, queries, targets and recovery facts remain intact.
`curate-reference-supported-sources` retains whole evidence groups from sources
with approved facts. `curate-reference-bibliographic-evidence` instead excludes
seller descriptions and sales/shipping policies by their source provenance,
retaining synopses, author descriptions, covers and all labelled content aliases.
Both write independent views and preserve every query, target, split and fact.
`curate-reference-title-anchored-text` requires text to mention two of the three
rarest source-title words occurring in at most 1% of book rows (one word when
only one distinctive anchor exists). It retains all images and labelled aliases.
`curate-reference-positive-sources` keeps whole evidence groups for sources with
any existing explicit or implicit positive qrel, plus labelled aliases. This
uses all-split annotations for dataset curation, not an unlabeled deployment rule.
`audit_abebooks_evidence_catalog.py` compares these catalogs on dev with a fixed
checkpoint; its output is a retrieval diagnostic, not a newly trained result.
`run_abebooks_source_experiment.py train/dev/test` uses the existing CQET method
with 10 epochs and batch 16, and allows dev evaluation before opening test.
For controlled optimizer comparisons, `--reference-inputs` reuses an unchanged
dataset and frozen features in a new run, and `--student-lr-p`/`--student-lr-r`
override the protocol recipe's learning rates (defaults 1e-4 / 1e-3).
`summarize_abebooks_source_experiment.py` checks source/fact/split integrity and
independently recomputes per-query and aggregate recall. Its primary metric is
selected KD multimodal RRF Recall@10; Teacher scores remain diagnostics.
The current numerical gate requires test overall Recall@10 >= 0.40 and implicit
Recall@10 >= 0.10 simultaneously. Data integrity and construction constraints
must also be checked before accepting an experiment.
`audit_abebooks_recall_mechanism.py --run-root <experiment>` audits every
implicit dev/test query for literal visibility of the known recovered values
and retained known witnesses. Its extra metrics do not change qrels, model
selection, or the main Recall denominator. They are evidence diagnostics,
not new extraction labels or verification of generated values.

## Stage-1 main flow (`mmdd_stage1`, protocol 4.2.0)

`src/run_stage1.py` runs Stage 1 end to end, from a dataset artifact to the Stage-2
handoff. `src/mmdd_stage1/` holds the implementation: dataset construction, the
frozen-feature layers, the CQET Teacher/Student training kernels, ANN retrieval,
dev selection, frozen evaluation, and export. A run is fully described by
`<run>/protocol.json`. `init` writes it from `configs/mmdd_stage1_cqet_protocol.json`
and binds the dataset, backbone, feature directory, run layout, and GPU. The protocol
must carry `version = "4.2.0"`.

The frozen feature layer is **not** part of a run. It lives in its own directory
(`--features-dir`, default `work/stage1_features/<dataset name>`) so that any number
of runs — different seeds, recipes, or ablations — train on one encoding:

```
<features-dir>/data/      stage1_objects, edge_lists, target_lists, stage1_corpus  (build-data)
<features-dir>/encoder/   two-tier Qwen cache merged from two shards              (encode)
<features-dir>/features/  packed retrieval z and content tokens                    (encode / pack)
```

`build-data` and `encode` refuse to overwrite a feature directory that is already
populated. `work/stage1_features/entitables/` links the existing EntiTables encoding
in this layout; `init --features-dir work/stage1_features/entitables` reuses it and
`build-data` / `encode` are skipped (`init` prints which case applies).

```bash
RUN=work/stage1_entitables
S="conda run -n MMDD python src/run_stage1.py"
$S init --run-root $RUN --dataset-root <dataset-artifact> --gpu 0   # --features-dir DIR, --seeds 13 17 ..., --side-gpu 1
$S build-data      --run-root $RUN          # only for a new feature directory
$S encode          --run-root $RUN          # only for a new feature directory; --gpus 0 1 runs the shards in parallel
$S lock            --run-root $RUN
$S import-teacher  --run-root $RUN --from-run <completed run> --reason TEXT   # optional: reuse its Teacher chain
$S verify-features --run-root $RUN
$S prepare         --run-root $RUN
$S validate        --run-root $RUN
$S smoke           --run-root $RUN
$S train           --run-root $RUN
$S train-side      --run-root $RUN          # only with --side-gpu: start next to train, same run root
$S export          --run-root $RUN          # exports the KD Student (protocol primary); --arm SUP for the SUP control
```

Inputs and outputs (`F` = feature directory, everything else under `<run>`):

| step | reads | writes |
|---|---|---|
| `build-data` | dataset artifact | `F/data/{stage1_objects,edge_lists,target_lists,stage1_corpus}.jsonl` (targets keep 20 rows) |
| `encode` | `F/data/`, backbone | `F/encoder/` (two-tier Qwen cache, merged from two shards), `F/content_shard*/`, `F/features/{z,content}`; logs under `<run>/logs/` |
| `lock` | dataset, backbone, `F/features/` | `DATASET_IDENTITY.json`, `FROZEN_RECIPE_LOCK.json`, `CONTENT_ALIASES.jsonl.gz`, `CACHE_MANIFEST.jsonl` |
| `import-teacher` | a completed run with the same dataset, feature cache, seeds and `teacher` / `retrieval` / `feature_provenance` blocks | copies of its `seed<N>/training_records/raw_train`, `seed<N>/eval/dev/raw` pool bundles, `TA/TB_SHARED/C1_*` lists and `TA`, `TB_CQET`, `TB_QT` stage directories; `SOURCE_AMENDMENTS.jsonl` carrying those stages from the source run's code; `IMPORTED_TEACHER_CHAIN.json` |
| `verify-features` | 16 objects per modality | `tests/real_tensor_probes/feature_provenance.json`; fails closed on any mismatch |
| `prepare` | train qrels/recoveries | `labels/`, `rows/`, `pca/` (train queries + lake + evidence only) |
| `validate` | | runs `tests/test_stage1_cqet.py` and `tests/test_stage1_reference_contracts.py`, bound to the source hash |
| `smoke` | 8 train queries | `tests/smoke/` |
| `train` | everything above | `seed<N>/<STAGE>/checkpoints`, `seed<N>/selections/`, `GLOBAL_SELECTION_FREEZE.json`, `seed<N>/eval/{dev,test}/`, `seed<N>/timing/stages.jsonl`, `reports/{DECISION.json,RESULTS.md}` |
| `train-side` | as `train` | `seed<N>/{TB_QT,QT_C1_SUP,QT_C2_SUP}/`, `seed<N>/selections/QT_C{1,2}.json`, the Teacher trajectory, `seed<N>/eval/test/`; process states in `processes/{main,side}.json` |
| `export` | frozen selection | `stage2_handoff/retrieval.{train,dev,test}.jsonl`, `stage1_gate.json` |

The encoder always uses two shards: `verify-features` replays the original
batches, which assume `stage1_objects` position modulo 2 and content shards by
object-ID hash. Encoder prompts are `cache_stage1_features.EMBEDDING_INSTRUCTIONS`.
GPU commands set `CUDA_VISIBLE_DEVICES` to the protocol's `hardware.uuid` (`train-side`:
`hardware.side_uuid`) before importing torch, and the pipeline checks that UUID again.

**Two GPUs.** `init --side-gpu M` pins a second GPU of the same model. `train` then keeps the
Native critical path (Raw pools → TA → `TB_CQET` → Native C1 → C2 graph → Native C2 SUP/KD →
dev evaluation) and `train-side` runs everything off it (`pipeline.qt_and_trajectory_work`):
`TB_QT` as soon as TA ends, `QT_C1_SUP` and its selection, the Teacher trajectory, `QT_C2_SUP`
once the shared C2 graph is on disk, and the test split after the global freeze. The two
processes share the run root and wait for each other's stage receipts and files; each fails as
soon as the other one died (`processes/{main,side}.json`). Restart both with the same commands:
completed stages and selections are reused. Every stage seeds itself, so the split does not
change any stage's inputs; each stage's PRE_RUN receipt names the GPU it ran on. Without
`--side-gpu`, `train` runs the same work in-process after the Native C2 selection.

Training stages per seed, in order (`pipeline.train_seed`):

1. Raw pools from frozen Qwen features (`lists.build_raw_pools_split`), then the
   TA / TB_SHARED / C1 training lists.
2. `TA` — fresh Teacher, `teacher.TA.epochs` epochs. `TB_CQET` and `TB_QT` (the
   evidence-free control) each train for one epoch from TA's endpoint on the candidate list
   `RawU ∪ RawDirect150 ∪ G ∪ U32`. Epochs, learning rate, weight decay, batch
   size, and support weight come from `teacher.TA` / `teacher.TB`.
3. `NATIVE_C1_SUP`, `QT_C1_SUP` — Students initialized from PCA/identity and trained
   on the C1 edge lists. The C1 endpoint is selected on dev (`student.native_selection`).
4. The selected C1 Student retrieves the train queries. Its pool, the Raw pool, and
   the gold targets form the shared C2 graph (`lists.build_c2_shared_graph`). The
   frozen `TB_CQET` endpoint scores every graph list once (`teacher_logits_cache`).
5. `NATIVE_C2_SUP` and `NATIVE_C2_KD` start from the selected Native C1, `QT_C2_SUP`
   from the selected QT C1; all three train on the shared C2 graph, so the arms differ in
   the model, not in the mined lists. The KD arm uses the SUP arm's selected fraction. The frozen dev/test evaluation
   follows. Test labels are exported only after `GLOBAL_SELECTION_FREEZE.json`. Every
   ranking is recomputed from raw IDs by `mmdd_stage1.independent_metrics`.

Student recipe (`train.StudentRecipe`, read from `protocol["student"]`). It is
written into every stage receipt and training-log row so a run can be audited post hoc:

| key | value | why |
|---|---|---|
| `P_lr` / `R_lr` | 1e-4 / 1e-3 (C2) | at 1e-6 / 1e-5 the Student never left its PCA/identity initialisation |
| `C1.P_lr` / `C1.R_lr` | 1e-5 / 1e-4 | a stage block may override the shared rates (`StudentRecipe.from_protocol(protocol, stage="C1")`). At the shared 1e-4 / 1e-3 the C1 edge lists (64 candidates, 32 Raw-space hard negatives) raised dev D150 coverage by 12 pp but cut E-pool coverage from 0.68 to 0.33 within 120 steps, so the C150-first selection fell back to fraction 0 (`work/stage1_entitables_abcd_s13`) |
| `logit_scale` | 20 | bilinear scores are cosine-sized; without the scale every softmax loss sits at `log N` |
| `temperature` | 10 | teacher logits span ~40; at τ=1 the KD target is one-hot and KD degenerates to SUP |
| `kd_weight` | 1.0 | KD-only and SUP+KD both beat SUP by 3–5 pp dev R@10 once τ is matched |
| `random_negatives` | 256 | uniform targets appended to every C2 direct list; without them a rank-one drift along the mean target direction destroys global retrieval |
| `anchor_weight` | 0 | the elementwise MSE anchor was ~1e-6 and did nothing; drift is logged instead as `sigma1/sigma2_R_QT_minus_I` |
| `lr_schedule` | `cosine` | LR decays to 0 over each Student stage (`C2.epochs = 3`); in the probe KD peaked mid-run and fell back under a constant LR |
| `kd_normalization` | `temperature` | `zscore` standardises each teacher list before dividing by `temperature`, removing the dependence on the teacher's logit range |
| `teacher_scored_negatives` | true | the random negatives are fixed per query (`train.c2_training_row`, epoch-independent), the frozen TB_CQET scores them in `teacher_logits_cache`, and KD covers the same lists as SUP. Every C2 arm trains on the same extended lists |
| `kd_top_k` | 0 | full-list KL. `k > 0` restricts the KL to the teacher's top-k and adds a rank-mass term that ranks the rest below them (`losses.top_k_list_kd`); with `k = 50` that term (≈ 49 non-gold candidates treated as positives, loss stuck at ≈ 1.1) cost 4–5 pp dev Direct R@10 against the full-list KL in a direct-only probe on `work/stage1_entitables_abcd_s13` and turned KD − SUP negative |
| `evidence_random_negatives` | 256 | that many random negatives get a one-path bag with a random canonical evidence object, so they also enter the evidence list and anchor `Q_text/Q_image/text_T/image_T` the way the direct negatives anchor `QT` |

The first rows are the probe-validated fix (plans A/B in the diagnosis). In the direct-only probe on
`work/stage1_entitables_abcd_s13` (`work/stage1_entitables_abcd_s13_probe/`), `teacher_scored_negatives` with the
cosine schedule and the full-list KL gave SUP+KD 0.470 vs SUP 0.436 dev R@10 at the selected fraction; `kd_top_k = 50`
alone reversed that (plan C.2, off since). `evidence_random_negatives` (plan D) is still unablated.
Each one is a protocol switch, so it can be ablated against its "off" value (`false` / `0` / `constant`).
`teacher_logits_cache/scores.pt` holds `{query_id: {direct, evidence, list_sha256}}`. `train_student_c2`
refuses a cache whose `list_sha256` differs from the lists it would train on.

The frozen evaluation also reports, without gating, the evidence narrative per split:
`SUMMARY.json → narrative` (per Student and segment: own `Direct_ANN_R10`, `E_target_coverage`,
`C150_target_coverage`, and TB_CQET `f0/Real/Swap` R@10). It also adds bootstrap contrasts for `Real − f0`
(overall and implicit), KD − SUP on implicit queries, the Student's own direct R@10, and E-pool coverage.
These are listed in a second table in `reports/RESULTS.md`.

Execution notes (none of these change a loss, a list or a checkpoint format):

- Raw and Student pools share one retrieval kernel (`retrieval.build_pools`): Raw is the
  same two-hop → D1 → P3 procedure with the frozen `z` in place of the Student's
  projected relation vectors. Each Student retrieval pass first builds three HNSW
  indices single-threaded (protocol `hnsw.threads = 1`, ≈3 min on EntiTables); the
  per-query work after that is ≈0.15 s.
- The Teacher scores a candidate list as one batched relation/global-feature
  computation per chunk (`FreshPathTeacher.score_pairs/score_triplets` over a
  pre-encoded, role-tagged cache); the content store keeps every chunk memory-mapped.
  On a 4090 a TA logical batch of 8 queries takes ≈1.3 s (was ≈5.9 s), TB ≈2.3 s
  (was ≈5.2 s), with losses bit-identical and gradients within float32 rounding.
- The C2 kernel projects every distinct object of a query microbatch once
  (`train._student_c2_batch_scores`) and scores paths with index gathers instead of a
  per-path Python loop. On a 4090 a 64-query step with ~1000 candidates per query takes
  ≈0.19 s (was ≈0.54 s). KD reads the frozen `teacher_logits_cache` only. The diagnosis behind the recipe
  is in `docs/entitables_kd_distillation_diagnosis_20261001.zh-CN.md`.

After a code change, completed stages are reused only if their recorded source
hash still matches. Run `amend-source --amendment-id ID --carry STAGE ... --reason
TEXT` before `prepare` to carry stages across an execution-only change.

**Reusing a Teacher chain.** Everything upstream of the Students (Raw pools, the TA /
TB_SHARED / C1 lists, `TA`, `TB_CQET`, `TB_QT`) depends only on the dataset, the frozen
features, the seed and the protocol's `teacher` and `retrieval` blocks. A run that changes
only the Student recipe can therefore take that chain from a completed run instead of
spending ~3.5 h retraining it: `import-teacher --from-run OLD --reason TEXT`, after `lock`
and before `prepare`. It refuses when the dataset or cache identity, those protocol blocks
or the seeds differ, copies the files, re-hashes every copy against the source receipts
and pool manifests, records a source amendment carrying the three stages from the source
run's code, and runs `train`'s own reuse check on the copies. Receipts are copied byte for
byte, so their `outputs[*].path` still name the source run; reuse checks resolve outputs
inside the stage directory, not at that recorded path. The Teacher trajectory, the test
Raw pools and every Student stage are recomputed in the new run.

`export` writes one record per query with `results[].{target_id, score,
direct_score, evidence_score, stage2_table_score, paths}`. Target order and table
score come from the frozen `TB_CQET` endpoint's reranking of the selected Student's
C150 pool. Direct paths mark targets in the Student's direct ANN top 100. Evidence
paths keep the first `--evidence-path-k` retained `Q -> E -> T` paths. Train
retrieval is computed once under `stage2_handoff/native_<arm>_train_retrieval/`;
dev/test reuse the frozen evaluation pools. `run_stage2.py` reads these files
(`--stage1-handoff`); `validate_stage2_gate` rejects any retrieval record not
produced by the gated checkpoint.

`cache_stage1_features.py` remains the frozen encoder used by `encode`.

The AbeBooks drivers (`run_abebooks_fresh.py`, `run_abebooks_data_ablation.py`,
`run_abebooks_source_experiment.py`) first copy the dataset into a filtered
`<run>/dataset_view/`, encode it into `work/stage1_features/<run name>/`, and then
call `run_stage1.build_protocol` / `build_data` / `encode` / `pack` and the same
`mmdd_stage1` training functions. They take `--gpu N` (physical index, PCI bus order)
and bind that GPU into the protocol the same way `init` does; nothing is hardcoded.
`prepare_abebooks_ablation_features.py` recomposes per-arm features inside each
ablation run and writes those locations into the arm's protocol explicitly.

Stage-1 execution reuses C1 object projections within each microbatch (including
their gradients), as C2 already does. Projections are rebuilt after every update.
Teacher sequence packing keeps relation codes on the CPU to avoid copying them
back from the GPU. Retrieval sorts each exact score vector once and caches both
the second-hop hits and their exact audit per evidence, within one frozen-model
invocation. Candidate budgets, stable tie rules, losses and optimizer schedules
are unchanged. Batched matrix multiplication can introduce float32 roundoff;
these optimizations do not promise bit-identical training trajectories.

## Stage-2 main flow (`mmdd_stage2`, B+IDF)

`src/run_stage2.py` reranks each query's Stage-1 **C30** with recovered bridge
attributes and visible-column evidence. Stage 2 reads only the top 30 Stage-1
targets: selector supervision, reader features, recovery and both scorers work on
C30. Stage-1 ranks 31–50 are appended unchanged, so @50 metrics stay comparable.
Every command takes `--run-root R`; `init` writes `R/config.json` from
`configs/mmdd_stage2_bidf.json`.

| Command | Device | Writes |
|---|---|---|
| `init --stage1-handoff H --dataset-root D [--no-selector-evidence]` | – | `config.json` |
| `catalog` | CPU | `catalog.sqlite`, `population/<split>.json` |
| `jobs` | CPU | `jobs/{train,dev,test}.jsonl`, `jobs/train_labels.json`, `jobs/FUNNEL.json` |
| `features --split S --gpu N` | GPU | `features/<split>/view<v>/<pair>.npz` |
| `train-head` | CPU | `head/head.pt`, `head/history.json` |
| `plans` | CPU | `plans/{dev,test}.jsonl` |
| `recover --gpu N` | GPU | `recovery/<split>/<query>.json` |
| `score` | CPU | `matching/` (MiniLM vectors), `scores/<split>.jsonl` |
| `evaluate` | CPU | `evaluation/{METRICS,PER_QUERY,CONTRASTS}.csv` |

`H` is the `run_stage1.py export` directory (`retrieval.<split>.jsonl`; per
target the table score and up to four retained evidence ids). Feature extraction
and recovery skip outputs that already exist, so an interrupted command resumes.

1. **Jobs** (`jobs.py`). Each pair is (query, C30 target, that target's natural
   evidence bag). Train labels are the hidden column of `model_recoverable_join_column`
   qrels, mapped through `source_column_index`. A labeled target outside C30 is
   dropped. The fit/holdout split hashes the source group. Dev/test jobs cover all
   30 targets and never read qrels.
2. **Selector** (`reader.py`, `selector.py`). Frozen Qwen3.5-9B reads query table,
   anonymized evidence and the candidate table. Each target column is restated at
   the end inside `<|object_ref_start|>…<|object_ref_end|>`, and the hidden states
   at those two markers are the column feature. A fresh MLP head (8192→256→1) is
   trained with the multi-positive softmax loss for a fixed 20 epochs. Train views
   use column permutations 13001/26002, alternating by epoch. The holdout only
   monitors training. `--no-selector-evidence` is the No-E ablation (arm A).
3. **Plan.** `pair_score = log_softmax_C30(Stage-1 score) + log_softmax_columns(head)`.
   Up to 10 pairs are taken, at most 3 per target. Pairs with the same column name
   and the same evidence bag form one recovery view.
4. **Recovery** (`recovery.py`, `localizer.py`). For every view and every query row
   without a visible value for that attribute, the 9B model answers one ROW1
   request: a JSON string or null. Rows still without a VALUE then retry each image
   alone. Text evidence that names only *other* rows' entities is gated out.
   Images are shown as the original plus, when RAEA-Attr+ConsensusMask (v_proj
   layers 15/23/27) finds a dense local ROI, a context crop and a tight zoom.
   VALUE claims are grouped by attribute into five row slots; disagreeing claims
   become CONFLICT. A token-limit hit sends the whole query back to Stage 1.
5. **Scoring** (`matching.py`, `visible.py`, `rerank.py`). Cells match on the typed
   key. TEXT values may also match when they share numeric tokens and have
   MiniLM cosine ≥ 0.98.
   - `bridge(T)` = max over same-attribute columns of matched recovered rows / 5.
   - `vis_row(T)` = max over (eligible query column, target column) of matched rows / 5.
   - `vis_idf(T)` weights each row by `w = log((K+1)/(df+1))/log(K+1)`, where df
     counts the C30 candidates containing the value and K = 30.
   - The B+IDF tiers: bridge > 0 by bridge score, then vis_idf > 0 by
     (vis_idf, vis_row), then Stage 1. The tiered order is RRF-fused (k = 60) with Stage 1.

`evaluate` reports `STAGE1`, `BRIDGE_RRF60` (bridges only), `VISIBLE_IDF_RRF60`
(no recovered evidence) and `BIDF_RRF60`. It covers dev/test × overall/implicit/explicit,
with Recall/Precision/nDCG and a paired source-group bootstrap. The
`BIDF − VISIBLE_IDF` contrast isolates what recovered evidence adds over visible
columns.

```bash
python src/run_stage2.py init --run-root work/stage2_entitables \
  --stage1-handoff work/stage1_entitables/stage2_handoff --dataset-root <dataset>
python src/run_stage2.py catalog --run-root work/stage2_entitables
python src/run_stage2.py jobs --run-root work/stage2_entitables
for s in train dev test; do python src/run_stage2.py features --split $s --gpu 1 --run-root work/stage2_entitables; done
python src/run_stage2.py train-head --run-root work/stage2_entitables
python src/run_stage2.py plans --run-root work/stage2_entitables
python src/run_stage2.py recover --gpu 1 --run-root work/stage2_entitables
python src/run_stage2.py score --run-root work/stage2_entitables
python src/run_stage2.py evaluate --run-root work/stage2_entitables
```

This is the R7 fresh-selector B arm (`audit/MMDD_R7_FRESH_AB_C30_D098_PACKAGE`) plus
the LEXICO IDF scorer, merged and moved to C30. The earlier RATA/FOCUS verifier
and the r4/r4c/r5b/r12/r25/r26/column-R1/R2 experiment code were removed. They
can be restored from git `3649483`.

## Shared joinability construction

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
