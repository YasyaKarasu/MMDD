# EntiTables Multimodal Table Dataset Builder

This repository contains a builder for creating a multimodal table data lake and a query workload from EntiTables-style JSON files.

The goal is dataset construction only: source tables, projected query views, entity mentions, optional Wikipedia text/image metadata, table-to-asset links, and source-level splits.

It does not construct joinability benchmark labels. It does not generate positive pairs, negative pairs, joinable/not-joinable fields, augmentation targets, or query-target pairs.

## Input Format

Each EntiTables JSON file is expected to be a top-level dictionary:

```json
{
  "table-0001-590": {
    "title": ["Column A", "Column B"],
    "numCols": 2,
    "numericColumns": [],
    "pgTitle": "Wikipedia page title",
    "numDataRows": 4,
    "secondTitle": "Section title",
    "numHeaderRows": 1,
    "caption": "Table caption",
    "data": [
      ["[Page_Title|Display Text]", "plain text"],
      ["[Another_Page]", "123"]
    ]
  }
}
```

Cells may contain Wikipedia links in either `[Page_Title|Display Text]` or `[Page_Title]` form. Plain text cells are also supported.

## Outputs

The script writes these artifacts to `--output_dir`:

- `source_tables/part-*.jsonl`: cleaned and standardized source tables, including normalized columns, parsed cells, column profiles, and candidate entity columns.
- `query_views/part-*.jsonl`: query workload tables produced by projecting one source table into one or more smaller views.
- `entities/part-*.jsonl`: unique Wikipedia entities found in source tables and query views, with cell-level appearances.
- `bridge_assets/part-*.jsonl`: optional Wikipedia text extracts and downloaded image assets with source URLs and metadata.
- `table_asset_links/part-*.jsonl`: links from entity cells in source tables/query views to entity assets. Empty `asset_ids` are retained when assets are unavailable.
- `splits.json`: train/dev/test split at source-table or page-title level.
- `stats.json`: processing counts, skipped reasons, asset counts, and workload statistics.
- `dataset_manifest.json`: the shard manifest listing every artifact directory, part file, and record count.

Large JSONL outputs are streamed and sharded. The builder writes and flushes part files incrementally instead of keeping the full dataset in memory until the end.

## Query Views

`query_views` are part of the dataset and represent query workload tables projected from source tables. A single source table may produce multiple query views.

Supported view strategies:

- `entity_plus_one_attr`
- `entity_plus_two_attrs`
- `entity_only`
- `random_projection`

These views are not positive samples, negative samples, labels, or training pairs. They are query tables for later system experiments.

## Hidden Columns

`hidden_column_indices` and `hidden_column_names` only record which original source-table columns were omitted when constructing a projected query view.

They are provenance fields only. They are not augmentation ground truth, not target columns, not positive labels, and not joinability labels.

## Splits

Splits are source-level splits. With `--split_by page_title`, all source tables from the same Wikipedia page are kept in the same split when possible. With `--split_by source_table_id`, all query views derived from a source table stay with that source table.

This avoids leakage where different projected views from the same source table appear in different train/dev/test partitions.

## Setup

Use the existing conda environment:

```bash
conda run -n MMDD python -m pip install -r requirements.txt
```

`hnswlib` is the default ANN backend for stage-1 retrieval. `pytest` is used for tests. `transformers` and `qwen-vl-utils` are used by the local Qwen3-VL-Embedding-2B encoder wrapper.

## Usage

Build the table dataset and query workload with Wikipedia metadata and downloaded images:

```bash
conda run -n MMDD python scripts/build_mm_table_dataset.py \
  --input_dir dataset/tables_redi2_1 \
  --output_dir output \
  --max_tables 100 \
  --max_query_views_per_source_table 5 \
  --max_entities 1000 \
  --max_images_per_entity 3 \
  --wiki_link_threshold 0.3 \
  --flush_every_records 500 \
  --records_per_shard 50000 \
  --split_by page_title
```

Build only tables, query views, entities, and entity links without calling Wikipedia:

```bash
conda run -n MMDD python scripts/build_mm_table_dataset.py \
  --input_dir dataset/tables_redi2_1 \
  --output_dir output_no_wikipedia \
  --max_tables 100 \
  --max_query_views_per_source_table 5 \
  --wiki_link_threshold 0.3 \
  --flush_every_records 500 \
  --records_per_shard 50000 \
  --split_by page_title \
  --no_wikipedia
```

When `--no_wikipedia` is set, `bridge_assets.jsonl` is empty, but `entities.jsonl` and `table_asset_links.jsonl` are still produced. Entity links remain useful because `asset_ids` are simply empty lists.

## Image Downloads

When Wikipedia fetching is enabled, image assets are downloaded to:

```text
<output_dir>/images/
```

Each image record in `bridge_assets/part-*.jsonl` keeps both provenance and local file information:

- `image_url`: original Wikipedia/Commons image URL.
- `description_url`: image description page URL.
- `local_path`: absolute local path to the downloaded image.
- `relative_path`: path relative to `output_dir`.
- `file_name`: downloaded image file name.
- `bytes`: downloaded file size.
- `sha256`: file digest.
- `metadata`: MediaWiki `imageinfo` metadata, including `extmetadata`.

The script still filters obvious low-value files such as icons, logos, flags, and placeholders before downloading.

## Streaming Writes

Use `--flush_every_records` to control how often JSONL writers flush buffered records to disk. The default is `500`.

Use `--records_per_shard` to control how many JSONL records go into each part file before the writer starts the next shard. The default is `50000`.

This applies to the large streaming outputs:

- `source_tables/part-*.jsonl`
- `query_views/part-*.jsonl`
- `entities/part-*.jsonl`
- `bridge_assets/part-*.jsonl`
- `table_asset_links/part-*.jsonl`

Read `dataset_manifest.json` to discover the current shard set. This is safer than globbing when reusing an output directory, because stale part files from an older run may still exist.

`splits.json` and `stats.json` are written as single JSON files after the needed indexes have been finalized.

## Important Non-Goals

This builder intentionally does not generate:

- `original_joinable`
- `bridged_joinable`
- `negative_type`
- `positive_pairs`
- `negative_pairs`
- `labels`
- query-target pairs
- joinability labels
- augmentation ground truth

The output is a multimodal table dataset plus query workload, not a joinability benchmark label generator.

## EntiTables inference on local and remote GPUs

`build_mm_joinability_dataset.py` can run local and remote OpenAI-compatible
vLLM endpoints at the same time. Local and remote requests use independent
worker limits. Within each modality, both worker groups consume one pending
task queue, so a faster remote GPU naturally completes a larger share without
raising the local GPU concurrency.

The existing endpoint and worker options remain the local pool. Configure the
remote pool separately:

```bash
export MMDD_REMOTE_TEXT_MODEL_API_KEY="..."   # optional
export MMDD_REMOTE_IMAGE_MODEL_API_KEY="..."  # optional
conda run --no-capture-output -n MMDD python \
  scripts/build_mm_joinability_dataset.py \
  --input_dir dataset/tables_redi2_1 \
  --output_dir output_mm_joinability \
  --text_model_base_url http://127.0.0.1:8001/v1 \
  --image_model_base_url http://127.0.0.1:8000/v1 \
  --text_model_workers 2 \
  --image_model_workers 1 \
  --remote_text_model_base_url http://127.0.0.1:18001/v1 \
  --remote_image_model_base_url http://127.0.0.1:18000/v1 \
  --remote_text_model_workers 12 \
  --remote_image_model_workers 8
```

The example assumes SSH tunnels expose the remote services on ports 18001 and
18000. The remote services must expose the same served model names selected by
`--text_model_name` and `--image_model_name`. Repeat remote endpoints with
`--remote_text_model_base_urls` / `--remote_image_model_base_urls`, or use the
corresponding `*_base_urls_file` options for dynamic endpoint lists. A remote
worker count of zero disables that remote modality. Remote API keys fall back
to `VLLM_API_KEY` and then the matching local modality key when a dedicated key
is not supplied.

### Explicit visible-join queries

The joinability builder can promote rejected multimodal tables into ordinary
visible-join queries. The default `ratio` mode uses a seeded 0.2 fraction and
keeps the historical one-query-per-source behavior. Use
`--explicit_join_fallback_mode match_implicit` when explicit query counts must
match implicit query counts independently in the train, dev, and test splits.
That mode works at query level: one source table may contribute multiple
explicit candidates, one per viable visible join column. Candidate join columns
are target-only across all sibling variants, and ordinary context columns are
partitioned once per source table, so a sibling query cannot join to another
sibling target through a context column.

The resulting `stats.json` reports both candidate source-table counts and
candidate query counts under `explicit_join_candidate_*`.

### WDC-priority borrowing of the remote GPUs

When the remote vLLM services are managed by the layout control agent,
EntiTables can borrow the same GPUs while WDC has no model work. Local
EntiTables workers never wait for the remote lease: they keep consuming the
shared queue while WDC is active, and remote workers join after WDC releases
it. Before WDC starts another model stage, EntiTables withdraws the remote
routes, drains in-flight requests, releases the lease, and acknowledges WDC's
priority request.

Both processes must use the same `--remote_layout_coordination_dir`. Add these
options to the WDC command:

```bash
--remote_layout_control_url http://127.0.0.1:18999 \
--remote_layout_control_token_file layout-control-token \
--remote_layout_primary_image_url http://127.0.0.1:18000/v1 \
--remote_layout_switchable_url http://127.0.0.1:18001/v1 \
--remote_layout_coordination_dir work_gpu_priority/remote
```

Run EntiTables through its dynamic local-GPU runner with the same remote
control and coordination settings:

```bash
conda run --no-capture-output -n MMDD python \
  scripts/run_mm_joinability_dynamic_vllm.py \
  --input_dir dataset/tables_redi2_1 \
  --output_dir output_mm_joinability \
  --text_model_path hf_models/Qwen3.5-9B \
  --image_model_path hf_models/Qwen3-VL-8B-Thinking \
  --gpu_coordination_dir work_gpu_priority/local \
  --remote_layout_control_url http://127.0.0.1:18999 \
  --remote_layout_control_token_file layout-control-token \
  --remote_layout_primary_image_url http://127.0.0.1:18000/v1 \
  --remote_layout_switchable_url http://127.0.0.1:18001/v1 \
  --remote_layout_coordination_dir work_gpu_priority/remote \
  --remote_text_model_workers 32 \
  --remote_image_model_workers 64
```

Remote text and image concurrency are independent totals. In this example the
switchable endpoint supplies 32 text slots, while the `image_burst` layout
supplies 32 image slots on each of two endpoints, for 64 image workers total.

The control token file must have mode `0600` and cannot be `.env.openai`.
Inference keys still come from `MMDD_REMOTE_TEXT_MODEL_API_KEY`,
`MMDD_REMOTE_IMAGE_MODEL_API_KEY`, or `VLLM_API_KEY`. A stale
`priority_requested` left by a failed WDC process keeps EntiTables remote
routing disabled until a later WDC run publishes `borrowable`; this is the
intentional fail-closed behavior.

## EntiTables Joinability with OpenAI

`build_mm_joinability_dataset_openai.py` runs the existing EntiTables
joinability builder with one non-streaming OpenAI Chat Completions model for
both text and image attribute extraction. It reuses the same extraction prompt, parser,
sampling, recovery rules, caches, and canonical output format as
`build_mm_joinability_dataset.py`.

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional
conda run --no-capture-output -n MMDD python \
  scripts/build_mm_joinability_dataset_openai.py \
  --input_dir dataset/tables_redi2_1 \
  --output_dir output_mm_joinability_openai \
  --cache_dir cache/mm_joinability \
  --openai_model gpt-5.6 \
  --openai_reasoning_effort none \
  --text_model_workers 4 \
  --image_model_workers 4 \
  --openai_max_inflight 4
```

Instead of exporting secrets in an interactive shell, create a dotenv file
that is already ignored by this repository:

```dotenv
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
```

Set its permissions to `0600`. Both OpenAI builders automatically load
`./.env.openai` when it exists; use
`--openai_env_file /path/to/another.env` to select a different file. The
loader accepts only those two variables and never executes the file as shell
code. File values override stale values inherited from the shell; an explicit
`--openai_base_url` still has the highest priority.

The API key is read only from `--openai_api_key_env` (default
`OPENAI_API_KEY`) and is never written to run metadata. The API base URL uses
`OPENAI_BASE_URL` when set and can be overridden by
`--openai_base_url`. Wikipedia downloads
and the shared extraction-cache file remain under `--cache_dir`. Every OpenAI
setting that can change a result is included in the provider model identity,
so changing the model, reasoning effort, output limit, image detail, or image
pixel budget cannot reuse incompatible extraction records. Fingerprinted run
configuration and cumulative token usage are stored under
`<output parent>/work_mm_joinability_openai/openai_model_runs/` by default;
override that root with `--openai_work_root`.

`--text_model_workers` and `--image_model_workers` still size the existing
per-modality task pools. `--openai_max_inflight` (default `4`) is a shared cap
across both pools. Optional `--openai_requests_per_minute` and
`--openai_tokens_per_minute` values proactively pace the shared request stream;
zero disables the corresponding limit. Transient responses use exponential
backoff, honor `Retry-After`, and share the resulting cooldown across both
modalities. EntiTables stops on an exhausted transient failure after caching
successful calls, so rerunning resumes instead of treating rate limiting as a
negative extraction result.

## WDC Schema.org 200K Joinability Pipeline

`build_wdc200k_mm_joinability_dataset.py` is the staged, resumable entry point
for the 200K-scale WDC Schema.org Table Corpus 2023 build.
`build_wdc200k_mm_joinability_dataset_openai.py` runs the same staged pipeline
with OpenAI inference and accepts the shared concurrency and rate-limit flags
described above; exhausted transient model jobs remain retryable on resume.
`build_wdc_mm_joinability_dataset.py` remains the legacy bounded builder; its
row-limited examples and in-memory behavior are not the 200K operating model.

The staged pipeline keeps three roots separate:

```text
output_wdc_200k/   canonical dataset artifacts only
work_wdc_200k/     selections, shards, durable job stores, checkpoints,
                   stage manifests, runtime markers, and progress.json
cache/wdc_200k/    reusable positive and negative network/media/model outcomes
```

Never nest or reuse these roots as one another. Only `output_dir` has the
canonical source/query/data-lake tables, entities, assets, links, extractions,
qrels, splits, errors, stats, and `dataset_manifest.json`. Source-table rows
are unbounded and every selected row is preserved. Do not pass
`--max_rows_per_source_table`: the compatibility option deliberately rejects
every value, including zero. The WDC `image` column is used only to discover
assets and is removed from table artifacts.

### Preflight, staged execution, and recovery

A read-only configuration preflight validates the input statistics archives,
path separation, runtime paths, disk reserve, and arguments without running a
stage:

```bash
conda run -n MMDD python scripts/build_wdc200k_mm_joinability_dataset.py \
  --input_dir wdc_schemaorg_2023 \
  --output_dir output_wdc_200k \
  --work_dir work_wdc_200k \
  --cache_dir cache/wdc_200k \
  --max_source_tables 200000 \
  --selection_seed 13 \
  --dry_run
```

The structural preflight writes deterministic selection and structural state
under `work_dir`, but performs no network or model work:

```bash
conda run --no-capture-output -n MMDD python \
  scripts/build_wdc200k_mm_joinability_dataset.py \
  --input_dir wdc_schemaorg_2023 \
  --output_dir output_wdc_200k \
  --work_dir work_wdc_200k \
  --cache_dir cache/wdc_200k \
  --max_source_tables 200000 \
  --selection_seed 13 \
  --web_max_retries 0 \
  --web_max_response_seconds 8 \
  --web_global_concurrency 128 \
  --web_per_host_concurrency 2 \
  --max_image_attempts_per_entity 3 \
  --max_images_per_entity 3 \
  --stop_after structural
```

Inspect `work_wdc_200k/progress.json` and the structural manifests before
continuing. Resume with the same four roots and parameters; completed,
checksummed shards and terminal URL outcomes are not replayed:

```bash
conda run --no-capture-output -n MMDD python \
  scripts/build_wdc200k_mm_joinability_dataset.py \
  --input_dir wdc_schemaorg_2023 \
  --output_dir output_wdc_200k \
  --work_dir work_wdc_200k \
  --cache_dir cache/wdc_200k \
  --max_source_tables 200000 \
  --selection_seed 13 \
  --resume \
  --text_model_base_url http://127.0.0.1:8001/v1 \
  --image_model_base_url http://127.0.0.1:8000/v1
```

Use `--stop_after` to establish a stage barrier. To intentionally rebuild a
named stage and everything downstream, use `--from_stage`, which validates
upstream state and archives replaced state rather than deleting it:

```bash
conda run --no-capture-output -n MMDD python \
  scripts/build_wdc200k_mm_joinability_dataset.py \
  --input_dir wdc_schemaorg_2023 \
  --output_dir output_wdc_200k \
  --work_dir work_wdc_200k \
  --cache_dir cache/wdc_200k \
  --max_source_tables 200000 \
  --from_stage pages \
  --stop_after pages
```

Add `--refresh_page_cache` only with `--from_stage pages` or earlier, and
`--refresh_image_cache` only with `--from_stage images` or earlier. A plain
resume reuses both successes and terminal failures.

### Real-table 100/1,000 scale gates

The full corpus contains more mandatory `top100` candidates than a 100- or
1,000-table target, so those gates must not run by merely lowering
`--max_source_tables` on the full input. Create isolated, exact-size
real-table gate inputs instead:

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
not change the formal stratified 200K selection. The helper creates filtered
production-format statistics ZIPs plus absolute source-data symlinks; it does
not copy, delete, fetch, or modify corpus data. The target must be absent or
strictly empty, and its resolved path must be outside the corpus tree (neither
root may contain the other). `scale_gate_manifest.jsonl` records absolute
sources, relative targets, and source gzip hashes;
`scale_gate_checksums.json` covers the manifest and filtered ZIPs. These are
all built and verified in a unique sibling staging directory before one atomic
no-clobber publish; malformed or duplicate catalog paths fail closed without a
partial target. These are operational scale-gate subcorpora, not the formal
200K sampling result. Use each gate input with the matching
`--max_source_tables` and fresh output/work/cache roots. The 10K structural
gate uses the full corpus directly.

### Direct endpoints, dynamic vLLM, and tmux

The resume command above is the direct runner: the operator supplies already
running OpenAI-compatible text and image endpoints. The dynamic runner instead
starts and reallocates both vLLM modalities, and its unowned options pass
through to the staged WDC 200K builder:

```bash
conda run --no-capture-output -n MMDD python \
  scripts/run_mm_joinability_dynamic_vllm.py \
  --input_dir wdc_schemaorg_2023 \
  --output_dir output_wdc_200k \
  --text_model_path /path/to/text-model \
  --image_model_path /path/to/vision-model \
  --work_dir work_wdc_200k \
  --cache_dir cache/wdc_200k \
  --max_source_tables 200000 \
  --selection_seed 13 \
  --resume
```

The dynamic runner owns endpoint files, model identity, and staged
start/ready/done markers under `work_wdc_200k/runtime`.

### Remote vLLM on one A100 80GB

This runbook serves both models on one remote A100 and lets the local builder
reach them through an SSH tunnel. The settings below are an operating starting
point, not a claim that this exact GPU allocation has been benchmarked. Keep the
builder stopped until both endpoint checks at the end of this section succeed.

On the remote host, first inspect the GPU and create a fresh Python 3.12
environment with `pip`:

```bash
ssh user@REMOTE_HOST
nvidia-smi
conda create -n vllm-023 python=3.12 pip -y
conda activate vllm-023
python --version
python -m pip --version
```

Confirm from `nvidia-smi` and the NVIDIA compatibility documentation that the
installed driver supports the CUDA 12.9 PyTorch wheel, then install the same
pinned vLLM release used locally. The command follows the
[official vLLM 0.23 GPU installation guidance](https://docs.vllm.ai/en/v0.23.0/getting_started/installation/gpu/):

```bash
python -m pip install 'vllm==0.23.0' \
  --extra-index-url https://download.pytorch.org/whl/cu129
python -c 'import torch, vllm; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)'
```

Do not subsequently install `torch` with conda: that can replace the matching
wheel selected for vLLM. Prepare the exact local model directories on the
remote host before serving. From the local repository, `rsync` is preferred:

```bash
ssh user@REMOTE_HOST 'mkdir -p /srv/mmdd/hf_models'
rsync -a --info=progress2 hf_models/Qwen3.5-9B/ \
  user@REMOTE_HOST:/srv/mmdd/hf_models/Qwen3.5-9B/
rsync -a --info=progress2 hf_models/Qwen3-VL-8B-Thinking/ \
  user@REMOTE_HOST:/srv/mmdd/hf_models/Qwen3-VL-8B-Thinking/
```

If those local directories are unavailable, downloading the corresponding
Hugging Face repositories instead requires remote network access and any model
authorization required by their publishers. Do not silently substitute a
different revision or model.

Back on the remote host, generate one API key in a mode-600 file without putting
the secret in a command argument or log. Use a dedicated tmux socket with one
session and two windows, so both windows inherit the same controlled setup and
their output stays in the panes without log redirection. Before running these
commands, stop an existing `mmdd-vllm` dedicated server with
`tmux -L mmdd-vllm kill-server` (an absent server only produces a harmless
"no server running" error), or choose a unique socket name instead:

```bash
conda activate vllm-023
umask 077
mkdir -p "$HOME/.config/mmdd"
python -c 'import secrets; print(secrets.token_urlsafe(32))' \
  > "$HOME/.config/mmdd/vllm-api-key"
chmod 600 "$HOME/.config/mmdd/vllm-api-key"

tmux -L mmdd-vllm kill-server
tmux -L mmdd-vllm new-session -d -s serve -n text \
  'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate vllm-023 && export VLLM_API_KEY="$(cat "$HOME/.config/mmdd/vllm-api-key")" && export REMOTE_MODEL_ROOT=/srv/mmdd/hf_models && exec env CUDA_VISIBLE_DEVICES=0 vllm serve "$REMOTE_MODEL_ROOT/Qwen3.5-9B" --host 127.0.0.1 --port 8001 --served-model-name Qwen3.5-9B --trust-remote-code --dtype bfloat16 --max-model-len 8192 --enforce-eager --gpu-memory-utilization 0.44 --max-num-seqs 16 --max-num-batched-tokens 8192 --language-model-only'

tmux -L mmdd-vllm new-window -d -t serve -n image \
  'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate vllm-023 && export VLLM_API_KEY="$(cat "$HOME/.config/mmdd/vllm-api-key")" && export REMOTE_MODEL_ROOT=/srv/mmdd/hf_models && exec env CUDA_VISIBLE_DEVICES=0 vllm serve "$REMOTE_MODEL_ROOT/Qwen3-VL-8B-Thinking" --host 127.0.0.1 --port 8000 --served-model-name Qwen3-VL-8B-Thinking --trust-remote-code --dtype bfloat16 --max-model-len 8192 --enforce-eager --gpu-memory-utilization 0.44 --max-num-seqs 16 --max-num-batched-tokens 8192 --limit-mm-per-prompt "{\"image\":1,\"video\":0}" --mm-processor-cache-gb 1'

tmux -L mmdd-vllm attach-session -t serve:text
# Detach with Ctrl-b d; switch to the image window with Ctrl-b n.
```

vLLM 0.23 documents `--language-model-only`; confirm it with
`vllm serve --help`. If a locally patched 0.23 build omits that option, replace
it on the text command with
`--limit-mm-per-prompt '{"image":0,"video":0}'`. The image command keeps one
image, disables video, and reduces the multimodal processor cache to 1 GiB;
vLLM documents that this cache is duplicated across API and engine processes.
See the [v0.23 serve option reference](https://docs.vllm.ai/en/v0.23.0/cli/serve/).

Keep vLLM bound to `127.0.0.1` and, on the builder host, open the recommended
tunnel:

```bash
ssh -N \
  -L 18001:127.0.0.1:8001 \
  -L 18000:127.0.0.1:8000 \
  user@REMOTE_HOST
```

For a trusted private LAN only, an alternative is `--host 0.0.0.0` plus a
firewall rule that permits these two ports solely from the builder's IP. Never
expose the vLLM ports to the public internet; vLLM's
[security guidance](https://docs.vllm.ai/en/latest/usage/security/) notes that
an API key does not protect every server endpoint.

In another local shell, read the same mode-600 key over SSH directly into the
environment, then verify both served IDs through the tunnel. The key is neither
a command argument nor printed to the terminal. Each unquoted heredoc expands
the variable into curl's stdin config; shell history contains only the literal
`$VLLM_API_KEY` text:

```bash
export VLLM_API_KEY="$(ssh user@REMOTE_HOST \
  'cat "$HOME/.config/mmdd/vllm-api-key"')"
curl --fail --silent --show-error \
  --config - http://127.0.0.1:18001/v1/models <<EOF | python -m json.tool
header = "Authorization: Bearer $VLLM_API_KEY"
EOF
curl --fail --silent --show-error \
  --config - http://127.0.0.1:18000/v1/models <<EOF | python -m json.tool
header = "Authorization: Bearer $VLLM_API_KEY"
EOF
```

The first response must contain `Qwen3.5-9B`; the second must contain
`Qwen3-VL-8B-Thinking`. Only after both are healthy, resume the stopped local
builder with the formal roots and ordinary `--resume`:

```bash
conda run --no-capture-output -n MMDD python \
  scripts/build_wdc200k_mm_joinability_dataset.py \
  --input_dir /home/oycy/MMDD/wdc_schemaorg_2023 \
  --output_dir /home/oycy/MMDD/output_wdc_200k_sampled_20260720 \
  --work_dir /home/oycy/MMDD/work_wdc_200k_eta_advisory_20260719 \
  --cache_dir /home/oycy/MMDD/cache/wdc_200k_sampled_20260720 \
  --max_source_tables 200000 \
  --selection_seed 13 \
  --sampled_entities_per_table 8 \
  --entity_sampling_seed 20260720 \
  --min_free_disk_bytes 107374182400 \
  --resume \
  --text_model_base_url http://127.0.0.1:18001/v1 \
  --text_model_name Qwen3.5-9B \
  --image_model_base_url http://127.0.0.1:18000/v1 \
  --image_model_name Qwen3-VL-8B-Thinking \
  --text_model_workers 1 \
  --image_model_workers 1 \
  --model_endpoint_ready_timeout_seconds 180 \
  --model_timeout_seconds 300 \
  --model_max_retries 4 \
  --model_retry_sleep_seconds 2
```

The endpoint client uses its existing precedence of explicit modality CLI key,
modality-specific environment key, then common `VLLM_API_KEY`. Thus the common
export above authenticates both endpoints without placing the secret on the
builder command line; keys remain excluded from run fingerprints, registries,
progress, and logs. Do not add `--from_stage models`: this is a plain resume and
does not re-fetch completed network work. The 408 connection failures have
already been restored to `pending` and will be retried. The three expired
`leased` jobs are reclaimed automatically when workers resume.

For a structural tmux preflight, keep process output attached directly to the
tmux pane and use a second pane for the atomic progress snapshot:

```bash
tmux new-session -d -s wdc_200k \
  'conda run --no-capture-output -n MMDD python scripts/build_wdc200k_mm_joinability_dataset.py --input_dir wdc_schemaorg_2023 --output_dir output_wdc_200k --work_dir work_wdc_200k --cache_dir cache/wdc_200k --max_source_tables 200000 --selection_seed 13 --stop_after structural'
tmux new-window -t wdc_200k -n progress \
  "watch -n 5 'conda run -n MMDD python -m json.tool work_wdc_200k/progress.json'"
tmux attach-session -t wdc_200k
```

Do not redirect stdout/stderr to a log file: the runner emits bounded periodic
progress directly, while durable progress lives in
`work_wdc_200k/progress.json`. Scale-gate measurements and acceptance criteria
are recorded in
`docs/superpowers/reports/2026-07-17-wdc-200k-scale-validation.md`.

## Stage-1 Logic Connectivity Pipeline

Stage-1 builds a coarse recall benchmark and training pipeline from the existing `output_medium` artifacts only. It does not refetch Wikipedia, does not rebuild `output_medium`, and does not use `query_views` as supervision.

Recommended consolidated entrypoints:

```bash
# 1) Build Stage-1 logic fragments, evidence paths, weak labels, and frozen embeddings.
conda run -n MMDD python scripts/stage1_prepare_data.py \
  --input_dir output_medium \
  --stage1_dir output_stage1_logic \
  --encoder_path ./Qwen3-VL-Embedding-2B \
  --embedding_batch_size 8 \
  --device cuda \
  --dtype bf16

# 2) Train teacher, run HITL annotation rounds, retrain teacher, and distill the student.
conda run -n MMDD python scripts/stage1_train_models.py \
  --stage1_dir output_stage1_logic \
  --hitl_rounds 3 \
  --hitl_batch_size 50 \
  --distill_loss pairwise \
  --force_retrain \
  --lan \
  --port 7860

# 3) Build ANN indexes and evaluate relation-aware multi-hop recall.
conda run -n MMDD python scripts/stage1_index_eval.py \
  --stage1_dir output_stage1_logic \
  --topk 10 50 100 \
  --max_hops 3 \
  --beam_width 64 \
  --beam_neighbors 50
```

`stage1_run_all.py` is available when you want a single command for the whole flow. It will pause during HITL rounds until labels are merged in the browser UI:

```bash
conda run -n MMDD python scripts/stage1_run_all.py \
  --input_dir output_medium \
  --stage1_dir output_stage1_logic \
  --hitl_rounds 3 \
  --hitl_batch_size 50 \
  --distill_loss pairwise \
  --force_retrain \
  --lan \
  --port 7860
```

To inspect constructed connection groups in a browser, run:

```bash
conda run -n MMDD python scripts/stage1_connection_viewer.py --stage1_dir output_stage1_logic --port 7861
```

The viewer paginates by `chain_id` and shows the visible query, hidden query, target fragment, logic pairs, qrels, and evidence paths for each group.

To inspect `build_mm_joinability_dataset.py` outputs in a browser, run:

```bash
conda run -n MMDD python scripts/mm_joinability_dataset_viewer.py --output_dir output_mm_joinability --port 7863
```

The viewer paginates through every `query_table_id -> target_table_id` pair in `qrels.jsonl` and shows the query table, target table, evidence recovery paths, and the referenced text/image material for each path. Train-time disjoint row views are grouped by `chain_id` and `row_view_index`; use the row-view filter to inspect canonical or augmented views separately. The viewer builds `<output_dir>/.mm_joinability_viewer.sqlite3` on first use and reuses it while the relevant dataset shards are unchanged, so WDC-scale datasets are browsed without retaining all tables and assets in memory. Use `--index_path` to place this generated index elsewhere.

After the new datasets finish building, launch the 1% implicit-query quality checkers with:

```bash
conda run -n MMDD python scripts/entitables_dataset_checker.py --lan
conda run -n MMDD python scripts/wdc_dataset_checker.py --lan
```

Each checker uses a fixed seed to select exactly 1% of unique implicit queries and presents one `query -> multimodal evidence -> attribute -> target` chain per page. Qualified/unqualified judgments and optional notes persist in SQLite, the completed sample reports its final qualified percentage, and reviews can be exported as JSONL. Startup rejects legacy datasets that still contain one-to-many implicit qrels.

To have a multimodal model judge whether every sampled `query row -> evidence -> attribute` recovery is actually supported, run the stable 10% auto-check:

```bash
conda run --no-capture-output -n MMDD python \
  scripts/mm_joinability_dataset_auto_checker.py \
  --output_dir output_mm_joinability_v15 \
  --provider local \
  --local_base_url http://127.0.0.1:8001/v1 \
  --local_model Qwen3.5-9B \
  --sample_rate 0.10 \
  --seed 13 \
  --cache_path cache/mm_joinability/auto_checker.sqlite3
```

For every `(row, evidence, attribute)`, the auto checker performs an independent leave-one-attribute-out extraction. It removes only that attribute from the original complete row, and gives the model the remaining row, one evidence item, and the attribute name. The model never receives the dataset's claimed value or any target row; it returns only `extracted_value`. The checker then uses the builder's normalization locally: a match is `supported`, a nonempty mismatch is `contradicted`, and an empty extraction is `insufficient`. The local model runs first. Only local `contradicted` or `insufficient` results are sent through the same blind extraction with `gpt-5.6-terra` at `medium` reasoning, with OpenAI secondary concurrency hard-limited to five. The checker, not either model, derives the final verdict by comparing the secondary extraction when one was requested.

Reports under `<output_dir>/auto_checker_reviews/` contain both extraction stages at path level, query coverage, aggregate statistics, errors, and `patch_candidates-*.jsonl` for later dataset cleanup. Successful results from both stages are cached by `(row, evidence, attribute, model identity)` in SQLite; failures are not cached. Stable hash-prefix sampling lets a later 20% run reuse the earlier 10%. Secondary screening reads `OPENAI_API_KEY` and follows the OpenAI builder's restricted dotenv convention; keys are never written to cache or reports. Use `--no_secondary_openai` to disable it. Every request contains exactly one `masked row + evidence + attribute name`; multiple attributes claimed by the same evidence are requested separately.

By default, Flask GUIs bind to `127.0.0.1` and are available only on the same machine. Add `--lan` to `stage1_train_models.py`, `stage1_run_all.py`, `run_hitl_training_rounds.py`, `hitl_annotation_app.py`, `stage1_connection_viewer.py`, or `mm_joinability_dataset_viewer.py` to bind to `0.0.0.0`; the startup log prints both the local URL and the LAN URL for other devices. If the LAN URL is still unreachable, allow the selected port through the host firewall and make sure the devices are on the same network.

Use `--force_retrain` when you want to ignore partially generated training outputs and start the training/HITL loop cleanly. It removes teacher/student models, teacher scores, train pairs, open HITL selections, templates, and annotation status files while preserving prepared data, embeddings, and merged human labels. Add `--reset_human_labels` only when you also want to discard previously merged human labels.

Lower-level commands are still available for debugging or running individual steps:

```bash
conda run -n MMDD python scripts/build_stage1_logic_connectivity.py --input_dir output_medium --output_dir output_stage1_logic --seed 13
conda run -n MMDD python scripts/build_stage1_evidence_paths.py --input_dir output_medium --stage1_dir output_stage1_logic --seed 13
conda run -n MMDD python scripts/generate_weak_labels.py --stage1_dir output_stage1_logic
conda run -n MMDD python scripts/select_hitl_batch.py --stage1_dir output_stage1_logic --round_id 0 --batch_size 50
conda run -n MMDD python scripts/build_stage1_embeddings.py --input_dir output_medium --stage1_dir output_stage1_logic --encoder_path ./Qwen3-VL-Embedding-2B --batch_size 8 --device cuda --dtype bf16
conda run -n MMDD python scripts/build_teacher_training_data.py --stage1_dir output_stage1_logic
conda run -n MMDD python scripts/train_teacher.py --stage1_dir output_stage1_logic --embedding_dir output_stage1_logic/embeddings
conda run -n MMDD python scripts/train_student.py --stage1_dir output_stage1_logic --embedding_dir output_stage1_logic/embeddings --teacher_scores output_stage1_logic/teacher_scores.jsonl --distill_loss pairwise
conda run -n MMDD python scripts/build_hnsw_indices.py --stage1_dir output_stage1_logic --student_dir output_stage1_logic/student --space cosine --m 32 --ef_construction 200 --ef_search 100
conda run -n MMDD python scripts/eval_stage1_recall.py --stage1_dir output_stage1_logic --student_dir output_stage1_logic/student --hnsw_dir output_stage1_logic/hnsw_indices --topk 10 50 100 --max_hops 3 --beam_width 64 --beam_neighbors 50
```

`train_student.py` supports `--distill_loss pairwise` and `--distill_loss listwise`. ANN indexes store projected object vectors `u_b`; online retrieval computes the relation-aware query vector `u_a R_{\tau(a),\tau(b)}` before querying each target-type index. Path-aware evaluation uses beam-search multi-hop expansion and only returns table endpoints; assets are retained only as intermediate path evidence.

After filling `output_stage1_logic/human_labels_template_round_0.jsonl`, merge labels with:

```bash
conda run -n MMDD python scripts/merge_human_labels.py --stage1_dir output_stage1_logic --human_labels output_stage1_logic/human_labels_template_round_0.filled.jsonl
```

To run active-learning rounds where the teacher is retrained, uncertain paths are selected, and a browser UI collects human labels:

```bash
conda run -n MMDD python scripts/run_hitl_training_rounds.py \
  --stage1_dir output_stage1_logic \
  --embedding_dir output_stage1_logic/embeddings \
  --rounds 3 \
  --batch_size 50 \
  --force_retrain \
  --lan \
  --port 7860
```

Each round trains the teacher on the latest merged human labels, scores all candidate paths, writes `human_labels_template_round_<N>.jsonl`, starts the Flask annotation UI, and waits until the UI merges the completed labels into `human_labeled_paths.jsonl`. The selector excludes already merged labels and previously selected open batches by default.

To annotate an existing round without running training:

```bash
conda run -n MMDD python scripts/hitl_annotation_app.py --stage1_dir output_stage1_logic --round_id 0 --lan --port 7860
```

The default encoder is the local Qwen3-VL-Embedding-2B model. If `./Qwen3-VL-Embedding-2B` is not present, the wrapper also checks `./hf_models/Qwen3-VL-Embedding-2B`; pass `--encoder_path` for any other local path.
