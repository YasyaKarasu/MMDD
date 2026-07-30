# EntiTables-priority local GPU sharing

This setup lets the WDC model stage add one local text endpoint and one local
image endpoint while EntiTables is crawling. EntiTables remains the priority
owner and reclaims both local GPUs before every model-analysis round.

## Protocol

The processes share one coordination directory for ownership records.
EntiTables publishes generation-scoped `borrowable` and `priority_requested`
records. The WDC builder requires its dynamic endpoint files to live under
`work_dir/runtime`, so those files are separate from the coordination records.
The WDC borrower:

1. publishes local endpoints only after both local vLLM servers are healthy;
2. atomically empties both WDC endpoint files on a priority request;
3. terminates the complete local vLLM process groups;
4. writes a matching `released` acknowledgement only after both groups exit.

EntiTables rejects stale acknowledgements and refuses to start when a registered
borrower heartbeat is stale. A cleanly stopped borrower removes its
registration. WDC always retains the configured remote endpoints, so a local
request interrupted by preemption is persisted as retryable and can continue
remotely.

The examples below use:

```text
/home/oycy/MMDD/runtime_gpu_share/
  priority_request.json
  borrower_status.json
  borrower_ack.json
  logs/

/home/oycy/MMDD/work_wdc_200k_eta_advisory_20260719/runtime/
  wdc_borrowed_text_endpoints.txt
  wdc_borrowed_image_endpoints.txt
```

The endpoint files must be dedicated to the borrower sidecar because it owns
and clears them.

## EntiTables runner

Add the shared directory to the existing EntiTables command:

```bash
conda run --no-capture-output -n MMDD python \
  /home/oycy/MMDD/.worktrees/mm-joinability-random-replacement/scripts/run_mm_joinability_dynamic_vllm.py \
  --input_dir /home/oycy/MMDD/dataset/tables_redi2_1 \
  --output_dir /home/oycy/MMDD/output_mm_joinability_v6 \
  --text_model_path /home/oycy/MMDD/hf_models/Qwen3.5-9B \
  --image_model_path /home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking \
  --gpu_coordination_dir /home/oycy/MMDD/runtime_gpu_share \
  --dynamic_model_workers 8 \
  --vllm_max_num_seqs 4 \
  --vllm_max_num_batched_tokens 2048 \
  --min_cols 4 \
  --max_source_tables 40000 \
  --max_images_per_entity 3 \
  --min_recovered_value_ratio 0.5 \
  --query_rows_per_table 5 \
  --min_rows_per_output_table 5
```

Without `--gpu_coordination_dir`, the runner retains its previous standalone
behavior. With coordination enabled, round-mode startup is deferred until the
per-round handshake, and all EntiTables vLLM services are stopped before the
next crawling phase is published as borrowable.

## WDC builder

Keep the remote tunnel endpoints as the fixed base URLs and add the two dynamic
files:

```bash
conda run --no-capture-output -n MMDD python \
  /home/oycy/MMDD/.worktrees/wdc-200k/scripts/build_wdc200k_mm_joinability_dataset.py \
  --input_dir /home/oycy/MMDD/wdc_schemaorg_2023 \
  --output_dir /home/oycy/MMDD/output_wdc_200k_sampled_20260720 \
  --work_dir /home/oycy/MMDD/work_wdc_200k_eta_advisory_20260719 \
  --cache_dir /home/oycy/MMDD/cache/wdc_200k_sampled_20260720 \
  --runtime_dir /home/oycy/MMDD/work_wdc_200k_eta_advisory_20260719/runtime \
  --max_source_tables 200000 \
  --selection_seed 13 \
  --sampled_entities_per_table 8 \
  --entity_sampling_seed 20260720 \
  --min_free_disk_bytes 107374182400 \
  --resume \
  --text_model_base_url http://127.0.0.1:18001/v1 \
  --text_model_base_urls_file /home/oycy/MMDD/work_wdc_200k_eta_advisory_20260719/runtime/wdc_borrowed_text_endpoints.txt \
  --text_model_name Qwen3.5-9B \
  --image_model_base_url http://127.0.0.1:18000/v1 \
  --image_model_base_urls_file /home/oycy/MMDD/work_wdc_200k_eta_advisory_20260719/runtime/wdc_borrowed_image_endpoints.txt \
  --image_model_name Qwen3-VL-8B-Thinking \
  --text_model_workers 8 \
  --image_model_workers 4 \
  --model_endpoint_ready_timeout_seconds 180 \
  --model_timeout_seconds 300 \
  --model_max_retries 4 \
  --model_retry_sleep_seconds 2
```

The builder re-reads the dynamic files before each request. The fixed remote
URLs are never removed.

## WDC borrower sidecar

Run one sidecar on the local host:

```bash
conda run --no-capture-output -n MMDD python \
  /home/oycy/MMDD/.worktrees/wdc-200k/scripts/run_wdc_local_gpu_borrower.py \
  --coordination_dir /home/oycy/MMDD/runtime_gpu_share \
  --text_endpoints_file /home/oycy/MMDD/work_wdc_200k_eta_advisory_20260719/runtime/wdc_borrowed_text_endpoints.txt \
  --image_endpoints_file /home/oycy/MMDD/work_wdc_200k_eta_advisory_20260719/runtime/wdc_borrowed_image_endpoints.txt \
  --text_model_path /home/oycy/MMDD/hf_models/Qwen3.5-9B \
  --image_model_path /home/oycy/MMDD/hf_models/Qwen3-VL-8B-Thinking
```

If the local vLLM servers require the same API key as the tunneled servers,
export `VLLM_API_KEY` before starting both the WDC builder and sidecar. The
sidecar lock prevents two borrowers from managing the same coordination
directory.

Existing Python processes do not load these changes dynamically. Restart both
builders with their resume options, then start the sidecar. Durable WDC model
jobs and EntiTables caches make those restarts resumable.

## Operational checks

- During crawling, `priority_request.json` is `borrowable`,
  `borrower_ack.json` becomes `serving`, and the borrowed endpoint files under
  the WDC runtime directory contain ports `18101` and `18100`.
- Before EntiTables model analysis, the request becomes
  `priority_requested`; both endpoint files become empty before the
  acknowledgement becomes `released`.
- A stale `borrower_status.json` deliberately blocks EntiTables model startup.
  Inspect and stop orphaned borrower vLLM groups before removing stale state.
