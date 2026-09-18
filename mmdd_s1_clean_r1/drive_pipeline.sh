#!/usr/bin/env bash
# Waits for the running cache shards, merges them and executes the rest of the
# CLEAN-R1 pipeline.  Detached so it survives the invoking shell.
set -uo pipefail

REPO=/home/oycy/MMDD
cd "$REPO"
OUT="$REPO/work/s1_clean_r1_20260917"
LOG="$OUT/DRIVER.log"
mkdir -p "$OUT"

exec >>"$LOG" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] driver started; waiting for cache shards"

# The currently running shards (restarted at 12:04) process interleaved object
# subsets.  If they die early the driver still proceeds and the merge step will
# report the shortfall rather than silently filling it.
while pgrep -f "mmdd_stage1_clean cache --config .* --shard-id" >/dev/null 2>&1; do
  sleep 60
done
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] cache shards finished"

for i in 0 1; do
  echo "gpu-$i objects: $(wc -l < "$OUT/cache/gpu-$i/manifest.jsonl" 2>/dev/null || echo missing)"
done

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export MMDD_QWEN_MODEL_DIR="$REPO/hf_models/Qwen3-VL-Embedding-8B"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

SKIP_CACHE_BUILD=1 bash "$REPO/mmdd_s1_clean_r1/run_clean_r1.sh" clean_r1.json
STATUS=$?
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pipeline exit status: $STATUS"
exit $STATUS
