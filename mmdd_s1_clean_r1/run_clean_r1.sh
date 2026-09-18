#!/usr/bin/env bash
# CLEAN-R1 full pipeline, spec section 11.  Run from the repository root.
#
#   bash mmdd_s1_clean_r1/run_clean_r1.sh
#
# Every command appends a receipt to work/s1_clean_r1_20260917/COMMANDS.jsonl and
# the script stops at the first non-zero exit: a failed step is never skipped.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export MMDD_QWEN_MODEL_DIR="${MMDD_QWEN_MODEL_DIR:-$REPO/hf_models/Qwen3-VL-Embedding-8B}"
# Per-object work is dominated by single-image CPU preprocessing; a large intra-op
# thread pool only thrashes across the two cache shards.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

CONFIG="${1:-clean_r1.json}"
SKIP_CACHE_BUILD="${SKIP_CACHE_BUILD:-0}"
RUN=(python -m mmdd_stage1_clean)

have_manifest() {
  [ -s work/s1_clean_r1_20260917/cache/manifest.jsonl ]
}

step() { echo; echo "=== $* ==="; date -u +%Y-%m-%dT%H:%M:%SZ; }

if [ -s work/s1_clean_r1_20260917/INPUT_INVENTORY.json ] \
   && [ -s work/s1_clean_r1_20260917/resolved_config.json ]; then
  echo "audit-input and build-objects already present; reusing them"
else
  step "audit-input"
  "${RUN[@]}" audit-input --config "$CONFIG"

  step "build-objects"
  "${RUN[@]}" build-objects --config "$CONFIG"
fi

if have_manifest; then
  echo "cache manifest already present; skipping the encoding pass"
elif [ "$SKIP_CACHE_BUILD" = "1" ]; then
  echo "SKIP_CACHE_BUILD=1 but no cache manifest exists; refusing to continue"; exit 1
else
  step "cache (two GPU shards, run in parallel)"
  CUDA_VISIBLE_DEVICES=0 "${RUN[@]}" cache --config "$CONFIG" --shard-id 0 --shard-count 2 &
  PID0=$!
  CUDA_VISIBLE_DEVICES=1 "${RUN[@]}" cache --config "$CONFIG" --shard-id 1 --shard-count 2 &
  PID1=$!
  wait $PID0 || { echo "cache shard 0 failed"; exit 1; }
  wait $PID1 || { echo "cache shard 1 failed"; exit 1; }

  step "cache merge"
  "${RUN[@]}" cache --config "$CONFIG" --merge-shards
fi

step "raw-retrieve train"
"${RUN[@]}" raw-retrieve --split train --config "$CONFIG"

step "raw-retrieve dev"
"${RUN[@]}" raw-retrieve --split dev --config "$CONFIG"

step "build-supervision"
"${RUN[@]}" build-supervision --config "$CONFIG"

step "train-teacher"
"${RUN[@]}" train-teacher --config "$CONFIG"

step "freeze-teacher"
"${RUN[@]}" freeze-teacher --config "$CONFIG"

step "train-student SUP"
"${RUN[@]}" train-student --arm SUP --config "$CONFIG"

step "train-student KD"
"${RUN[@]}" train-student --arm KD --config "$CONFIG"

step "freeze-selection"
"${RUN[@]}" freeze-selection --config "$CONFIG"

step "evaluate dev"
"${RUN[@]}" evaluate --split dev --config "$CONFIG"

step "raw-retrieve test"
"${RUN[@]}" raw-retrieve --split test --config "$CONFIG"

step "evaluate test"
"${RUN[@]}" evaluate --split test --config "$CONFIG"

step "diagnose"
"${RUN[@]}" diagnose --config "$CONFIG"

step "package"
"${RUN[@]}" package --config "$CONFIG"

echo
echo "CLEAN-R1 pipeline finished."
