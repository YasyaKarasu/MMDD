#!/usr/bin/env bash
# CLEAN-R1 driver, with the two independent phases spread over both GPUs.
#
#   bash mmdd_s1_clean_r1/run_clean_r1_parallel.sh [config]
#
# Ordering is fixed by the protocol, and only the parts that are genuinely
# independent are run at once:
#
#   Teacher trains and is frozen alone -- both Student arms need its frozen
#   checkpoint, and no Student exists before it.
#
#   S-SUP and S-KD then run on one GPU each.  They share an initialisation and
#   their own hard-negative mining, and neither can influence the other, so they
#   are independent by construction.
#
#   Evaluation is split by method across the two GPUs.  A group is chosen so that
#   methods sharing an index travel together: RAW-D, RAW-2H and RAW+T read the same
#   frozen Qwen vectors, so building the raw index once for the group avoids paying
#   its HNSW construction three times.  The teacher-dependent groups are separated
#   because each needs its own Student index anyway.
#
# Results from the parallel groups are recombined by `merge-methods`, so every
# downstream reader still sees one EVAL_{split}.json.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export MMDD_QWEN_MODEL_DIR="${MMDD_QWEN_MODEL_DIR:-$REPO/hf_models/Qwen3-VL-Embedding-8B}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

CONFIG="${1:-clean_r1.json}"
RUN=(python -m mmdd_stage1_clean)

step() { echo; echo "=== $* ==="; date -u +%Y-%m-%dT%H:%M:%SZ; }

run_group() {  # name device split methods...
  local name="$1" device="$2" split="$3" methods="$4"
  CUDA_VISIBLE_DEVICES="$device" "${RUN[@]}" evaluate --config "$CONFIG" \
    --split "$split" --methods "$methods" --device cuda \
    > "work/s1_clean_r1_20260917/evaluate_${split}_${name}.log" 2>&1
}

# ---------------------------------------------------------------- teacher only
step "train-teacher"

step "freeze-teacher"
"${RUN[@]}" freeze-teacher --config "$CONFIG" || exit 1

# ------------------------------------------------------- both Student arms
step "train-student SUP and KD in parallel (one GPU each)"
CUDA_VISIBLE_DEVICES=0 "${RUN[@]}" train-student --arm SUP --config "$CONFIG" \
  > /tmp/student_SUP.log 2>&1 &
SUP_PID=$!
CUDA_VISIBLE_DEVICES=1 "${RUN[@]}" train-student --arm KD --config "$CONFIG" \
  > /tmp/student_KD.log 2>&1 &
KD_PID=$!
wait $SUP_PID || { echo "S-SUP failed"; exit 1; }
wait $KD_PID || { echo "S-KD failed"; exit 1; }

step "freeze-selection"
"${RUN[@]}" freeze-selection --config "$CONFIG" || exit 1

# ------------------------------------------------------------- evaluation
# Group A holds every method that reads the raw vectors: one raw index serves all
# three.  Group B holds the teacher-dependent students, which need their own index.
# Each group walks both splits, so at any moment the two GPUs are working on the
# same split with different methods.
for split in dev test; do
  step "evaluate $split (two groups in parallel)"
  run_group raw   0 "$split" "RAW-D,RAW-2H,RAW+T" &
  RAW_PID=$!
  run_group studs 1 "$split" "SUP+T,KD+T" &
  STUD_PID=$!
  wait $RAW_PID || { echo "evaluate $split raw group failed"; exit 1; }
  wait $STUD_PID || { echo "evaluate $split student group failed"; exit 1; }
done

step "merge-methods"
"${RUN[@]}" merge-methods --config "$CONFIG" || exit 1

step "diagnose"
"${RUN[@]}" diagnose --config "$CONFIG" || exit 1

step "package"
"${RUN[@]}" package --config "$CONFIG" || exit 1

echo
echo "CLEAN-R1 parallel pipeline finished."
