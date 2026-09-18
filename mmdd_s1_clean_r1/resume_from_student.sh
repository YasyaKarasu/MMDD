#!/usr/bin/env bash
# Resume CLEAN-R1 after the Teacher is already trained and frozen.
#
#   bash mmdd_s1_clean_r1/resume_from_student.sh [config]
#
# Skips audit-input, build-objects, cache, raw-retrieve, build-supervision,
# train-teacher and freeze-teacher, all of which are already complete on disk and
# whose inputs have not changed.  Everything after the Teacher runs exactly as in
# run_clean_r1_parallel.sh.
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

[ -f work/s1_clean_r1_20260917/teacher/best.pt ] || {
  echo "teacher/best.pt is missing; run the full pipeline instead"; exit 1; }

run_group() {
  local name="$1" device="$2" split="$3" methods="$4"
  CUDA_VISIBLE_DEVICES="$device" "${RUN[@]}" evaluate --config "$CONFIG" \
    --split "$split" --methods "$methods" --device cuda \
    > "work/s1_clean_r1_20260917/evaluate_${split}_${name}.log" 2>&1
}

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

for split in dev test; do
  step "evaluate $split (two groups in parallel)"
  run_group raw   0 "$split" "RAW-D,RAW-2H,RAW+T" & RAW_PID=$!
  run_group studs 1 "$split" "SUP+T,KD+T" & STUD_PID=$!
  wait $RAW_PID || { echo "evaluate $split raw group failed"; exit 1; }
  wait $STUD_PID || { echo "evaluate $split student group failed"; exit 1; }
done

step "merge-methods"
"${RUN[@]}" merge-methods --config "$CONFIG" || exit 1
step "diagnose"
"${RUN[@]}" diagnose --config "$CONFIG" || exit 1
step "package"
"${RUN[@]}" package --config "$CONFIG" || exit 1
echo; echo "CLEAN-R1 resumed pipeline finished."
