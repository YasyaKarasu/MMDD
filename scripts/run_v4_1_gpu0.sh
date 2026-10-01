#!/usr/bin/env bash
# Pin the process to the GPU named in a protocol file, then exec the command.
#
#   scripts/run_v4_1_gpu0.sh PROTOCOL.json COMMAND [ARGS...]
#
# The protocol's hardware.physical_index / hardware.uuid / hardware.model are checked
# against nvidia-smi, CUDA_VISIBLE_DEVICES is set to that UUID so the process sees exactly
# one device (cuda:0), and the pipeline's own _gpu_guard re-checks the UUID at start.
set -euo pipefail

[[ $# -gt 1 ]] || { echo 'Usage: run_v4_1_gpu0.sh PROTOCOL.json COMMAND ...' >&2; exit 2; }
protocol="$1"; shift
read -r expected_index expected_uuid expected_model < <(python3 - "$protocol" <<'PY'
import json, sys
hw = json.load(open(sys.argv[1]))["hardware"]
print(hw["physical_index"], hw["uuid"], hw["model"])
PY
)
actual_uuid="$(nvidia-smi -i "$expected_index" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
actual_model="$(nvidia-smi -i "$expected_index" --query-gpu=name --format=csv,noheader)"
[[ "$actual_uuid" == "$expected_uuid" ]] || { echo "BLOCKED_GPU_IDENTITY: index $expected_index is $actual_uuid, protocol pins $expected_uuid" >&2; exit 10; }
[[ "$actual_model" == *"$expected_model"* ]] || { echo "BLOCKED_GPU_MODEL: $actual_model" >&2; exit 11; }

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$expected_uuid"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

conda run -n MMDD python -c "import os, torch; p=torch.cuda.get_device_properties(0); expected=os.environ['CUDA_VISIBLE_DEVICES'].removeprefix('GPU-').lower(); assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; assert str(p.uuid).lower() == expected; print({'mapped_device':'cuda:0','uuid':'GPU-'+str(p.uuid),'model':p.name})"
exec "$@"
