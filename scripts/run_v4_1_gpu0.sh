#!/usr/bin/env bash
set -euo pipefail

expected_uuid='GPU-3d43b1bc-b727-456f-2b9f-e3c3b69eb725'
actual_uuid="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
actual_model="$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)"
[[ "$actual_uuid" == "$expected_uuid" ]] || { echo 'BLOCKED_GPU_IDENTITY' >&2; exit 10; }
[[ "$actual_model" == *'RTX 4090'* ]] || { echo 'BLOCKED_GPU_MODEL' >&2; exit 11; }

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$expected_uuid"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

conda run -n MMDD python -c "import os, torch; p=torch.cuda.get_device_properties(0); expected=os.environ['CUDA_VISIBLE_DEVICES'].removeprefix('GPU-').lower(); assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; assert 'RTX 4090' in p.name; assert str(p.uuid).lower() == expected; print({'mapped_device':'cuda:0','uuid':'GPU-'+str(p.uuid),'model':p.name})"
[[ $# -gt 0 ]] || { echo 'Usage: run_v4_1_gpu0.sh COMMAND ...' >&2; exit 2; }
exec "$@"
