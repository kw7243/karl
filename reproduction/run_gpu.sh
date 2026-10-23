#!/bin/bash
set -euo pipefail
export TZ=UTC
export OMP_NUM_THREADS=4
export TORCH_HOME=/data/scratch-fast/kwen1/.cache/karl-reproduce-20261023/torch
export HF_HOME=/data/scratch-fast/kwen1/.cache/karl-reproduce-20261023/huggingface
export DREAMSIM_CACHE=/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-assets/dreamsim
export PATH=/data/scratch-fast/kwen1/envs/karl-reproduce-20261023/bin:$PATH
out=$1
shift
mkdir -p "$(dirname "$out")"
trap 'code=$?; python reproduction/finalize_run.py "$out" "$code"; exit "$code"' EXIT
hostname
pwd
nvidia-smi
free -h
findmnt -T "$PWD"
findmnt -T "$DREAMSIM_CACHE"
python -c 'import torch; print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_properties(0)); assert torch.__version__.split("+")[0]=="2.3.0"; assert torch.cuda.is_available()'
python -u reproduction/evaluate_checkpoint.py --output "$out" "$@"
