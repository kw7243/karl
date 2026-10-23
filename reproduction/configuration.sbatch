#!/bin/bash
#SBATCH --job-name=karl-cpu-configuration
#SBATCH --partition=tig-cpu
#SBATCH --account=csail
#SBATCH --qos=tig-main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
export TZ=UTC
export OMP_NUM_THREADS=4
export TORCH_HOME=/data/scratch-fast/kwen1/.cache/karl-reproduce-20261023/torch
export HF_HOME=/data/scratch-fast/kwen1/.cache/karl-reproduce-20261023/huggingface
export DREAMSIM_CACHE=/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-assets/dreamsim
export PATH=/data/scratch-fast/kwen1/envs/karl-reproduce-20261023/bin:$PATH
out=/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-results/cpu-configuration-$SLURM_JOB_ID
mkdir -p "$out" "$DREAMSIM_CACHE"
exec > >(tee "$out/configuration.log") 2>&1
hostname
pwd
free -h
findmnt -T "$PWD"
python -c 'import torch; assert torch.__version__.split("+")[0]=="2.3.0"; print(torch.__version__)'
python -m pip install --no-compile transformers==4.44.2 peft==0.13.2 accelerate==0.34.2 open-clip-torch==2.24.0 setuptools==80.9.0 torch==2.3.0 torchvision==0.18.0 numpy==1.24.1 timm==0.4.9
python -m pip freeze > "$out/pip-freeze.txt"
python -m pip check
python -u reproduction/validate_configuration.py --preparation /data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-results/cpu-prepare-2826204 --output "$out"
