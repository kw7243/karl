#!/bin/bash
#SBATCH --job-name=karl-cpu-prepare
#SBATCH --partition=tig-cpu
#SBATCH --account=csail
#SBATCH --qos=tig-main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
export TZ=UTC
export OMP_NUM_THREADS=4
export MAMBA_ROOT_PREFIX=/data/scratch-fast/kwen1/.cache/karl-reproduce-20261023/mamba
export PIP_CACHE_DIR=/data/scratch-fast/kwen1/.cache/karl-reproduce-20261023/pip
envdir=/data/scratch-fast/kwen1/envs/karl-reproduce-20261023
out=/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-results/cpu-prepare-$SLURM_JOB_ID
mkdir -p "$out"
exec > >(tee "$out/prepare.log") 2>&1
hostname
pwd
printf 'JOB_ID=%s\n' "$SLURM_JOB_ID"
free -h
findmnt -T "$PWD"
findmnt -T /data/scratch-fast/kwen1
/data/scratch-fast/kwen1/micromamba/bin/micromamba create --yes --prefix "$envdir" -c conda-forge python=3.10 pip=23.3.2
"$envdir/bin/pip" install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
"$envdir/bin/pip" install numpy==1.24.1 einops==0.3.0 omegaconf==2.0.0 pytorch-lightning==1.0.8 timm==0.4.9 gdown opencv-python pytorch-fid tqdm wandb torchmetrics==1.4.0 lpips==0.1.4 dreamsim
"$envdir/bin/pip" freeze > "$out/pip-freeze.txt"
"$envdir/bin/python" -u reproduction/prepare.py --output "$out"
