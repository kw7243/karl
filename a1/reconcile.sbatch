#!/bin/bash
#SBATCH --job-name=karl-a1-cpu
#SBATCH --partition=tig-cpu
#SBATCH --account=csail
#SBATCH --qos=tig-main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
set -euo pipefail
printf 'FIRST_SCRIPT_MARKER job=%s host=%s\n' "$SLURM_JOB_ID" "$(hostname)"
export PATH=/data/scratch-fast/kwen1/envs/karl-reproduce-20261023/bin:/usr/local/bin:/usr/bin:/bin
export TZ=UTC OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
root=/data/vision/torralba/u/kwen1/image-tokenization-20261023
out="$root/karl-a1-20261024-results/A1/cpu-$SLURM_JOB_ID"
trap 'code=$?; printf "TERMINAL_SCRIPT_MARKER job=%s exit=%s\n" "$SLURM_JOB_ID" "$code"' EXIT
hostname
pwd
free -h
findmnt -T "$PWD"
python -c 'import torch,os; print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),os.environ.get("CUDA_VISIBLE_DEVICES")); assert not torch.cuda.is_available()'
python -u a1/reconcile.py --output "$out"
python -m pip freeze > "$out/pip-freeze.txt"
