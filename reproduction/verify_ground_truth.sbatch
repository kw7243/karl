#!/bin/bash
#SBATCH --job-name=karl-gt-cache-readback
#SBATCH --partition=tig-cpu
#SBATCH --account=csail
#SBATCH --qos=tig-main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
export TZ=UTC OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PATH=/data/scratch-fast/kwen1/envs/karl-reproduce-20261023/bin:$PATH
out=/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-results/gt-cache-readback-$SLURM_JOB_ID
trap 'code=$?; if [ "$code" -ne 0 ]; then python reproduction/finalize_run.py "$out" "$code"; fi; exit "$code"' EXIT
hostname
pwd
free -h
findmnt -T "$PWD"
python -u reproduction/verify_ground_truth.py --output "$out"
