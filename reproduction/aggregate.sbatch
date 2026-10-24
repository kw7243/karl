#!/bin/bash
#SBATCH --job-name=karl-cpu-completeness
#SBATCH --partition=tig-cpu
#SBATCH --account=csail
#SBATCH --qos=tig-main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
set -euo pipefail
export TZ=UTC OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PATH=/data/scratch-fast/kwen1/envs/karl-reproduce-20261023/bin:$PATH
out=/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-results/aggregation-$SLURM_JOB_ID
trap 'code=$?; if [ "$code" -ne 0 ]; then python reproduction/finalize_run.py "$out" "$code"; fi; exit "$code"' EXIT
hostname
pwd
free -h
findmnt -T "$PWD"
python -m pip install --no-compile matplotlib==3.7.5 numpy==1.24.1
python -m pip check
python -m pip freeze > "$out.pip-freeze.txt"
python -u reproduction/aggregate.py --output "$out"
