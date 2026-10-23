#!/bin/bash
#SBATCH --job-name=karl-cpu-input-audit
#SBATCH --partition=tig-cpu
#SBATCH --account=csail
#SBATCH --qos=tig-main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
export TZ=UTC
export OMP_NUM_THREADS=1
python=/data/scratch-fast/kwen1/envs/karl-reproduce-20261023/bin/python
out=/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-results/cpu-input-audit-$SLURM_JOB_ID
"$python" -u reproduction/audit_inputs.py "$out"
