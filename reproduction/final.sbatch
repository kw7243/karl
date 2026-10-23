#!/bin/bash
#SBATCH --job-name=karl-released-final
#SBATCH --partition=vision-shared-rtx2080ti
#SBATCH --account=vision-torralba-urops-meng
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
set -euo pipefail
export TZ=UTC
checkpoint_dataset=$1
case "$checkpoint_dataset" in imagenet|imagenet100) ;; *) exit 2 ;; esac
root=/data/vision/torralba/u/kwen1/image-tokenization-20261023
bash reproduction/run_gpu.sh "$root/karl-reproduction-results/final-$SLURM_JOB_ID-$checkpoint_dataset" --checkpoint-manifest "$root/karl-reproduction-results/cpu-prepare-2826204/checkpoints.json" --dataset-manifest "$root/karl-reproduction-results/cpu-prepare-2826204/dataset-manifest.json" --checkpoint-dataset "$checkpoint_dataset" --mode final
