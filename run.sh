#!/bin/bash
#SBATCH --job-name=mirage-example
#SBATCH --chdir=/users/$USER/meditron/MIRAGE/src/mirage
#SBATCH --output=/users/$USER/reports/R-%x.%A_%a.out
#SBATCH --error=/users/$USER/reports/R-%x.%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=11:59:59
#SBATCH -A a127
#SBATCH --array=0-3

# --- outputs & config ---
export ROOT=$SCRATCH/mirage_example
export SHARDS_ROOT="$ROOT/shards"
export MERGED_DIR="$ROOT/merged"
export CFG=/users/$USER/meditron/MIRAGE/configs/config_small.yaml

# HF cache/home
export HF_HOME=$SCRATCH/hf

mkdir -p "$SHARDS_ROOT"
mkdir -p "$MERGED_DIR"

export CMD="python /users/$USER/meditron/MIRAGE/src/mirage/shard_process.py --config $CFG"

SRUN_ARGS=" \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --jobid $SLURM_JOB_ID \
  --wait 60 \
  -A a127 \
  --reservation sai-a127 \
  --environment /users/$USER/.edf/mirage.toml
  "
# bash -c is needed for the delayed interpolation of env vars to work
srun $SRUN_ARGS bash -c "$CMD"
echo "END TIME: $(date)"
