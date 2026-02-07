#!/bin/bash
#SBATCH --job-name=mmirage-example
#SBATCH --chdir=/users/fabnem/MIRAGE/src/mmirage
#SBATCH --output=/users/fabnem/reports/R-%x.%A_%a.out
#SBATCH --error=/users/fabnem/reports/R-%x.%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=11:59:59
#SBATCH -A a127
#SBATCH --array=0-15

# --- outputs & config ---
export ROOT=$SCRATCH/mmirage_example
export SHARDS_ROOT="$ROOT/shards"
export MERGED_DIR="$ROOT/merged"
export CFG=/users/fabnem/MIRAGE/configs/config_medtrinity_2.yaml

# HF cache/home
export HF_HOME=$SCRATCH/hf

mkdir -p "$SHARDS_ROOT"
mkdir -p "$MERGED_DIR"

export CMD="pip install /users/fabnem/MIRAGE --break-system-packages && python /users/fabnem/MIRAGE/src/mmirage/shard_process.py --config $CFG --profiler-log /users/fabnem/MIRAGE/prof.log"

SRUN_ARGS=" \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --jobid $SLURM_JOB_ID \
  --wait 60 \
  -A a127 \
  --reservation sai-a127 \
  --environment /users/$USER/.edf/mmirage.toml
  "
# bash -c is needed for the delayed interpolation of env vars to work
srun $SRUN_ARGS bash -c "$CMD"
echo "END TIME: $(date)"
