#!/bin/bash
#SBATCH --job-name=mmirage-medtrinity
#SBATCH --chdir=/users/qchapp/meditron/MIRAGE/src/mmirage
#SBATCH --output=/users/qchapp/reports/R-%x.%A_%a.out
#SBATCH --error=/users/qchapp/reports/R-%x.%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=11:59:59
#SBATCH -A a127
#SBATCH --array=0-31

# --- outputs & config ---
export CFG=/users/qchapp/meditron/MIRAGE/configs/config_medtrinity.yaml

# HF cache/home
export HF_HOME=/capstor/store/cscs/swissai/a127/homes/qchapp/hf

export CMD="python /users/qchapp/meditron/MIRAGE/src/mmirage/shard_process.py --config $CFG"

SRUN_ARGS=" \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --jobid $SLURM_JOB_ID \
  --wait 60 \
  -A a127 \
  --reservation sai-a127 \
  --environment /users/qchapp/.edf/sglang.toml
  "
# bash -c is needed for the delayed interpolation of env vars to work
srun $SRUN_ARGS bash -c "$CMD"
echo "END TIME: $(date)"

