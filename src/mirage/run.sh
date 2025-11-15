#!/bin/bash
#SBATCH --job-name=med-sharded
#SBATCH --chdir /users/$USER/meditron/MIRAGE/src/mirage
#SBATCH --output /users/$USER/reports/R-%x.%j.out
#SBATCH --error  /users/$USER/reports/R-%x.%j.err
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH --time 11:59:59
#SBATCH --environment /users/$USER/.edf/sglang.toml
#SBATCH -A a127

# --- input datasets (HF load_from_disk folders) ---
export DATA1=/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/medtrinity_conversations_1/
export DATA2=/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/medtrinity_conversations_2/

# --- outputs & config ---
export ROOT=/capstor/store/cscs/swissai/a127/homes/$USER/datasets/medtrinity                   
export SHARDS_ROOT="$ROOT/shards"
export MERGED_DIR="$ROOT/merged"
export CFG=/users/$USER/meditron/MIRAGE/src/mirage/config.yaml 
export HF_HOME=/capstor/store/cscs/swissai/a127/homes/$USER/hub

mkdir -p "$SHARDS_ROOT"
mkdir -p "$MERGED_DIR"

export HF_HOME=/capstor/store/cscs/swissai/a127/homes/$USER/hf

export CMD="python /users/$USER/meditron/MIRAGE/src/mirage/shard_process.py \
  --datasets $DATA1 $DATA2 \
  --output_dir $SHARDS_ROOT \
  --num_shards $SLURM_JOB_NUM_NODES \
  --shard_id \$SLURM_NODEID \
  --config $CFG"

echo $CMD

SRUN_ARGS=" \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --jobid $SLURM_JOB_ID \
  --wait 60 \
  -A a06 \
  --reservation=sai-a127
  "

srun $SRUN_ARGS bash -c "$CMD"
