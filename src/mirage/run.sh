#!/bin/bash
#SBATCH --job-name=med-sharded
#SBATCH --chdir /users/$USER/meditron/MIRAGE/src/mirage
#SBATCH --output /users/$USER/reports/R-%x.%j.out
#SBATCH --error  /users/$USER/reports/R-%x.%j.err
#SBATCH --nodes 32
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH --time 11:59:59
#SBATCH --environment /users/$USER/.edf/sglang.toml
#SBATCH -A a127

# --- input datasets (HF load_from_disk folders) ---
DATA1=/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/medtrinity_conversations_1/
DATA2=/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/medtrinity_conversations_2/

# --- outputs & config ---
OUTDIR=/capstor/store/cscs/swissai/a127/homes/$USER/datasets/medtrinity                   
SHARDS_ROOT="$ROOT/shards"
MERGED_DIR="$ROOT/merged"
CFG=/users/$USER/meditron/MIRAGE/src/mirage/config.yaml 

mkdir -p "$SHARDS_ROOT"
mkdir -p "$MERGED_DIR"

export HF_HOME=/capstor/store/cscs/swissai/a127/homes/$USER/hf

python /users/$USER/meditron/MIRAGE/src/mirage/shard_process.py \
  --datasets "$DATA1" "$DATA2" \
  --output_dir "$SHARDS_ROOT" \
  --num_shards "$SLURM_JOB_NUM_NODES" \
  --shard_id "$SLURM_NODEID" \
  --config "$CFG"