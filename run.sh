#!/bin/bash
#SBATCH --job-name=med-sharded
#SBATCH --chdir=/users/$USER/meditron/MIRAGE/src/mirage
#SBATCH --output=/users/$USER/reports/R-%x.%A_%a.out
#SBATCH --error=/users/$USER/reports/R-%x.%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=11:59:59
#SBATCH --environment=/users/$USER/.edf/sglang.toml
#SBATCH -A a127
#SBATCH --array=0-31

# --- outputs & config ---
export ROOT=/capstor/store/cscs/swissai/a127/homes/$USER/datasets/english_small
export SHARDS_ROOT="$ROOT/shards"
export MERGED_DIR="$ROOT/merged"
export CFG=/users/$USER/MIRAGE/configs/config_small.yaml

# HF cache/home
export HF_HOME=/capstor/store/cscs/swissai/a127/homes/$USER/hf

mkdir -p "$SHARDS_ROOT"
mkdir -p "$MERGED_DIR"

python /users/$USER/MIRAGE/src/mirage/shard_process.py \
  --config "$CFG"
