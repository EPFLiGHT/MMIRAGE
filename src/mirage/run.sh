#!/bin/bash
#SBATCH --job-name=med-sharded
#SBATCH --chdir /users/$USER/datasets
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
CFG=config-sglang.yaml                    
SCRIPT=split_script_sglang.py            

mkdir -p "$OUTDIR"

srun --ntasks=${SLURM_NTASKS} --ntasks-per-node=1 bash -lc "
  python $SCRIPT \
    --datasets $DATA1 $DATA2 \
    --output_dir $OUTDIR \
    --num_shards $SLURM_NTASKS \
    --shard_id \$SLURM_PROCID \
    --config $CFG \
    --write_every 100 \
    --resume
"