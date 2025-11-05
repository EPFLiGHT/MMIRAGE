#!/bin/bash
#SBATCH --job-name=med-sharded
#SBATCH --chdir /users/$CSCS_USERNAME/datasets
#SBATCH --output /users/$CSCS_USERNAME/reports/R-%x.%j.out
#SBATCH --error  /users/$CSCS_USERNAME/reports/R-%x.%j.err
#SBATCH --nodes 32
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH --time 11:59:59
#SBATCH --environment /users/$CSCS_USERNAME/.edf/sglang.toml
#SBATCH -A a127

module load cuda || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

# --- input datasets (HF load_from_disk folders) ---
DATA1=/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/medtrinity_conversations_1/
DATA2=/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/medtrinity_conversations_2/

# --- outputs & config ---
OUTDIR=/users/$CSCS_USERNAME/datasets/outputs_medtrinity
CFG=config-sglang.yaml                    
SCRIPT=rewrite_assistant_md.py            

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