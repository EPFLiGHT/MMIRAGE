#!/bin/bash
#SBATCH --job-name=mmirage-auto-retry
#SBATCH --chdir=/users/$USER/meditron/MMIRAGE/src/mmirage
#SBATCH --output=/users/$USER/reports/R-%x.%A_%a.out
#SBATCH --error=/users/$USER/reports/R-%x.%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=11:59:59
#SBATCH -A a127

##############################################################################
# MMIRAGE with Automatic Retry
#
# This script automatically detects and relaunches failed shards until all
# complete successfully or max retries is reached.
#
# Usage:
#   1. Edit the configuration section below
#   2. Run locally (NOT via sbatch): ./run_with_retry.sh
#   3. That's it - everything else is automatic
#
# NOTE: This script submits jobs to SLURM internally, so run it as a regular
#       bash script, not with sbatch.
##############################################################################

# ============================================================================
# CONFIGURATION - Edit these to match your setup
# ============================================================================
export USER="username"

export MMIRAGE_PATH="/users/$USER/meditron/MMIRAGE"

# Number of shards (will launch array 0 to NUM_SHARDS-1)
export NUM_SHARDS=32

# Path to your MMIRAGE config file
export CFG=$MMIRAGE_PATH/configs/config_medtrinity.yaml

# Output directory for shards
export SHARDS_ROOT=$SCRATCH/mmirage_output/shards

# HF cache/home
export HF_HOME=$SCRATCH/hf

# Maximum retry attempts per shard (prevents infinite loops)
export MAX_RETRIES=3

# SLURM settings for worker nodes
export WORKER_ACCOUNT="a127"
export WORKER_RESERVATION="sai-a127"
export WORKER_ENVIRONMENT="/users/$USER/.edf/sglang.toml"

# ============================================================================
# END CONFIGURATION - Don't edit below unless you know what you're doing
# ============================================================================

mkdir -p "$SHARDS_ROOT"

# Detect which mode we're in based on SLURM variables
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # ========================================================================
    # WORKER MODE - Process one shard
    # ========================================================================
    echo "=========================================="
    echo "Worker: Processing shard $SLURM_ARRAY_TASK_ID"
    echo "Started at: $(date)"
    echo "=========================================="
    
    export CMD="python $MMIRAGE_PATH/src/mmirage/shard_process.py --config $CFG"
    
    SRUN_ARGS=" \
      --cpus-per-task $SLURM_CPUS_PER_TASK \
      --jobid $SLURM_JOB_ID \
      --wait 60 \
      -A $WORKER_ACCOUNT \
      --reservation $WORKER_RESERVATION \
      --environment $WORKER_ENVIRONMENT
      "
    
    srun $SRUN_ARGS bash -c "$CMD"
    EXIT_CODE=$?
    
    echo "END TIME: $(date)"
    echo "EXIT CODE: $EXIT_CODE"
    
    exit $EXIT_CODE

elif [ -n "$SLURM_JOB_ID" ]; then
    # ========================================================================
    # CONTROLLER MODE - Check for failures and resubmit
    # ========================================================================
    echo "=========================================="
    echo "Controller: Checking for failed shards"
    echo "Started at: $(date)"
    echo "=========================================="
    echo ""
    
    # Function to check for successful shards
    check_shards() {
        local failed_shards=()
        local success_count=0
        local failed_count=0
        local missing_count=0
        
        for i in $(seq 0 $((NUM_SHARDS-1))); do
            # Find shard directories (may be nested under dataset dirs)
            shard_dirs=$(find "$SHARDS_ROOT" -type d -name "shard_$i" 2>/dev/null)
            
            if [ -z "$shard_dirs" ]; then
                echo "❓ Shard $i: MISSING (no directory)"
                failed_shards+=($i)
                ((missing_count++))
                continue
            fi
            
            # Check each shard directory for success marker
            shard_success=false
            for shard_dir in $shard_dirs; do
                if [ -f "$shard_dir/.SUCCESS" ]; then
                    shard_success=true
                    break
                fi
            done
            
            if [ "$shard_success" = true ]; then
                echo "✅ Shard $i: SUCCESS"
                ((success_count++))
            else
                # Check retry count
                retry_count=0
                for shard_dir in $shard_dirs; do
                    if [ -f "$shard_dir/.retry_count" ]; then
                        retry_count=$(cat "$shard_dir/.retry_count")
                        break
                    fi
                done
                
                if [ $retry_count -ge $MAX_RETRIES ]; then
                    echo "🛑 Shard $i: MAX RETRIES EXCEEDED ($retry_count/$MAX_RETRIES)"
                else
                    echo "❌ Shard $i: FAILED (retries: $retry_count/$MAX_RETRIES)"
                    failed_shards+=($i)
                    ((failed_count++))
                fi
            fi
        done
        
        echo ""
        echo "=========================================="
        echo "📊 Summary:"
        echo "  ✅ Successful: $success_count / $NUM_SHARDS"
        echo "  ❌ Failed/Missing: $failed_count"
        echo "=========================================="
        echo ""
        
        # Return failed shards as comma-separated list
        if [ ${#failed_shards[@]} -eq 0 ]; then
            return 0
        else
            IFS=','
            echo "${failed_shards[*]}"
            return 1
        fi
    }
    
    # Check for failures
    FAILED_LIST=$(check_shards)
    CHECK_EXIT=$?
    
    if [ $CHECK_EXIT -eq 0 ]; then
        echo "🎉 All shards completed successfully!"
        echo "You can now merge with:"
        echo "  python $MMIRAGE_PATH/src/mmirage/merge_shards.py \\"
        echo "    --input-dir $SHARDS_ROOT \\"
        echo "    --output-dir \$MERGED_DIR"
        exit 0
    fi
    
    echo "🔄 Relaunching failed shards: $FAILED_LIST"
    echo ""
    
    # Resubmit workers for failed shards
    WORKER_JOB=$(sbatch --parsable --array=$FAILED_LIST $0)
    echo "✅ Worker job submitted: $WORKER_JOB"
    
    # Resubmit controller to check again after workers finish
    CONTROLLER_JOB=$(sbatch --parsable --dependency=afterany:$WORKER_JOB $0)
    echo "✅ Controller job submitted: $CONTROLLER_JOB"
    
    echo ""
    echo "Automatic retry chain activated."
    echo "Monitor with: squeue -u \$USER | grep mmirage-auto-retry"

else
    # ========================================================================
    # INITIAL MODE - Submit the first job array
    # ========================================================================
    echo "=========================================="
    echo "Submitting initial MMIRAGE job array"
    echo "=========================================="
    echo "Config: $CFG"
    echo "Shards: $NUM_SHARDS (0-$((NUM_SHARDS-1)))"
    echo "Output: $SHARDS_ROOT"
    echo "Max retries: $MAX_RETRIES"
    echo ""
    
    # Submit worker array
    WORKER_JOB=$(sbatch --parsable --array=0-$((NUM_SHARDS-1)) $0)
    echo "✅ Worker job submitted: $WORKER_JOB"
    
    # Submit controller to run after workers
    CONTROLLER_JOB=$(sbatch --parsable --dependency=afterany:$WORKER_JOB $0)
    echo "✅ Controller job submitted: $CONTROLLER_JOB"
    
    echo ""
    echo "=========================================="
    echo "Jobs submitted successfully!"
    echo "=========================================="
    echo ""
    echo "The system will automatically:"
    echo "  1. Process all $NUM_SHARDS shards"
    echo "  2. Check for failures"
    echo "  3. Retry failed shards"
    echo "  4. Repeat until all succeed or max retries"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER | grep mmirage-auto-retry"
    echo ""
    echo "Cancel with:"
    echo "  scancel -n mmirage-auto-retry"
fi
