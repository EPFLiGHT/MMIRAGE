#!/bin/bash
# Check for failed shards and relaunch them
#
# Usage: ./retry_failed.sh

# Configuration
SHARDS_ROOT="/capstor/store/cscs/swissai/a127/homes/qchapp/datasets/medtrinity/medtrinity_conversations_sampled"
NUM_SHARDS=32
MAX_RETRIES=3
SCRIPT_PATH="/users/qchapp/meditron/MIRAGE/run_with_retry.sh"

echo "Checking for failed shards in: $SHARDS_ROOT"
echo ""

failed_shards=()
success_count=0

for i in $(seq 0 $((NUM_SHARDS-1))); do
    # Find shard directories (may be nested under dataset dirs)
    shard_dirs=$(find "$SHARDS_ROOT" -type d -name "shard_$i" 2>/dev/null)
    
    if [ -z "$shard_dirs" ]; then
        echo "❌ Shard $i: MISSING"
        failed_shards+=($i)
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
        fi
    fi
done

echo ""
echo "=========================================="
echo "Summary:"
echo "  ✅ Successful: $success_count / $NUM_SHARDS"
echo "  ❌ To retry: ${#failed_shards[@]}"
echo "=========================================="
echo ""

if [ ${#failed_shards[@]} -eq 0 ]; then
    echo "🎉 All shards completed successfully!"
    exit 0
fi

# Build array spec
IFS=','
ARRAY_SPEC="${failed_shards[*]}"
unset IFS

echo "Failed shards: $ARRAY_SPEC"
echo ""
read -p "Submit retry job for these shards? (y/N) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    JOB_ID=$(sbatch --array=$ARRAY_SPEC "$SCRIPT_PATH" | grep -oE '[0-9]+')
    echo "✅ Job submitted: $JOB_ID"
    echo ""
    echo "Monitor with: squeue -j $JOB_ID"
else
    echo "Cancelled."
fi
