#!/bin/bash
# Check for failed logical shards and relaunch them
#
# Usage: bash retry_failed.sh

set -euo pipefail

STATE_ROOT="/users/$USER/meditron/MMIRAGE/tests/output/data/_pipeline_state"
NUM_SHARDS=32
MAX_RETRIES=3
SCRIPT_PATH="/users/$USER/meditron/MMIRAGE/run.sh"

echo "Checking shard states in: $STATE_ROOT"
echo ""

failed_shards=()
success_count=0

for i in $(seq 0 $((NUM_SHARDS - 1))); do
    state_dir="$STATE_ROOT/shard_$i"
    status_file="$state_dir/status.json"

    if [ ! -f "$status_file" ]; then
        echo "❌ Shard $i: MISSING STATUS"
        failed_shards+=("$i")
        continue
    fi

    status=$(python - <<PY
import json
with open("$status_file", "r") as f:
    data = json.load(f)
print(data.get("status", "unknown"))
PY
)

    retry_count=$(python - <<PY
import json
with open("$status_file", "r") as f:
    data = json.load(f)
print(int(data.get("retry_count", 0)))
PY
)

    if [ "$status" = "success" ]; then
        echo "✅ Shard $i: SUCCESS"
        success_count=$((success_count + 1))
    elif [ "$retry_count" -ge "$MAX_RETRIES" ]; then
        echo "🛑 Shard $i: MAX RETRIES EXCEEDED ($retry_count/$MAX_RETRIES)"
    else
        echo "❌ Shard $i: $status (retries: $retry_count/$MAX_RETRIES)"
        failed_shards+=("$i")
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

ARRAY_SPEC=$(IFS=,; echo "${failed_shards[*]}")

echo "Failed shards: $ARRAY_SPEC"
echo ""
read -p "Submit retry job for these shards? (y/N) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    JOB_ID=$(sbatch --array="$ARRAY_SPEC" "$SCRIPT_PATH" | grep -oE '[0-9]+')
    echo "✅ Job submitted: $JOB_ID"
    echo ""
    echo "Monitor with: squeue -j $JOB_ID"
else
    echo "Cancelled."
fi