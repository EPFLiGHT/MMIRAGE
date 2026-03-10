#!/bin/bash
set -euo pipefail

# =========================================================
# MMIRAGE full pipeline wrapper
# - submits shard-processing SLURM array jobs
# - waits for completion
# - checks missing/failed shards
# - retries failed shards up to MAX_RETRIES
# - writes all terminal output to a global log file
# =========================================================

# -----------------------------
# User configuration
# -----------------------------
JOB_NAME="mmirage-sharded"
ACCOUNT="a127"
RESERVATION="sai-a127"

MMIRAGE_CHDIR="/users/qchapp/meditron/MIRAGE/src/mmirage"
REPORT_DIR="/users/qchapp/reports"
EDF_ENV="/users/qchapp/.edf/mmirage.toml"

CFG="${MMIRAGE_PATH}/configs/config_medtrinity.yaml"
# HF_HOME="${SCRATCH}/hf"
HF_HOME="/capstor/store/cscs/swissai/a127/homes/qchapp/hf"

SHARDS_ROOT="/capstor/store/cscs/swissai/a127/homes/qchapp/datasets/medtrinity/medtrinity_conversations_sampled"
NUM_SHARDS=32
MAX_RETRIES=3

# SLURM resources
NODES=1
NTASKS_PER_NODE=1
GPUS=4
CPUS_PER_TASK=288
TIME_LIMIT="11:59:59"

# Optional: poll interval while waiting for jobs
POLL_SECONDS=30

# -----------------------------
# Logging setup
# -----------------------------
mkdir -p "$REPORT_DIR"
LOG_FILE="$REPORT_DIR/${JOB_NAME}_logs.out"

# Send everything to terminal + logfile
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=================================================="
echo "Pipeline   : $JOB_NAME"
echo "User       : $USER"
echo "Host       : $(hostname)"
echo "Start Time : $(date)"
echo "Log File   : $LOG_FILE"
echo "=================================================="
echo ""

# -----------------------------
# Environment
# -----------------------------
export HF_HOME
export CFG
export TOTAL_SHARDS="$NUM_SHARDS"

mkdir -p "$HF_HOME"

echo "[INFO] Environment snapshot"
echo "  MMIRAGE_CHDIR : $MMIRAGE_CHDIR"
echo "  CFG           : $CFG"
echo "  HF_HOME       : $HF_HOME"
echo "  SHARDS_ROOT   : $SHARDS_ROOT"
echo "  NUM_SHARDS    : $NUM_SHARDS"
echo "  MAX_RETRIES   : $MAX_RETRIES"
echo ""

# -----------------------------
# Retry state
# -----------------------------
declare -A RETRY_COUNTS
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    RETRY_COUNTS[$i]=0
done

# -----------------------------
# Submit an array job
# -----------------------------
submit_array_job() {
    local array_spec="$1"

    echo "[INFO] Submitting SLURM array job for shards: $array_spec"

    local job_id
    job_id=$(
        sbatch --parsable \
            --job-name="$JOB_NAME" \
            --chdir="$MMIRAGE_CHDIR" \
            --output="$REPORT_DIR/R-%x.%A_%a.out" \
            --error="$REPORT_DIR/R-%x.%A_%a.err" \
            --nodes="$NODES" \
            --ntasks-per-node="$NTASKS_PER_NODE" \
            --gres="gpu:${GPUS}" \
            --cpus-per-task="$CPUS_PER_TASK" \
            --time="$TIME_LIMIT" \
            -A "$ACCOUNT" \
            --array="$array_spec" \
            --export=ALL,CFG="$CFG",TOTAL_SHARDS="$NUM_SHARDS",HF_HOME="$HF_HOME" \
            --wrap="
                set -euo pipefail
                export CFG='$CFG'
                export TOTAL_SHARDS='$NUM_SHARDS'
                export HF_HOME='$HF_HOME'

                echo 'START TIME: ' \$(date)
                echo 'HOST: ' \$(hostname)
                echo 'SLURM_JOB_ID: ' \$SLURM_JOB_ID
                echo 'SLURM_ARRAY_TASK_ID: ' \$SLURM_ARRAY_TASK_ID

                CMD=\"python \$MMIRAGE_PATH/src/mmirage/shard_process.py --config \$CFG\"

                SRUN_ARGS=\" \
                  --cpus-per-task $CPUS_PER_TASK \
                  --jobid \$SLURM_JOB_ID \
                  --wait 60 \
                  -A $ACCOUNT \
                  --reservation $RESERVATION \
                  --environment $EDF_ENV \
                \"

                echo \"COMMAND: \$CMD\"
                srun \$SRUN_ARGS bash -c \"\$CMD\"

                echo 'END TIME: ' \$(date)
            "
    )

    echo "[INFO] Submitted job ID: $job_id"
    SUBMITTED_JOB_ID="$job_id"
}

# -----------------------------
# Wait for job completion
# -----------------------------
wait_for_job() {
    local job_id="$1"

    echo "[INFO] Waiting for job $job_id to finish..."

    while squeue -h -j "$job_id" | grep -q .; do
        echo "[INFO] Job $job_id still running or pending at $(date)"
        sleep "$POLL_SECONDS"
    done

    echo "[INFO] Job $job_id finished at $(date)"
}

# -----------------------------
# Check shard results
# Populates the named array passed as $1
# -----------------------------
check_failed_shards() {
    local -n out_failed_shards="$1"
    out_failed_shards=()

    local success_count=0
    local exhausted_count=0

    echo ""
    echo "[INFO] Checking shard status in: $SHARDS_ROOT"
    echo ""

    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        mapfile -t shard_dirs < <(find "$SHARDS_ROOT" -type d -name "shard_$i" 2>/dev/null)

        if [ "${#shard_dirs[@]}" -eq 0 ]; then
            if [ "${RETRY_COUNTS[$i]}" -ge "$MAX_RETRIES" ]; then
                echo "🛑 Shard $i: MISSING, max retries exceeded (${RETRY_COUNTS[$i]}/$MAX_RETRIES)"
                ((exhausted_count+=1))
            else
                echo "❌ Shard $i: MISSING"
                out_failed_shards+=("$i")
            fi
            continue
        fi

        local shard_success=false
        for shard_dir in "${shard_dirs[@]}"; do
            if [ -f "$shard_dir/.SUCCESS" ]; then
                shard_success=true
                break
            fi
        done

        if [ "$shard_success" = true ]; then
            echo "✅ Shard $i: SUCCESS"
            ((success_count+=1))
        else
            if [ "${RETRY_COUNTS[$i]}" -ge "$MAX_RETRIES" ]; then
                echo "🛑 Shard $i: FAILED, max retries exceeded (${RETRY_COUNTS[$i]}/$MAX_RETRIES)"
                ((exhausted_count+=1))
            else
                echo "❌ Shard $i: FAILED (retries used: ${RETRY_COUNTS[$i]}/$MAX_RETRIES)"
                out_failed_shards+=("$i")
            fi
        fi
    done

    echo ""
    echo "=================================================="
    echo "Summary"
    echo "  Successful           : $success_count / $NUM_SHARDS"
    echo "  Failed to retry      : ${#out_failed_shards[@]}"
    echo "  Retry budget expired : $exhausted_count"
    echo "=================================================="
    echo ""
}

# -----------------------------
# Main loop
# -----------------------------
main() {
    local failed_shards=()
    local retryable_shards=()
    local array_spec="0-$((NUM_SHARDS - 1))"
    local iteration=0

    while true; do
        ((iteration+=1))
        echo "--------------------------------------------------"
        echo "[INFO] Iteration $iteration started at $(date)"
        echo "--------------------------------------------------"

        submit_array_job "$array_spec"
        wait_for_job "$SUBMITTED_JOB_ID"

        check_failed_shards failed_shards

        if [ "${#failed_shards[@]}" -eq 0 ]; then
            echo "🎉 All shards completed successfully!"
            echo ""
            echo "=================================================="
            echo "Pipeline finished successfully at: $(date)"
            echo "=================================================="
            exit 0
        fi

        retryable_shards=()
        for shard in "${failed_shards[@]}"; do
            RETRY_COUNTS[$shard]=$((RETRY_COUNTS[$shard] + 1))
            if [ "${RETRY_COUNTS[$shard]}" -le "$MAX_RETRIES" ]; then
                retryable_shards+=("$shard")
            fi
        done

        if [ "${#retryable_shards[@]}" -eq 0 ]; then
            echo "🛑 No retryable failed shards remain."
            echo ""
            echo "[INFO] Final retry counters:"
            for i in $(seq 0 $((NUM_SHARDS - 1))); do
                echo "  Shard $i -> ${RETRY_COUNTS[$i]}"
            done
            echo ""
            echo "=================================================="
            echo "Pipeline finished with failures at: $(date)"
            echo "=================================================="
            exit 1
        fi

        array_spec=$(IFS=,; echo "${retryable_shards[*]}")
        echo "[INFO] Retrying failed shards: $array_spec"
        echo ""
    done
}

main