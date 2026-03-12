#!/bin/bash
set -euo pipefail

# =========================================================
# MMIRAGE pipeline controller
# - submit shard-processing SLURM array
# - wait for completion
# - inspect shard state_dir
# - retry failed shards
# =========================================================

# -----------------------------
# User configuration
# -----------------------------
JOB_NAME="mmirage-sharded"
ACCOUNT="a127"
RESERVATION="sai-a127"

MMIRAGE_CHDIR="/users/qchapp/meditron/MIRAGE/src/mmirage"
REPORT_DIR="/users/qchapp/reports"
EDF_ENV="/users/qchapp/.edf/sglang.toml"

CFG="/users/qchapp/meditron/MIRAGE/configs/config_medtrinity.yaml"
HF_HOME="/capstor/store/cscs/swissai/a127/homes/qchapp/hf"

STATE_ROOT="/capstor/store/cscs/swissai/a127/homes/qchapp/datasets/medtrinity/_pipeline_state"

NUM_SHARDS=32
MAX_RETRIES=3

# SLURM resources
NODES=1
NTASKS_PER_NODE=1
GPUS=4
CPUS_PER_TASK=288
TIME_LIMIT="11:59:59"

POLL_SECONDS=30

# -----------------------------
# Logging
# -----------------------------
mkdir -p "$REPORT_DIR"
LOG_FILE="$REPORT_DIR/${JOB_NAME}_logs.out"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=================================================="
echo "Pipeline   : $JOB_NAME"
echo "User       : $USER"
echo "Host       : $(hostname)"
echo "Start Time : $(date)"
echo "Log File   : $LOG_FILE"
echo "=================================================="

# -----------------------------
# Environment
# -----------------------------
export HF_HOME
export CFG
export TOTAL_SHARDS="$NUM_SHARDS"

mkdir -p "$HF_HOME"

# -----------------------------
# Submit SLURM array job
# -----------------------------
submit_array_job() {

    local array_spec="$1"

    echo "[INFO] Submitting job array: $array_spec"

    SUBMITTED_JOB_ID=$(
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

                echo 'START:' \$(date)
                echo 'HOST:' \$(hostname)
                echo 'TASK:' \$SLURM_ARRAY_TASK_ID

                srun \
                  --cpus-per-task $CPUS_PER_TASK \
                  --jobid \$SLURM_JOB_ID \
                  --wait 60 \
                  -A $ACCOUNT \
                  --reservation $RESERVATION \
                  --environment $EDF_ENV \
                  python shard_process.py --config \$CFG

                echo 'END:' \$(date)
            "
    )

    echo "[INFO] Submitted job ID: $SUBMITTED_JOB_ID"
}

# -----------------------------
# Wait for job completion
# -----------------------------
wait_for_job() {

    local job_id="$1"

    echo "[INFO] Waiting for job $job_id"

    while squeue -h -j "$job_id" | grep -q .; do
        sleep "$POLL_SECONDS"
    done

    echo "[INFO] Job finished"
}

# -----------------------------
# Inspect shard states
# -----------------------------
check_failed_shards() {

    local -n result="$1"
    result=()

    success=0
    exhausted=0

    echo ""
    echo "[INFO] Inspecting shard states"
    echo ""

    for i in $(seq 0 $((NUM_SHARDS-1))); do

        status_file="$STATE_ROOT/shard_$i/status.json"

        if [[ ! -f "$status_file" ]]; then
            echo "❌ shard $i: missing state"
            result+=("$i")
            continue
        fi

        status=$(python - <<PY
import json
with open("$status_file") as f:
    print(json.load(f).get("status"))
PY
)

        retry=$(python - <<PY
import json
with open("$status_file") as f:
    print(json.load(f).get("retry_count",0))
PY
)

        if [[ "$status" == "success" ]]; then
            echo "✅ shard $i: success"
            ((success+=1))

        elif [[ "$retry" -ge "$MAX_RETRIES" ]]; then
            echo "🛑 shard $i: retries exhausted ($retry)"
            ((exhausted+=1))

        else
            echo "❌ shard $i: $status (retry=$retry)"
            result+=("$i")
        fi
    done

    echo ""
    echo "Summary"
    echo "  success: $success / $NUM_SHARDS"
    echo "  retry: ${#result[@]}"
    echo "  exhausted: $exhausted"
    echo ""
}

# -----------------------------
# Main loop
# -----------------------------
main() {

    local failed=()
    local array_spec="0-$((NUM_SHARDS-1))"

    while true; do

        echo "--------------------------------------------------"
        echo "[INFO] Starting iteration at $(date)"
        echo "--------------------------------------------------"

        submit_array_job "$array_spec"

        wait_for_job "$SUBMITTED_JOB_ID"

        check_failed_shards failed

        if [[ ${#failed[@]} -eq 0 ]]; then
            echo "🎉 Pipeline completed successfully"
            exit 0
        fi

        array_spec=$(IFS=,; echo "${failed[*]}")

        echo "[INFO] Retrying shards: $array_spec"
    done
}

main