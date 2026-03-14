#!/bin/bash
set -uo pipefail

JOB_NAME="mmirage-sharded"
ACCOUNT="a127"
RESERVATION=""   # leave empty unless a real reservation exists

PROJECT_ROOT="/users/$USER/meditron/MIRAGE"
MMIRAGE_CHDIR="$PROJECT_ROOT/src/mmirage"
REPORT_DIR="/users/$USER/reports"
EDF_ENV="/users/$USER/.edf/sglang.toml"

CFG="$PROJECT_ROOT/configs/config_mock.yaml"
HF_HOME="/capstor/store/cscs/swissai/a127/homes/$USER/hf"
STATE_ROOT="$PROJECT_ROOT/tests/output/data/_pipeline_state"

NUM_SHARDS=4
MAX_RETRIES=3

NODES=1
NTASKS_PER_NODE=1
GPUS=4
CPUS_PER_TASK=288
TIME_LIMIT="11:59:59"

POLL_SECONDS=30
SETTLE_SECONDS=60
SETTLE_POLL=10

mkdir -p "$REPORT_DIR" "$HF_HOME"
LOG_FILE="$REPORT_DIR/${JOB_NAME}_logs.out"
exec > >(tee -a "$LOG_FILE") 2>&1

export CFG HF_HOME
export TOTAL_SHARDS="$NUM_SHARDS"

echo "=================================================="
echo "Pipeline   : $JOB_NAME"
echo "User       : $USER"
echo "Host       : $(hostname)"
echo "Start Time : $(date)"
echo "Log File   : $LOG_FILE"
echo "State Root : $STATE_ROOT"
echo "=================================================="

submit_array_job() {
    local array_spec="$1"
    local extra=()

    [[ -n "$RESERVATION" ]] && extra+=(--reservation="$RESERVATION")

    echo "[INFO] Submitting job array: $array_spec"

    local submitted
    submitted=$(
        sbatch --parsable \
            "${extra[@]}" \
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
                set -uo pipefail
                echo START: \$(date)
                echo HOST: \$(hostname)
                echo TASK: \$SLURM_ARRAY_TASK_ID

                srun \
                  --cpus-per-task=$CPUS_PER_TASK \
                  --wait=60 \
                  --environment=$EDF_ENV \
                  python \"$MMIRAGE_CHDIR/shard_process.py\" --config \$CFG

                rc=\$?
                echo EXIT_CODE: \$rc
                echo END: \$(date)
                exit \$rc
            "
    )

    if [[ -z "$submitted" ]]; then
        echo "[ERROR] sbatch submission failed"
        exit 1
    fi

    SUBMITTED_JOB_ID="${submitted%%;*}"
    echo "[INFO] Job ID: $SUBMITTED_JOB_ID"
}

wait_for_job() {
    local job_id="$1"

    echo "[INFO] Waiting for job array $job_id"

    while true; do
        local active
        active=$(squeue -j "$job_id" -h 2>/dev/null | wc -l)

        echo "[INFO] Poll $(date): active entries = $active"

        if [[ "$active" -eq 0 ]]; then
            break
        fi

        squeue -j "$job_id" -o "%.18i %.10T %.10M %.20R" || true
        sleep "$POLL_SECONDS"
    done

    echo "[INFO] Job array $job_id finished"
}

get_status() {
    local shard="$1"
    local status_file="$STATE_ROOT/shard_$shard/status.json"

    if [[ ! -f "$status_file" ]]; then
        echo "missing"
        return
    fi

    python - <<PY 2>/dev/null
import json
with open("$status_file") as f:
    print(json.load(f).get("status", "unknown"))
PY
    if [[ $? -ne 0 ]]; then
        echo "unknown"
    fi
}

get_retry_count() {
    local shard="$1"
    local status_file="$STATE_ROOT/shard_$shard/status.json"

    if [[ ! -f "$status_file" ]]; then
        echo 0
        return
    fi

    python - <<PY 2>/dev/null
import json
with open("$status_file") as f:
    print(int(json.load(f).get("retry_count", 0)))
PY
    if [[ $? -ne 0 ]]; then
        echo 0
    fi
}

count_state_files() {
    if [[ ! -d "$STATE_ROOT" ]]; then
        echo 0
        return
    fi

    find "$STATE_ROOT" -maxdepth 2 -name status.json 2>/dev/null | wc -l
}

wait_for_settle() {
    local waited=0

    echo "[INFO] Waiting up to ${SETTLE_SECONDS}s for shard states to settle"

    while [[ "$waited" -lt "$SETTLE_SECONDS" ]]; do
        local running=0
        local present=0

        present=$(count_state_files)

        for i in $(seq 0 $((NUM_SHARDS - 1))); do
            if [[ "$(get_status "$i")" == "running" ]]; then
                running=$((running + 1))
            fi
        done

        echo "[INFO] Settle $(date): status files=$present running_states=$running"

        if [[ "$running" -eq 0 ]]; then
            break
        fi

        sleep "$SETTLE_POLL"
        waited=$((waited + SETTLE_POLL))
    done

    echo "[INFO] Settle phase finished"
}

check_failed_shards() {
    local -n failed_ref=$1
    failed_ref=()

    local success=0
    local exhausted=0
    local running=0
    local missing=0

    echo
    echo "[INFO] Inspecting shard states"
    echo

    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        local status retry
        status="$(get_status "$i")"
        retry="$(get_retry_count "$i")"

        if [[ "$status" == "success" ]]; then
            echo "✅ shard $i: success"
            success=$((success + 1))

        elif [[ "$status" == "running" ]]; then
            echo "⏳ shard $i: still running in state file (retry=$retry)"
            running=$((running + 1))

        elif [[ "$status" == "missing" ]]; then
            echo "⚠️ shard $i: missing state file"
            missing=$((missing + 1))

        elif [[ "$retry" -ge "$MAX_RETRIES" ]]; then
            echo "🛑 shard $i: retries exhausted ($retry)"
            exhausted=$((exhausted + 1))

        else
            echo "❌ shard $i: $status (retry=$retry)"
            failed_ref+=("$i")
        fi
    done

    echo
    echo "Summary"
    echo "  success: $success / $NUM_SHARDS"
    echo "  retry: ${#failed_ref[@]}"
    echo "  still running: $running"
    echo "  missing: $missing"
    echo "  exhausted: $exhausted"
    echo

    local present
    present=$(count_state_files)

    if [[ "$present" -eq 0 ]]; then
        echo "[ERROR] No shard state files were created in: $STATE_ROOT"
        echo "[ERROR] The job finished, but no status.json files were found."
        echo "[ERROR] This usually means STATE_ROOT is wrong, or shard_process.py wrote elsewhere."
        echo "[ERROR] Refusing to auto-retry all shards."
        exit 2
    fi
}

main() {
    local failed=()
    local array_spec="0-$((NUM_SHARDS - 1))"

    while true; do
        echo "--------------------------------------------------"
        echo "[INFO] Starting iteration at $(date)"
        echo "--------------------------------------------------"

        submit_array_job "$array_spec"
        wait_for_job "$SUBMITTED_JOB_ID"
        wait_for_settle
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
