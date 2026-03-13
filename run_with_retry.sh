#!/bin/bash
set -uo pipefail

JOB_NAME="mmirage-sharded"
ACCOUNT="a127"
RESERVATION="" # e.g. "sai-a127" if needed

MMIRAGE_CHDIR="/users/$USER/meditron/MMIRAGE/src/mmirage"
REPORT_DIR="/users/$USER/reports"
EDF_ENV="/users/$USER/.edf/mmirage.toml"

CFG="/users/$USER/meditron/MMIRAGE/configs/config_mock.yaml"
HF_HOME="/capstor/store/cscs/swissai/a127/homes/$USER/hf"
STATE_ROOT="/users/$USER/meditron/MMIRAGE/tests/output/data/_pipeline_state"

NUM_SHARDS=32
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
echo "=================================================="

submit_array_job() {
    local array_spec="$1"
    local extra=()

    if [[ -n "$RESERVATION" ]]; then
        extra+=(--reservation="$RESERVATION")
    fi

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
                echo PYTHON_EXIT: \$rc
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
        local n
        n=$(squeue -j "$job_id" -h 2>/dev/null | wc -l)

        if [[ "$n" -eq 0 ]]; then
            break
        fi

        squeue -j "$job_id" -o "%.18i %.10T %.10M %.20R"
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

wait_for_settle() {
    local waited=0

    echo "[INFO] Waiting up to ${SETTLE_SECONDS}s for shard states to settle"

    while [[ "$waited" -lt "$SETTLE_SECONDS" ]]; do
        local running=0

        for i in $(seq 0 $((NUM_SHARDS - 1))); do
            if [[ "$(get_status "$i")" == "running" ]]; then
                running=$((running + 1))
            fi
        done

        if [[ "$running" -eq 0 ]]; then
            echo "[INFO] State files settled"
            return
        fi

        echo "[INFO] $running shard(s) still marked running"
        sleep "$SETTLE_POLL"
        waited=$((waited + SETTLE_POLL))
    done

    echo "[INFO] Continuing after settle timeout"
}

check_failed_shards() {
    local -n failed_ref=$1
    failed_ref=()

    local success=0
    local exhausted=0
    local running=0

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
    echo "  exhausted: $exhausted"
    echo
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