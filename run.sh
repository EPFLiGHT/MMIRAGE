#!/bin/bash
# MMIRAGE launch script. MMIRAGE should now be ran with a single command, and the behavior is driven by the config file.
#
# Launch behavior is driven by the config file:
# - execution_params.retry=false: submit one SLURM array job, or run locally
# - execution_params.retry=true: submit and automatically retry failed shards
#
# Usage:
#   bash run.sh
#   CFG=configs/config_mock.yaml bash run.sh

set -euo pipefail

CFG="${CFG:-configs/config_mock.yaml}"

if [[ ! -f "$CFG" ]]; then
    echo "Config file not found: $CFG" >&2
    exit 1
fi

python -m mmirage.cli run --config "$CFG"

echo "END TIME: $(date)"
