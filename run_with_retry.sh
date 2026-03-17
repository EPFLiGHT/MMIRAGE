#!/bin/bash
# MMIRAGE pipeline orchestration with forced automatic retry.
#
# Usage:
#   bash run_with_retry.sh [--config path/to/config.yaml]

set -euo pipefail
IFS=$'\n\t'

CFG="${CFG:-configs/config_mock.yaml}"

while (( $# > 0 )); do
    case "$1" in
        --config)
            CFG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "$CFG" ]]; then
    echo "Config file not found: $CFG" >&2
    exit 1
fi

echo "Config: $CFG"
python -m mmirage.cli run --config "$CFG" --force-retry
