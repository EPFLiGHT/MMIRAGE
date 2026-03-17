#!/bin/bash
# MMIRAGE retry failed shards script
#
# Check for failed logical shards and relaunch them interactively.
#
# Usage:
#   bash retry_failed.sh [--config path/to/config.yaml]
#
# Configuration:
#   Set the CFG environment variable to point to your config file, or
#   use the --config argument. Defaults to configs/config_mock.yaml.
#

set -euo pipefail
IFS=$'\n\t'

# Parse command line arguments
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
    echo "❌ Config file not found: $CFG" >&2
    exit 1
fi

echo "Checking shard states from config: $CFG"
echo ""

# Use MMIRAGE CLI to check failed shards (summary only; no retry submission)
python -m mmirage.cli check --config "$CFG" --summary-only || true

echo ""
read -p "Submit retry job for failed shards? (y/N) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python -m mmirage.cli retry --config "$CFG" --no-interactive
else
    echo "Cancelled."
    exit 1
fi
