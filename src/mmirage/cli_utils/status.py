"""Shard status and retry helpers for the MMIRAGE CLI."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from mmirage.config.config import MMirageConfig
from mmirage.cli_utils.slurm import submit_slurm_job


logger = logging.getLogger(__name__)


@dataclass
class ShardSummary:
    """Compact status summary for shard execution."""

    total: int
    successful: int
    running: int
    failed: int
    max_retries_exceeded: int


def get_shard_state_dir(state_root: str, shard_id: int) -> str:
    """Return the state directory for a shard."""
    return os.path.join(state_root, f"shard_{shard_id}")


def get_shard_status(state_dir: str) -> Tuple[str, int]:
    """Read the current status and retry count for a shard."""
    status_file = os.path.join(state_dir, "status.json")
    if not os.path.exists(status_file):
        return ("missing", 0)

    try:
        with open(status_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read shard status from %s: %s", status_file, exc)
        return ("unknown", 0)

    return (str(data.get("status", "unknown")), int(data.get("retry_count", 0)))


def check_failed_shards(cfg: MMirageConfig) -> Tuple[List[int], ShardSummary]:
    """Return retryable failed shards and a compact summary."""
    state_root = cfg.loading_params.get_state_root()

    num_shards = cfg.loading_params.get_num_shards()
    max_retries = cfg.execution_params.max_retries
    failed_shards: List[int] = []
    success_count = 0
    running_count = 0
    exhausted_count = 0

    for shard_id in range(num_shards):
        status, retry_count = get_shard_status(get_shard_state_dir(state_root, shard_id))
        if status == "success":
            success_count += 1
            continue

        if status == "running":
            running_count += 1
            continue
        
        retries_used = max(retry_count - 1, 0)
        if retries_used >= max_retries:
            exhausted_count += 1
            logger.warning(
                "Shard %s exceeded retry budget (%s/%s)",
                shard_id,
                retries_used,
                max_retries,
            )
            continue

        failed_shards.append(shard_id)

    summary = ShardSummary(
        total=num_shards,
        successful=success_count,
        running=running_count,
        failed=len(failed_shards),
        max_retries_exceeded=exhausted_count,
    )
    return failed_shards, summary


def confirm_retry(count: int, interactive: bool) -> bool:
    """Return whether retry submission is confirmed."""
    if not interactive:
        return True
    if not sys.stdin.isatty():
        logger.error("Non-interactive input detected; use --no-interactive")
        return False
    response = input(f"Retry {count} shard(s)? (y/N) ")
    return response.strip().lower() == "y"


def status_exit_code(failed_shards: Sequence[int], summary: ShardSummary) -> int:
    """Map shard status to an exit code."""
    return (
        0
        if not failed_shards
        and summary.max_retries_exceeded == 0
        and summary.running == 0
        and summary.successful == summary.total
        else 1
    )


def submit_failed_shards(
    cfg: MMirageConfig,
    config_path: str,
    failed_shards: Sequence[int],
    interactive: bool,
) -> int:
    """Submit retry jobs for failed shards when requested."""
    if not failed_shards:
        return 0

    if not confirm_retry(len(failed_shards), interactive):
        return 1

    job_id = submit_slurm_job(cfg, config_path, failed_shards)
    if job_id is None:
        return 1

    print(job_id)
    return 0
