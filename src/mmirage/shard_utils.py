"""Utility functions for shard processing.

This module contains helper functions for dataset sharding, state management,
and file operations used in the MMIRAGE shard processing pipeline.
"""

from datetime import datetime
from functools import reduce
import json
import logging
import os
import shutil
import socket
from typing import Any, Dict, List

from datasets import DatasetDict

from mmirage.core.loader.base import BaseDataLoaderConfig, DatasetLike

logger = logging.getLogger(__name__)


def _count_rows(ds: DatasetLike) -> int:
    """Count total rows in a dataset or dataset dict."""
    if isinstance(ds, DatasetDict):
        return sum(len(split) for split in ds.values())
    return len(ds)


def _shard_dataset(ds: DatasetLike, num_shards: int, shard_id: int) -> DatasetLike:
    """Shard a dataset or dataset dict."""
    if isinstance(ds, DatasetDict):
        return DatasetDict(
            {
                split: split_ds.shard(num_shards=num_shards, index=shard_id)
                for split, split_ds in ds.items()
            }
        )
    return ds.shard(num_shards=num_shards, index=shard_id)


def _remove_columns(ds: DatasetLike, enable: bool) -> List[str]:
    """Get columns to remove from dataset if enabled."""
    if not enable:
        return []
    if isinstance(ds, DatasetDict):
        columns_set = [set(split_ds.column_names) for split_ds in ds.values()]
        return list(reduce(lambda x, y: x | y, columns_set))
    return ds.column_names


def _save_dataset_atomic(ds_processed: DatasetLike, out_dir: str):
    """Save dataset atomically via temporary directory + rename."""
    parent_dir = os.path.dirname(out_dir)
    os.makedirs(parent_dir, exist_ok=True)

    tmp_dir = f"{out_dir}.tmp.{os.getpid()}"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    ds_processed.save_to_disk(tmp_dir)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.replace(tmp_dir, out_dir)


def _dataset_out_dir(shard_idx: int, ds_config: BaseDataLoaderConfig) -> str:
    """Get dataset-specific output directory for a shard."""
    return os.path.join(ds_config.output_dir, f"shard_{shard_idx}")


def _shard_state_dir(shard_idx: int, state_root: str) -> str:
    """Get central state directory for a logical shard."""
    return os.path.join(state_root, f"shard_{shard_idx}")


def _cleanup_old_shard_data(out_dir: str):
    """Remove old dataset shard output before retry."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        logger.info(f"Removed old shard output: {out_dir}")


def _status_file(state_dir: str) -> str:
    """Canonical status file path."""
    return os.path.join(state_dir, "status.json")


def _read_status(state_dir: str) -> dict:
    """Read status.json if present."""
    path = _status_file(state_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read status file {path}: {e}")
        return {}


def _write_status(state_dir: str, payload: dict):
    """Atomically write status.json."""
    os.makedirs(state_dir, exist_ok=True)
    tmp_path = _status_file(state_dir) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, _status_file(state_dir))


def _clear_markers(state_dir: str):
    """Remove status marker files."""
    for name in (".RUNNING", ".SUCCESS", ".FAILED"):
        path = os.path.join(state_dir, name)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as e:
                logger.warning(f"Failed to remove marker {path}: {e}")


def _touch_marker(state_dir: str, name: str):
    """Create a marker file."""
    os.makedirs(state_dir, exist_ok=True)
    path = os.path.join(state_dir, name)
    with open(path, "w") as f:
        f.write(f"{datetime.now().isoformat()}\n")


def _mark_running(
    state_dir: str,
    shard_id: int,
    datasets_config: List[BaseDataLoaderConfig],
) -> int:
    """Mark shard as running and increment retry count."""
    prev = _read_status(state_dir)
    retry_count = int(prev.get("retry_count", 0)) + 1

    payload = {
        "status": "running",
        "retry_count": retry_count,
        "shard_id": shard_id,
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "error": None,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "datasets": [
            {
                "path": ds_config.path,
                "output_dir": ds_config.output_dir,
            }
            for ds_config in datasets_config
        ],
    }

    _write_status(state_dir, payload)
    _clear_markers(state_dir)
    _touch_marker(state_dir, ".RUNNING")
    return retry_count


def _mark_success(state_dir: str):
    """Mark shard as successful."""
    prev = _read_status(state_dir)
    prev["status"] = "success"
    prev["finished_at"] = datetime.now().isoformat()
    prev["error"] = None
    _write_status(state_dir, prev)
    _clear_markers(state_dir)
    _touch_marker(state_dir, ".SUCCESS")


def _mark_failure(state_dir: str, error_msg: str):
    """Mark shard as failed."""
    prev = _read_status(state_dir)
    prev["status"] = "failed"
    prev["finished_at"] = datetime.now().isoformat()
    prev["error"] = error_msg
    _write_status(state_dir, prev)
    _clear_markers(state_dir)
    _touch_marker(state_dir, ".FAILED")
