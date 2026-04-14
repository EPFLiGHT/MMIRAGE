"""Utility functions for shard and merge processing.

This module contains helper functions for dataset sharding, state management,
and file operations used in the MMIRAGE shard processing pipeline.
"""

from datetime import datetime
from dataclasses import dataclass
import json
import logging
import os
import shutil
import socket
import uuid
from typing import Any, Dict, List, Optional

from datasets import DatasetDict

from mmirage.core.loader.base import BaseDataLoaderConfig, DatasetLike

logger = logging.getLogger(__name__)


@dataclass
class ShardStatus:
    """Typed representation of the shard status.json payload."""

    status: str = "unknown"
    retry_count: int = 0
    shard_id: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    hostname: Optional[str] = None
    pid: Optional[int] = None
    slurm_job_id: Optional[str] = None
    slurm_array_task_id: Optional[str] = None
    datasets: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ShardStatus":
        """Build a status object from a JSON payload."""
        data = payload or {}
        try:
            retry_count = int(data.get("retry_count", 0))
        except (TypeError, ValueError):
            retry_count = 0

        shard_id = data.get("shard_id")
        if shard_id is not None:
            try:
                shard_id = int(shard_id)
            except (TypeError, ValueError):
                shard_id = None

        pid = data.get("pid")
        if pid is not None:
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                pid = None

        datasets = data.get("datasets")
        if not isinstance(datasets, list):
            datasets = None

        return cls(
            status=str(data.get("status", "unknown")),
            retry_count=retry_count,
            shard_id=shard_id,
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            error=data.get("error"),
            hostname=data.get("hostname"),
            pid=pid,
            slurm_job_id=data.get("slurm_job_id"),
            slurm_array_task_id=data.get("slurm_array_task_id"),
            datasets=datasets,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize status to the JSON payload written on disk."""
        return {
            "status": self.status,
            "retry_count": self.retry_count,
            "shard_id": self.shard_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "hostname": self.hostname,
            "pid": self.pid,
            "slurm_job_id": self.slurm_job_id,
            "slurm_array_task_id": self.slurm_array_task_id,
            "datasets": self.datasets,
        }


@dataclass
class MergeReport:
    """Summary of a merge operation for one dataset directory."""

    dataset_name: str
    input_dir: str
    output_dir: str
    used_shards: int
    merged_rows: int
    skipped_invalid_dirs: int
    skipped_zero_rows: int


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


def _remove_columns(ds: DatasetLike) -> List[str]:
    """Get columns to remove from dataset if enabled."""
    if isinstance(ds, DatasetDict):
        return list(set(x for split_ds in ds.values() for x in split_ds.column_names))
    return ds.column_names


def _save_dataset_atomic(ds_processed: DatasetLike, out_dir: str):
    """Save dataset atomically via temporary directory + rename."""
    parent_dir = os.path.dirname(out_dir)
    os.makedirs(parent_dir, exist_ok=True)

    tmp_dir = (
        f"{out_dir}.tmp.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex}"
    )
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    ds_processed.save_to_disk(tmp_dir)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.replace(tmp_dir, out_dir)


def _validate_safe_output_dir(dataset_dir: str, output_dir: str) -> None:
    """Reject output paths that could delete input data.

    We forbid output directories that are the same as, or ancestors of,
    the input dataset directory. This prevents accidental deletion when
    clearing pre-existing output_dir before writing merged data.
    """
    dataset_real = os.path.realpath(os.path.abspath(dataset_dir))
    output_real = os.path.realpath(os.path.abspath(output_dir))

    if output_real == dataset_real:
        raise RuntimeError(
            "Unsafe merge output path: output_dir equals dataset_dir "
            f"(dataset_dir={dataset_real}, output_dir={output_real})."
        )

    try:
        common = os.path.commonpath([dataset_real, output_real])
    except ValueError:
        # Different drives (Windows) -> no ancestor relationship possible
        return

    if common == output_real:
        raise RuntimeError(
            "Unsafe merge output path: output_dir contains dataset_dir "
            f"(dataset_dir={dataset_real}, output_dir={output_real})."
        )


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


def _read_status(state_dir: str) -> ShardStatus:
    """Read status.json if present."""
    path = _status_file(state_dir)
    if not os.path.exists(path):
        return ShardStatus(status="missing")
    try:
        with open(path, "r") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                logger.warning(f"Invalid status format in {path}; expected object")
                return ShardStatus(status="unknown")
            return ShardStatus.from_dict(data)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read status file {path}: {e}")
        return ShardStatus(status="unknown")


def _write_status(state_dir: str, payload: ShardStatus):
    """Atomically write status.json."""
    os.makedirs(state_dir, exist_ok=True)
    tmp_path = _status_file(state_dir) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload.to_dict(), f, indent=2, sort_keys=True)
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
    retry_count = prev.retry_count + 1

    payload = ShardStatus(
        status="running",
        retry_count=retry_count,
        shard_id=shard_id,
        started_at=datetime.now().isoformat(),
        finished_at=None,
        error=None,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        datasets=[
            {
                "path": ds_config.path,
                "output_dir": ds_config.output_dir,
            }
            for ds_config in datasets_config
        ],
    )

    _write_status(state_dir, payload)
    _clear_markers(state_dir)
    _touch_marker(state_dir, ".RUNNING")
    return retry_count


def _mark_success(state_dir: str):
    """Mark shard as successful."""
    prev = _read_status(state_dir)
    prev.status = "success"
    prev.finished_at = datetime.now().isoformat()
    prev.error = None
    _write_status(state_dir, prev)
    _clear_markers(state_dir)
    _touch_marker(state_dir, ".SUCCESS")


def _mark_failure(state_dir: str, error_msg: str):
    """Mark shard as failed."""
    prev = _read_status(state_dir)
    prev.status = "failed"
    prev.finished_at = datetime.now().isoformat()
    prev.error = error_msg
    _write_status(state_dir, prev)
    _clear_markers(state_dir)
    _touch_marker(state_dir, ".FAILED")


def _list_shard_dirs(dataset_dir: str) -> List[str]:
    """List shard directories in a dataset directory."""
    shard_dirs: List[str] = []
    for name in os.listdir(dataset_dir):
        if not name.startswith("shard_"):
            continue
        # Only accept canonical shard directories of the form "shard_<int>"
        # and explicitly skip atomic-save temp dirs like
        # "shard_0.tmp.<host>.<pid>.<uuid>".
        if ".tmp" in name:
            continue
        suffix = name[len("shard_") :]
        if not suffix.isdigit():
            continue
        path = os.path.join(dataset_dir, name)
        if os.path.isdir(path):
            shard_dirs.append(path)

    def _shard_key(path: str) -> int:
        base = os.path.basename(path)
        suffix = base.removeprefix("shard_")
        return int(suffix) if suffix.isdigit() else 0

    shard_dirs.sort(key=_shard_key)
    return shard_dirs


def _dataset_dirs(input_dir: str) -> List[str]:
    """Find dataset directories containing shard folders."""
    candidates: List[str] = []
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if not os.path.isdir(path):
            continue
        if _list_shard_dirs(path):
            candidates.append(path)
    return sorted(candidates)

def _validate_input_dir(path: str, arg_name: str) -> None:
    """Ensure a user-provided input path exists and is a directory."""
    normalized = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
    if not os.path.isdir(normalized):
        raise RuntimeError(
            f"{arg_name} does not exist or is not a directory: {normalized}"
        )