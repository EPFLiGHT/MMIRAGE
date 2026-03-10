"""Main script for processing dataset shards with MMIRAGE.

Supports both text-only and multimodal (vision-language) processing.
"""

import argparse
from datetime import datetime
from functools import reduce
import json
import logging
import os
import shutil
import socket
import sys
import traceback
from typing import Any, Dict, List

from datasets import DatasetDict

from mmirage.config.utils import load_mmirage_config
from mmirage.core.loader.base import BaseDataLoaderConfig, DatasetLike
from mmirage.core.loader.utils import load_datasets_from_configs
from mmirage.core.process.mapper import MMIRAGEMapper
from mmirage.core.writer.renderer import TemplateRenderer

logger = logging.getLogger(__name__)


def _count_rows(ds: DatasetLike) -> int:
    """Count total rows in a dataset or dataset dict."""
    if isinstance(ds, DatasetDict):
        return sum(len(split) for split in ds.values())
    return len(ds)


def _dataset_out_dir(shard_idx: int, ds_config: BaseDataLoaderConfig) -> str:
    """Get dataset-specific output directory for a shard."""
    return os.path.join(ds_config.output_dir, f"shard_{shard_idx}")


def _shard_state_dir(shard_idx: int, state_root: str) -> str:
    """Get central state directory for a logical shard."""
    return os.path.join(state_root, f"shard_{shard_idx}")


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


def _cleanup_old_shard_data(out_dir: str):
    """Remove old dataset shard output before retry."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        logger.info(f"Removed old shard output: {out_dir}")


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


def rewrite_batch(
    batch: Dict[str, List[Any]],
    mapper: MMIRAGEMapper,
    renderer: TemplateRenderer,
    image_base_path: str = None,
) -> Dict[str, List[Any]]:
    """Rewrite a batch of samples by applying transformations."""
    if not mapper.validate_vars():
        raise ValueError(
            "Uncomputable variables detected. Verify your configuration and make sure that there is no undefined variables"
        )

    batch_environment = mapper.rewrite_batch(batch, image_base_path)
    rendered_list = renderer.batch_render(batch_environment)
    return rendered_list


def _get_state_root(cfg) -> str:
    """Get the shared pipeline state root from config."""
    state_dir = getattr(cfg.loading_params, "state_dir", None)
    if not state_dir:
        raise ValueError(
            "loading_params.state_dir must be set when using multiple datasets with independent output_dir values"
        )
    return state_dir


def main():
    """Process a single logical shard across all configured datasets."""
    ap = argparse.ArgumentParser("Process dataset shards using MMIRAGE with SGLang.")
    ap.add_argument(
        "--config",
        help="YAML config for MMIRAGE pipeline.",
        required=True,
    )
    args = ap.parse_args()

    cfg = load_mmirage_config(args.config)
    loading_params = cfg.loading_params
    processing_params = cfg.processing_params
    datasets_config = loading_params.datasets

    if not datasets_config:
        raise ValueError("No datasets provided in config.loading_params.datasets")

    shard_id = loading_params.get_shard_id()
    num_shards = loading_params.get_num_shards()

    if not (0 <= shard_id < num_shards):
        raise ValueError(f"Invalid shard_id={shard_id}, num_shards={num_shards}")

    state_root = _get_state_root(cfg)
    state_dir = _shard_state_dir(shard_id, state_root)

    try:
        retry_count = _mark_running(state_dir, shard_id, datasets_config)
        logger.info(f"Starting shard {shard_id}/{num_shards - 1} (attempt #{retry_count})")

        if retry_count > 1:
            for ds_config in datasets_config:
                out_dir = _dataset_out_dir(shard_id, ds_config)
                _cleanup_old_shard_data(out_dir)

        ds_all = load_datasets_from_configs(datasets_config)
        total_rows = sum(_count_rows(ds) for ds in ds_all)

        ds_all_shard = [_shard_dataset(ds, num_shards, shard_id) for ds in ds_all]
        shard_rows = sum(_count_rows(ds) for ds in ds_all_shard)

        logger.info(
            f"Loaded {len(datasets_config)} dataset(s): {datasets_config} "
            f"→ {total_rows} total rows; this logical shard has {shard_rows} rows."
        )

        mapper = MMIRAGEMapper(
            cfg.processors,
            processing_params.inputs,
            processing_params.outputs,
        )
        renderer = TemplateRenderer(processing_params.output_schema)

        ds_processed_all: List[DatasetLike] = []
        for ds_idx, ds_shard in enumerate(ds_all_shard):
            ds_config = datasets_config[ds_idx]
            remove_columns = _remove_columns(ds_shard, processing_params.remove_columns)

            logger.info(
                f"Processing dataset {ds_idx} for shard {shard_id}: "
                f"path={ds_config.path}, output_dir={ds_config.output_dir}"
            )

            ds_processed = ds_shard.map(
                rewrite_batch,
                batched=True,
                batch_size=loading_params.get_batch_size(),
                load_from_cache_file=False,
                desc=f"Shard {shard_id}/{num_shards - 1} dataset {ds_idx}",
                fn_kwargs={
                    "mapper": mapper,
                    "renderer": renderer,
                    "image_base_path": ds_config.image_base_path,
                },
                remove_columns=remove_columns,
            )
            ds_processed_all.append(ds_processed)

        for ds_idx, (ds_config, ds_processed) in enumerate(zip(datasets_config, ds_processed_all)):
            out_dir = _dataset_out_dir(shard_id, ds_config)
            _save_dataset_atomic(ds_processed, out_dir)
            logger.info(f"✅ Saved dataset {ds_idx} shard in: {out_dir}")

        _mark_success(state_dir)
        logger.info(f"✅ Logical shard {shard_id} completed successfully")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"❌ Shard {shard_id} failed: {error_msg}")
        logger.error(traceback.format_exc())
        _mark_failure(state_dir, error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()