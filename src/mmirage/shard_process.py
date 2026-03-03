"""Main script for processing dataset shards with MMIRAGE.

Supports both text-only and multimodal (vision-language) processing.
"""

import argparse
from datetime import datetime
from functools import reduce
import os
import sys
import traceback
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict

from mmirage.core.loader.base import BaseDataLoaderConfig, DatasetLike
from mmirage.core.process.mapper import MMIRAGEMapper

from mmirage.config.utils import load_mmirage_config
from mmirage.core.writer.renderer import TemplateRenderer
from mmirage.core.loader.utils import load_datasets_from_configs
import logging

logger = logging.getLogger(__name__)


def _count_rows(ds: DatasetLike) -> int:
    """Count total rows in a dataset or dataset dict."""
    if isinstance(ds, DatasetDict):
        return sum(len(split) for split in ds.values())
    return len(ds)


def _dataset_out_dir(shard_idx: int, ds_config: BaseDataLoaderConfig) -> str:
    """Get output directory for a shard of a dataset."""
    return os.path.join(ds_config.output_dir, f"shard_{shard_idx}")


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


def _get_retry_count(shard_dir: str) -> int:
    """Get retry count for a shard from retry marker file."""
    retry_file = os.path.join(shard_dir, ".retry_count")
    if not os.path.exists(retry_file):
        return 0
    try:
        with open(retry_file, "r") as f:
            return int(f.read().strip())
    except (ValueError, IOError):
        return 0


def _increment_retry_count(shard_dir: str) -> int:
    """Increment and write retry count for a shard."""
    count = _get_retry_count(shard_dir) + 1
    retry_file = os.path.join(shard_dir, ".retry_count")
    os.makedirs(shard_dir, exist_ok=True)
    with open(retry_file, "w") as f:
        f.write(str(count))
    return count


def _write_success_marker(shard_dir: str):
    """Write success marker file for a completed shard."""
    marker_file = os.path.join(shard_dir, ".SUCCESS")
    os.makedirs(shard_dir, exist_ok=True)
    with open(marker_file, "w") as f:
        f.write(f"completed_at: {datetime.now().isoformat()}\n")


def _write_failure_marker(shard_dir: str, error_msg: str):
    """Write failure marker file with error information."""
    marker_file = os.path.join(shard_dir, ".FAILED")
    os.makedirs(shard_dir, exist_ok=True)
    with open(marker_file, "w") as f:
        f.write(f"failed_at: {datetime.now().isoformat()}\n")
        f.write(f"error: {error_msg}\n")


def rewrite_batch(
    batch: Dict[str, List[Any]],
    mapper: MMIRAGEMapper,
    renderer: TemplateRenderer,
    image_base_path: str = None,
) -> Dict[str, List[Any]]:
    """Rewrite a batch of samples by applying transformations.

    Args:
        batch: Dictionary mapping column names to lists of values.
        mapper: MMIRAGEMapper for processing transformations.
        renderer: TemplateRenderer for generating output.
        image_base_path: Optional base directory for resolving relative image paths.

    Returns:
        Dictionary mapping output keys to lists of rendered values.

    Raises:
        ValueError: If variables are not computable given the configuration.
    """
    if not mapper.validate_vars():
        raise ValueError(
            "Uncomputable variables detected. Verify your configuration and make sure that there is no undefined variables"
        )

    batch_environment = mapper.rewrite_batch(batch, image_base_path)
    rendered_list = renderer.batch_render(batch_environment)
    return rendered_list


def main():
    """Process a single shard of the dataset.

    Loads configuration, datasets, processes the shard using MMIRAGE
    transformations (including multimodal), and saves the result to disk.
    """
    ap = argparse.ArgumentParser(
        "Process dataset shards using MMIRAGE with SGLang."
    )
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

    # Track shard directories for marker files
    shard_dirs = []

    try:
        ds_all = load_datasets_from_configs(datasets_config)
        total_rows = sum(_count_rows(ds) for ds in ds_all)

        ds_all_shard = [_shard_dataset(ds, num_shards, shard_id) for ds in ds_all]
        shard_rows = sum(_count_rows(ds) for ds in ds_all_shard)

        logger.info(
            f"Loaded {len(datasets_config)} dataset(s): {datasets_config} "
            f"→ {total_rows} total rows; this shard has {shard_rows} rows."
        )

        # Increment retry count for each shard directory
        for ds_config in datasets_config:
            shard_dir = _dataset_out_dir(shard_id, ds_config)
            retry_count = _increment_retry_count(shard_dir)
            shard_dirs.append(shard_dir)
            if retry_count > 1:
                logger.info(f"Retry attempt #{retry_count} for shard {shard_id}")

        mapper = MMIRAGEMapper(
            cfg.processors, processing_params.inputs, processing_params.outputs
        )
        renderer = TemplateRenderer(processing_params.output_schema)
        ds_processed_all: List[DatasetLike] = []
        for ds_idx, ds_shard in enumerate(ds_all_shard):
            ds_config = datasets_config[ds_idx]
            remove_columns = _remove_columns(ds_shard, processing_params.remove_columns)
            ds_processed = ds_shard.map(
                rewrite_batch,
                batched=True,
                batch_size=loading_params.get_batch_size(),
                load_from_cache_file=False,
                desc=f"Shard {shard_id}/{num_shards - 1} dataset {ds_idx}",
                fn_kwargs={"mapper": mapper, "renderer": renderer, "image_base_path": ds_config.image_base_path},
                remove_columns=remove_columns,
            )
            ds_processed_all.append(ds_processed)

        for ds_config, ds_processed in zip(datasets_config, ds_processed_all):
            out_dir = _dataset_out_dir(shard_id, ds_config)
            os.makedirs(out_dir, exist_ok=True)
            ds_processed.save_to_disk(out_dir)
            logger.info(f"✅ Saved dataset in: {out_dir}")

        # Write success markers for all shards
        for shard_dir in shard_dirs:
            _write_success_marker(shard_dir)
            logger.info(f"✅ Shard {shard_id} completed successfully")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"❌ Shard {shard_id} failed: {error_msg}")
        logger.error(traceback.format_exc())
        
        # Write failure markers for all shards
        for shard_dir in shard_dirs:
            _write_failure_marker(shard_dir, error_msg)
        
        # Re-raise to ensure non-zero exit code
        sys.exit(1)


if __name__ == "__main__":
    main()
