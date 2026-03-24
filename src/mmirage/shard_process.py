"""Main script for processing dataset shards with MMIRAGE.

Supports both text-only and multimodal (vision-language) processing.
"""

import argparse
import logging
import sys
import traceback
from typing import Any, Dict, List

from mmirage.config.utils import load_mmirage_config
from mmirage.core.loader.base import DatasetLike
from mmirage.core.loader.utils import load_datasets_from_configs
from mmirage.core.process.mapper import MMIRAGEMapper
from mmirage.core.writer.renderer import TemplateRenderer
from mmirage.shard_utils import (
    _cleanup_old_shard_data,
    _count_rows,
    _dataset_out_dir,
    _mark_failure,
    _mark_running,
    _mark_success,
    _remove_columns,
    _save_dataset_atomic,
    _shard_dataset,
    _shard_state_dir,
)

logger = logging.getLogger(__name__)


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
    """
        Process a single shard of the dataset.
        Loads configuration, datasets, processes the shard using MMIRAGE
        transformations (including multimodal), and saves the result to disk.
    """
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
    last_shard_id = num_shards - 1

    if not (0 <= shard_id < num_shards):
        raise ValueError(f"Invalid shard_id={shard_id}, num_shards={num_shards}")

    state_dir = _shard_state_dir(shard_id, loading_params.get_state_root())

    try:
        retry_count = _mark_running(state_dir, shard_id, datasets_config)
        logger.info(f"Starting shard {shard_id}/{last_shard_id} (attempt #{retry_count})")

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
            if processing_params.remove_columns:
                remove_columns = _remove_columns(ds_shard)
            else:
                remove_columns = []

            logger.info(
                f"Processing dataset {ds_idx} for shard {shard_id}: "
                f"path={ds_config.path}, output_dir={ds_config.output_dir}"
            )

            ds_processed = ds_shard.map(
                rewrite_batch,
                batched=True,
                batch_size=loading_params.get_batch_size(),
                load_from_cache_file=False,
                desc=f"Shard {shard_id}/{last_shard_id} dataset {ds_idx}",
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